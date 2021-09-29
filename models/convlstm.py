import torch.nn as nn
import torch


# https://github.com/ndrplz/ConvLSTM_pytorch
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size # a tuple
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        # i门，f门，o门，g门放在一起计算，然后在split开
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state # 每个timestamp包含两个状态张量：h和c

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined) # i门，f门，o门，g门放在一起计算，然后在split开
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # torch.nn.Sigmoid()(input)等价于torch.sigmoid(input)，nn.functional中的函数仅仅定义了一些具体的基本操作，
        # 不能构成PyTorch中的一个layer。orch.nn.Sigmoid()(input)和torch.sigmoid(input)可以视为已经封装好的类，
        # 而nn.functional.sigmoid只是一个运算的方法，仅仅执行sigmoid的运算。

        c_next = f * c_cur + i * g  # c状态张量更新
        h_next = o * torch.tanh(c_next) # h状态张量更新

        return h_next, c_next # 输出当前timestamp的两个状态张量

    def init_hidden(self, batch_size, image_size):
        """
        初始状态张量初始化.第一个timestamp的状态张量0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels # h,c两个状态张量的通道数，可以是一个列表
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other # 卷积层的层数，需要与len(hidden_dim)相等
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers # 是否返回所有lstm层的h状态
        Note: Will do same padding. # 相同的卷积核尺寸，相同的padding尺寸
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W # 需要是5维的
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
        # 返回的是两个列表：layer_output_list，last_state_list
            0 - layer_output_list is the list of lists of length T of each output
            单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
            双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=False, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers) # 转为列表
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers) # 转为列表
        if not len(kernel_size) == len(hidden_dim) == num_layers: # 判断一致性
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = [] #为了储存每一层的参数尺寸
        for i in range(0, self.num_layers): # 多层LSTM设置
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # 注意这里利用lstm单元得出到了输出h，h再作为下一层的输入，依次得到每一层的数据维度并储存
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            
        #将上面循环得到的每一层的参数尺寸/维度，储存在self.cell_list中，后面会用到

        self.cell_list = nn.ModuleList(cell_list)
        # 把定义的多个LSTM层串联成网络模型 
        # 注意这里用了ModuLelist函数，模块化列表

    def forward(self, input_tensor, hidden_state=None):
        # 这里forward有两个输入参数，input_tensor 是一个五维数据
        #（t时间步,b输入batch_ize,c输出数据通道数--维度,h,w图像高乘宽）
        # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # 取出图片的数据，供下面初始化使用
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w)) 
            # 初始化hidd_state,利用后面和lstm单元中的初始化函数

        # 储存输出数据的列表
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1) # 根据输入张量获取lstm的长度
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers): # 逐层计算

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # 逐个timestamp计算
                # 每一个时间步都更新 h,c
                # 注意这里self.cell_list是一个模块(容器)
                # 每个timestamp, lstmcell的implement
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h) # 第 layer_idx 层的第t个stamp的输出状态

            layer_output = torch.stack(output_inner, dim=1) # 第 layer_idx 层的第所有stamp的输出状态串联
            cur_layer_input = layer_output # 准备第layer_idx+1层的输入张量

            layer_output_list.append(layer_output) # 当前层的所有timestamp的h状态的串联
            last_state_list.append([h, c]) # 当前层的最后一个stamp的输出状态的[h,c]

        # 选择要输出所有数据，还是输出最后一层的数据  
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
         """
        所有lstm层的第一个timestamp的输入状态0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    
'''
注意事项：在用以上代码构建convLSTM时，要注意hidden_dim,kernel_size,num_layers三个参数在LSTM层上的一致。即代码中的：

len(kernel_size) == len(hidden_dim) == num_layers
如果hidden_dim=64，kernel_size = (3,3), num_layers=3: 会搭建一个3层的convLSTM网络，每一层的隐状态都是64通道，kernel_size=(3,3)

如果hidden_dim=[64,128,256]，kernel_size = (3,3), num_layers=3: 会搭建一个3层的convLSTM网络，各层的隐状态通道数分别是[64,128,256]，所有层的kernel_size==(3,3)

如果hidden_dim=[64,128,256]，kernel_size = [(3,3),(5,5),(7,7)], num_layers=3: 会搭建一个3层的convLSTM网络，各层的隐状态通道数分别是[64,128,256]，各层的kernel_size分别是[(3,3),(5,5),(7,7)]

'''    

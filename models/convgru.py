'''
different versions of ConvGRU from various gits

'''
# v1: https://gist.github.com/halochou/acbd669af86ecb8f988325084ba7a749#file-conv_gru-py
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.init as init
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = Variable(torch.zeros(state_size)).cuda()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = f.sigmoid(self.update_gate(stacked_inputs))
        reset = f.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = f.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state
    
    
    
# v2: https://github.com/csd111/pytorch-convgru/blob/master/convgru.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from typing import Union, List, Tuple

# ------------------------------------------------------------------------------
# One-dimensional Convolution Gated Recurrent Unit
# ------------------------------------------------------------------------------


class ConvGRU1DCell(nn.Module):

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        recurrent_kernel_size: int = 3,
    ):
        """
        One-Dimensional Convolutional Gated Recurrent Unit (ConvGRU1D) cell.
        The input-to-hidden convolution kernel can be defined arbitrarily using
        the kernel_size, stride and padding parameters. The hidden-to-hidden
        convolution kernel is forced to be unit-stride, with a padding assuming
        an odd kernel size, in order to keep the number of features the same.
        The hidden state is initialized by default to a zero tensor of the
        appropriate shape.
        Arguments:
            input_channels {int} -- [Number of channels of the input tensor]
            hidden_channels {int} -- [Number of channels of the hidden state]
            kernel_size {int} -- [Size of the input-to-hidden convolving kernel]
        Keyword Arguments:
            stride {int} -- [Stride of the input-to-hidden convolution]
                             (default: {1})
            padding {int} -- [Zero-padding added to both sides of the input]
                              (default: {0})
            recurrent_kernel_size {int} -- [Size of the hidden-to-hidden
                                            convolving kernel] (default: {3})
        """
        super(ConvGRU1DCell, self).__init__()
        # ----------------------------------------------------------------------
        self.kernel_size = kernel_size
        self.stride = stride
        self.h_channels = hidden_channels
        self.padding_ih = padding
        self.padding_hh = recurrent_kernel_size // 2
        # ----------------------------------------------------------------------
        self.weight_ih = nn.Parameter(
            torch.ones(hidden_channels * 3, input_channels, kernel_size),
            requires_grad=True,
        )
        self.weight_hh = nn.Parameter(
            torch.ones(
                hidden_channels * 3, input_channels, recurrent_kernel_size
            ),
            requires_grad=True,
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(hidden_channels * 3), requires_grad=True
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(hidden_channels * 3), requires_grad=True
        )
        # ----------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_hh)
        init.xavier_uniform_(self.weight_ih)
        init.zeros_(self.bias_hh)
        init.zeros_(self.bias_ih)

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------

    def forward(self, input, hx=None):
        output_size = (
            int(
                (input.size(-1) - self.kernel_size + 2 * self.padding_ih)
                / self.stride
            )
            + 1
        )
        # Handle the case of no hidden state provided
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.h_channels, output_size, device=input.device
            )
        # Run the optimized convgru-cell
        return _opt_convgrucell_1d(
            input,
            hx,
            self.h_channels,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.stride,
            self.padding_ih,
            self.padding_hh,
        )


# ------------------------------------------------------------------------------
# Two-dimensional Convolution Gated Recurrent Unit
# ------------------------------------------------------------------------------


class ConvGRU2DCell(nn.Module):

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int]] = (0, 0),
        recurrent_kernel_size: Union[int, Tuple[int, int]] = (3, 3),
    ):
        """
        Two-Dimensional Convolutional Gated Recurrent Unit (ConvGRU2D) cell.
        The input-to-hidden convolution kernel can be defined arbitrarily using
        the kernel_size, stride and padding parameters. The hidden-to-hidden
        convolution kernel is forced to be unit-stride, with a padding assuming
        an odd kernel size in both dimensions, in order to keep the number of
        features the same.
        The hidden state is initialized by default to a zero tensor of the
        appropriate shape.
        Arguments:
            input_channels {int} -- [Number of channels of the input tensor]
            hidden_channels {int} -- [Number of channels of the hidden state]
            kernel_size {int or tuple} -- [Size of the input-to-hidden
                                           convolving kernel]
        Keyword Arguments:
            stride {int or tuple} -- [Stride of the input-to-hidden convolution]
                                      (default: {(1, 1)})
            padding {int or tuple} -- [Zero-padding added to both sides of the
                                       input] (default: {0})
            recurrent_kernel_size {int or tuple} -- [Size of the hidden-to-
                                                     -hidden convolving kernel]
                                                     (default: {(3, 3)})
        """
        super(ConvGRU2DCell, self).__init__()
        # ----------------------------------------------------------------------
        # Handle int to tuple conversion
        if isinstance(recurrent_kernel_size, int):
            recurrent_kernel_size = (recurrent_kernel_size,) * 2
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        if isinstance(padding, int):
            padding = (padding,) * 2
        # ----------------------------------------------------------------------
        # Save input parameters for later
        self.kernel_size = kernel_size
        self.stride = stride
        self.h_channels = hidden_channels
        self.padding_ih = padding
        self.padding_hh = (
            recurrent_kernel_size[0] // 2,
            recurrent_kernel_size[1] // 2,
        )
        # ----------------------------------------------------------------------
        # Initialize the convolution kernels
        self.weight_ih = nn.Parameter(
            torch.ones(
                hidden_channels * 3,
                input_channels,
                kernel_size[0],
                kernel_size[1],
            ),
            requires_grad=True,
        )
        self.weight_hh = nn.Parameter(
            torch.ones(
                hidden_channels * 3,
                input_channels,
                recurrent_kernel_size[0],
                recurrent_kernel_size[1],
            ),
            requires_grad=True,
        )
        self.bias_ih = nn.Parameter(
            torch.zeros(hidden_channels * 3), requires_grad=True
        )
        self.bias_hh = nn.Parameter(
            torch.zeros(hidden_channels * 3), requires_grad=True
        )
        # ----------------------------------------------------------------------
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.weight_hh)
        init.xavier_uniform_(self.weight_ih)
        init.zeros_(self.bias_hh)
        init.zeros_(self.bias_ih)

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------

    def forward(self, input, hx=None):
        output_size = (
            int(
                (input.size(-2) - self.kernel_size[0] + 2 * self.padding_ih[0])
                / self.stride[0]
            )
            + 1,
            int(
                (input.size(-1) - self.kernel_size[1] + 2 * self.padding_ih[1])
                / self.stride[1]
            )
            + 1,
        )
        # Handle the case of no hidden state provided
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.h_channels, *output_size, device=input.device
            )
        # Run the optimized convgru-cell
        return _opt_convgrucell_2d(
            input,
            hx,
            self.h_channels,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.stride,
            self.padding_ih,
            self.padding_hh,
        )


# --------------------------------------------------------------------------
# Torchscript optimized cell functions
# --------------------------------------------------------------------------


@torch.jit.script
def _opt_cell_end(hidden, ih_1, hh_1, ih_2, hh_2, ih_3, hh_3):
    z = torch.sigmoid(ih_1 + hh_1)
    r = torch.sigmoid(ih_2 + hh_2)
    n = torch.tanh(ih_3 + r * hh_3)
    out = (1 - z) * n + z * hidden
    return out


@torch.jit.script
def _opt_convgrucell_1d(
    inputs,
    hidden,
    channels: int,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    stride: int,
    pad1: int,
    pad2: int,
):
    ih_output = functional.conv1d(
        inputs, w_ih, bias=b_ih, stride=stride, padding=pad1
    )
    hh_output = functional.conv1d(
        hidden, w_hh, bias=b_hh, stride=1, padding=pad2
    )
    output = _opt_cell_end(
        hidden,
        torch.narrow(ih_output, 1, 0, channels),
        torch.narrow(hh_output, 1, 0, channels),
        torch.narrow(ih_output, 1, channels, channels),
        torch.narrow(hh_output, 1, channels, channels),
        torch.narrow(ih_output, 1, 2 * channels, channels),
        torch.narrow(hh_output, 1, 2 * channels, channels),
    )
    return output


@torch.jit.script
def _opt_convgrucell_2d(
    inputs,
    hidden,
    channels: int,
    w_ih,
    w_hh,
    b_ih,
    b_hh,
    stride: List[int],
    pad1: List[int],
    pad2: List[int],
):
    ih_output = functional.conv2d(
        inputs, w_ih, bias=b_ih, stride=stride, padding=pad1
    )
    hh_output = functional.conv2d(
        hidden, w_hh, bias=b_hh, stride=1, padding=pad2
    )
    output = _opt_cell_end(
        hidden,
        torch.narrow(ih_output, 1, 0, channels),
        torch.narrow(hh_output, 1, 0, channels),
        torch.narrow(ih_output, 1, channels, channels),
        torch.narrow(hh_output, 1, channels, channels),
        torch.narrow(ih_output, 1, 2 * channels, channels),
        torch.narrow(hh_output, 1, 2 * channels, channels),
    )
    return output


# v3: https://gitee.com/linhongxiang/ConvLSTM-PyTorch/blob/master/ConvRNN.py
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ConvRNN.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   convrnn cell
'''

import torch
import torch.nn as nn


class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # seq_len=10 for moving_mnist
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1]).cuda()
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
        return torch.stack(output_inner), htnext


class CLSTM_cell(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      4 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0],
                             self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],
                                self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy
        return torch.stack(output_inner), (hy, cy)
    
    
# v4 (same as v1): https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


# v5: csdn
# for 1d GRUCell
# https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        # 公式1
        resetgate = F.sigmoid(i_r + h_r)
        # 公式2
        inputgate = F.sigmoid(i_i + h_i)
        # 公式3
        newgate = F.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)
        
        
        return hy

# 2d GRUCell
class GRUConvCell(nn.Module):

    def __init__(self, input_channel, output_channel):

        super(GRUConvCell, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size=3, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size=3, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        self.activation = nn.Tanh()

	# 公式1，2
    def gates(self, x, h):

        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = torch.sigmoid(rn)
        uns = torch.sigmoid(un)
        return rns, uns

    # 公式3
    def output(self, x, h, r, u):

        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):

        N, C, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
	    
	    # 公式4
        return u * h + (1 - u) * y



class GRUNet(nn.Module):

    def __init__(self, hidden_size=64):

        super(GRUNet,self).__init__()

        self.gru_1 = GRUConvCell(input_channel=4,          output_channel=hidden_size)
        self.gru_2 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)
        self.gru_3 = GRUConvCell(input_channel=hidden_size,output_channel=hidden_size)

        self.fc = nn.Conv2d(in_channels=hidden_size,out_channels=1,kernel_size=3,padding=1)

    def forward(self, x, h):

        if h is None:
            h = [None,None,None]

        h1 = self.gru_1( x,h[0])
        h2 = self.gru_2(h1,h[1])
        h3 = self.gru_3(h2,h[2])

        o = self.fc(h3)

        return o,[h1,h2,h3]
 

if __name__ == '__main__':

    from utils import *
    
    device = 'cuda'

    x = torch.rand(1,1,10,20).to(device)

    grunet=GRUNet()
    grunet=grunet.to(device)
    grunet.eval()

    h = None
    o,h_n = grunet(x,h)



import torch
import torch.nn as nn
import torch.nn.functional as f
# import torch.nn.init as init



# from .convgru_unfolded import ConvGRUCell_layer
class ConvGRUCell_layer(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size=3):

        super(ConvGRUCell_layer, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # padding=1

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, 
                                   kernel_size=self.kernel_size, padding=self.padding)
                                    # bias=True, bias=self.bias
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, 
                                     kernel_size=self.kernel_size, padding=self.padding)
                                    # bias=True, bias=self.bias
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-6, True)

        self.activation = nn.Tanh()

	# update gate and reset gate, formula 1 and 2
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

    # new hidden state, formula 3
    def output(self, x, h, r, u):

        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):

        N, C, H, W = x.shape
        HC = self.output_channel
        if (h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
	    
	    # final hidden state ht, formula 4
        return u * h + (1 - u) * y
    
    
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Any, Optional, List

__all__ = ['RLA_MobileNetV2', 'rla_mobilenetv2', 
           'rla_mobilenetv2_k6', 'rla_mobilenetv2_k12', 'rla_mobilenetv2_k32'
           ]


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def conv_out(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

def recurrent_conv(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        rla_channel: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        hidden_rla = hidden_dim + rla_channel
        
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv1x1 = None
        if expand_ratio != 1:
            self.conv1x1 = ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
        
        layers: List[nn.Module] = []
        # if expand_ratio != 1:
        #     # pw
        #     layers.append(ConvBNReLU(inp + rla_channel, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_rla, hidden_rla, stride=stride, groups=hidden_rla, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_rla, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

        self.averagePooling = None
        if self.stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))
        
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # keep a copy of x 
        identity = x

        if self.conv1x1 is not None:
            x = self.conv1x1(x)
        
        # get concatenation of x & h after expansion
        x = torch.cat((x, h), dim=1)
        y = self.conv(x)
        
        if self.use_res_connect:
            out = identity + y
        else:
            out = y
        
        if self.averagePooling is not None:
            h = self.averagePooling(h)
            
        return out, y, h


class RLA_MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        rla_channel: int = 32,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(RLA_MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # features: List[nn.Module] = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        self.conv1 = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)
        
        self.newcell = [0]
        for i in range(1, len(inverted_residual_setting)):
            if inverted_residual_setting[i][3] == 2:
                self.newcell.append(i)
        # newcell = [0, 1, 2, 3, 5]
        
        # placeholder for layers in stages
        num_stages = len(inverted_residual_setting)
        stages = [None] * num_stages        # List(List)
        stage_bns = [None] * num_stages     # List(List)
        conv_outs = [None] * num_stages     # List
        recurrent_convs = [recurrent_conv(rla_channel, rla_channel)] * len(self.newcell)   # List
        
        # building inverted residual blocks
        j = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            stages[j] = []  # to be appended
            stage_bns[j] = nn.ModuleList([norm_layer(rla_channel) for _ in range(n)])
            conv_outs[j] = conv_out(output_channel, rla_channel)
            for i in range(n):
                stride = s if i == 0 else 1
                stages[j].append(block(input_channel, output_channel, stride, expand_ratio=t, rla_channel=rla_channel, norm_layer=norm_layer))
                input_channel = output_channel
            stages[j] = nn.ModuleList(stages[j])
            j += 1
        
        # make it nn.Sequential
        # self.features = nn.Sequential(*features)
        self.stages = nn.ModuleList(stages)
        self.conv_outs = nn.ModuleList(conv_outs)
        self.recurrent_convs = nn.ModuleList(recurrent_convs)
        self.stage_bns = nn.ModuleList(stage_bns)
        self.rla_channel = rla_channel
        self.flops = False
        
        # building last several layers (conv1x1-BN-ReLU: 320 -> 1280)
        # features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.conv2 = ConvBNReLU(input_channel + rla_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        self.bn2 = norm_layer(rla_channel)
        self.relu = nn.ReLU6(inplace=True)
        self.tanh = nn.Tanh()
        
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        
        # first layer
        x = self.conv1(x)
        
        # initialize h
        batch, _, height, width = x.size()
        if self.flops: # flops = True, then we compute the flops and params of the model
            h = torch.zeros(batch, self.rla_channel, height, width)
        else:
            h = torch.zeros(batch, self.rla_channel, height, width, device='cuda')
        
        # stages
        # x, h = self.features(x, h)
        j = 0
        k = -1
        for stage, bns, conv_out in zip(self.stages, self.stage_bns, self.conv_outs):
            if j in self.newcell:   # [0, 1, 2, 3, 5] -> [0, 1, 2, 3, 4]
                k += 1
            recurrent_conv = self.recurrent_convs[k]
            for layer, bn in zip(stage, bns):
                x, y, h = layer(x, h)
                
                # RLA module updates
                y_out = conv_out(y)
                h = h + y_out
                h = bn(h)
                h = self.tanh(h)
                h = recurrent_conv(h)
            j += 1
        
        h = self.bn2(h)
        h = self.relu(h)
        
        x = torch.cat((x, h), dim=1)
        # last several layers (conv1x1-BN-ReLU: 320 -> 1280)
        x = self.conv2(x)
        
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


#=========================== available models ============================
# import torchvision.models as models
# rla_models = dict(
#     rla_mobilenetv2_k6 = RLA_MobileNetV2(rla_channel = 6),
#     rla_mobilenetv2_k12 = RLA_MobileNetV2(rla_channel = 12),
#     rla_mobilenetv2_k32 = RLA_MobileNetV2(rla_channel = 32),
# )


def rla_mobilenetv2(rla_channel=32):
    """ Constructs a RLA_MobileNetV2 model.
    default: 
        rla_channel = 32
    """
    print("Constructing rla_mobilenetv2......")
    model = RLA_MobileNetV2(rla_channel=rla_channel)
    return model

def rla_mobilenetv2_k6():
    """ Constructs a RLA_MobileNetV2 model.
    default: 
        rla_channel = 32
    """
    print("Constructing rla_mobilenetv2_k6......")
    model = RLA_MobileNetV2(rla_channel=6)
    return model

def rla_mobilenetv2_k12():
    """ Constructs a RLA_MobileNetV2 model.
    default: 
        rla_channel = 32
    """
    print("Constructing rla_mobilenetv2_k12......")
    model = RLA_MobileNetV2(rla_channel=12)
    return model

def rla_mobilenetv2_k32():
    """ Constructs a RLA_MobileNetV2 model.
    default: 
        rla_channel = 32
    """
    print("Constructing rla_mobilenetv2_k32......")
    model = RLA_MobileNetV2(rla_channel=32)
    return model


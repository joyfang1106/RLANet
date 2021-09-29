'''
RLA unshared version, us--unshared

failed verison

RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [128, 2048, 7, 7]], which is output 0 of CudnnBatchNormBackward, is at version 2; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

solved: change inplace=True --> inplace=False, += --> = +, change all the inplace operations
'''

import torch
import torch.nn as nn
from .eca_module import eca_layer
from .se_module import SELayer

# torch.autograd.set_detect_anomaly(True)


__all__ = [
            'RLAus_ResNet', 
            'rlaus_resnet50', # v1
            'rlaus_resnet101'
            ]

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}



# RLA channel k: rla_channel = 32 (default)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


#=========================== define bottleneck ============================
class RLAus_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 rla_channel=32, SE=False, ECA_size=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(RLAus_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
            
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes + rla_channel, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        # self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride
        
        self.conv_out = conv1x1(planes * self.expansion, rla_channel)
        self.recurrent_conv = conv3x3(rla_channel, rla_channel)
        self.bn_rla = norm_layer(rla_channel)
        self.tanh_rla = nn.Tanh()
        self.averagePooling = None
        if downsample is not None and stride != 1:
            self.averagePooling = nn.AvgPool2d((2, 2), stride=(2, 2))

        self.se = None
        if SE:
            self.se = SELayer(planes * self.expansion, reduction)
        
        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))

    def forward(self, x, h):
        identity = x
        
        x = torch.cat((x, h), dim=1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se != None:
            out = self.se(out)
            
        if self.eca != None:
            out = self.eca(out)
        
        y = out
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.averagePooling is not None:
            h = self.averagePooling(h)
            
        # RLA module updates (unshared module)
        y_out = self.conv_out(y)
        h = h + y_out
        h = self.bn_rla(h)
        h = self.tanh_rla(h)
        h = self.recurrent_conv(h)
        
        out = out + identity
        out = self.relu(out)

        return out, h
    
    
#=========================== define network ============================
class RLAus_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, 
                 rla_channel=32, SE=False, ECA=None, 
                 zero_init_last_bn=True, #zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(RLAus_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if ECA is None:
            ECA = [None] * 4
        elif len(ECA) != 4:
            raise ValueError("argument ECA should be a 4-element tuple, got {}".format(ECA))
        
        self.rla_channel = rla_channel
        self.flops = False
        # flops: whether compute the flops and params or not
        # when use paras_flops, set as True
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        stages = [None] * 4
        stages[0] = self._make_layer(block, 64, layers[0], 
                                       rla_channel=rla_channel, SE=SE, ECA_size=ECA[0])
        stages[1] = self._make_layer(block, 128, layers[1], 
                                       rla_channel=rla_channel, SE=SE, ECA_size=ECA[1], 
                                       stride=2, dilate=replace_stride_with_dilation[0])
        stages[2] = self._make_layer(block, 256, layers[2], 
                                       rla_channel=rla_channel, SE=SE, ECA_size=ECA[2], 
                                       stride=2, dilate=replace_stride_with_dilation[1])
        stages[3] = self._make_layer(block, 512, layers[3], 
                                       rla_channel=rla_channel, SE=SE, ECA_size=ECA[3], 
                                       stride=2, dilate=replace_stride_with_dilation[2])
        
        self.stages = nn.ModuleList(stages)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.tanh = nn.Tanh()
        self.bn2 = norm_layer(rla_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion + rla_channel, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_last_bn:
        # if zero_init_residual:
            for m in self.modules():
                if isinstance(m, RLAus_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, 
                    rla_channel, SE, ECA_size, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            rla_channel=rla_channel, SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                rla_channel=rla_channel, SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            
        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        batch, _, height, width = x.size()
        # self.rla_channel = rla_channel
        if self.flops: # flops = True, then we compute the flops and params of the model
            h = torch.zeros(batch, self.rla_channel, height, width)
        else:
            h = torch.zeros(batch, self.rla_channel, height, width, device='cuda')
        
        for layers in self.stages:
            for layer in layers:
                x, h = layer(x, h)

        h = self.bn2(h)
        h = self.relu(h) # tanh
        # h = self.tanh(h)
        x = torch.cat((x, h), dim=1)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def forward(self, x):
        return self._forward_impl(x)


#=========================== available models ============================
def rlaus_resnet50(rla_channel=32):
    """ Constructs a RLAus_ResNet-50 model.
    default: 
        num_classes=1000, rla_channel=32, SE=False, ECA=None
    ECA: a list of kernel sizes in ECA
    """
    print("Constructing rlaus_resnet50......")
    model = RLAus_ResNet(RLAus_Bottleneck, [3, 4, 6, 3])
    return model


def rlaus_resnet101(rla_channel=32):
    """ Constructs a RLAus_ResNet-101 model.
    default: 
        num_classes=1000, rla_channel=32, SE=False, ECA=None
    ECA: a list of kernel sizes in ECA
    """
    print("Constructing rlaus_resnet101......")
    model = RLAus_ResNet(RLAus_Bottleneck, [3, 4, 23, 3])
    return model



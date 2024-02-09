import math
import torch
import torch.nn as nn
import torch.nn.functional as F

backbone_selected_layers=[1, 2, 3]

class GhostModule(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, dw_size=3, ratio=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(GhostModule, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = None
        self.ratio = ratio
        self.dw_size = dw_size
        self.dw_dilation = (dw_size - 1) // 2
        self.init_channels = math.ceil(out_channels / ratio)
        self.new_channels = self.init_channels * (ratio - 1)
        
        self.conv1 = nn.Conv2d(self.in_channels, self.init_channels, kernel_size, self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.init_channels, self.new_channels, self.dw_size, 1, padding=int(self.dw_size/2), groups=self.init_channels)
        
        
        self.weight1 = nn.Parameter(torch.Tensor(self.init_channels, self.in_channels, kernel_size, kernel_size))
        self.bn1 = nn.BatchNorm2d(self.init_channels)
        if self.new_channels > 0:
            self.weight2 = nn.Parameter(torch.Tensor(self.new_channels, 1, self.dw_size, self.dw_size))
            self.bn2 = nn.BatchNorm2d(self.out_channels - self.init_channels)
        
        if bias:
            self.bias =nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
    
    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.new_channels > 0:
            nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, input): 
        x1 = self.conv1(input)
        if self.new_channels == 0:
            return x1
        x2 = self.conv2(x1)
        x2 = x2[:, :self.out_channels - self.init_channels, :, :]
        x = torch.cat([x1, x2], 1)
        return x


def conv3x3(in_planes, out_planes, stride=1, s=4, d=3):
    "3x3 convolution with padding"
    return GhostModule(in_planes, out_planes, kernel_size=3, dw_size=d, ratio=s,
                     stride=stride, padding=1, bias=False)



class GhostBottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = GhostModule(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = GhostModule(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = GhostModule(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    


class GhostResNet(nn.Module):

    def __init__(self, layers, block= GhostBottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inplanes = 64
        # (3, 550, 550) to (64, 138, 138) stage
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
      
        self.make_layer(block, 64, layers[0])              # stage 2
        self.make_layer(block, 128, layers[1], stride=2)   # stage 3
        self.make_layer(block, 256, layers[2], stride=2)   # stage 4
        self.make_layer(block, 512, layers[3], stride=2)   # stage 5


    def make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
        
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(planes * block.expansion))
            
        layers = [block(self.inplanes, planes, stride, downsample, self.norm_layer)]
        
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))
            
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)


    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)
        

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)

        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)

    def add_layer(self, conv_channels=1024, downsample=2, depth=1, block=GhostBottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self.make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)



def construct_backbone(cfg_backbone = GhostResNet ):
    # resnet101 has 3, 4, 23, 3 blocks for each stage
    backbone = cfg_backbone([3, 4, 23, 3])

    selected_layers=[1, 2, 3]
    num_layers = max(selected_layers) + 1

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone

from torchvision import models
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = construct_backbone().to(device)
summary(model, (3, 550, 550))
input=torch.randn(1, 3, 550, 550)
backbone=construct_backbone()(input)

print('ghost-resnet output feature :', len(backbone))
print('C2 output shape : ', backbone[0].shape)
print('C3 output shape : ', backbone[1].shape)
print('C4 output shape : ', backbone[2].shape)
print('C5 output shape : ', backbone[3].shape)
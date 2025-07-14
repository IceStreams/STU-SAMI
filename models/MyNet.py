'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-26 15:06:59
LastEditTime: 2025-07-14 16:16:07
Description: file function description
'''
import torch
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d,
)
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN
import torch.nn as nn
from typing import cast

def Conv3x3ReLUBNs(in_channels,
                   inner_channels,
                   num_convs):

    layers = [nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()
    )]
    layers += [nn.Sequential(
        nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()    
    ) for _ in range(num_convs - 1)]
    return nn.Sequential(*layers)

class changeBackbone(nn.Module):
    def __init__(self, backbone = 'resnet18'):
        super(changeBackbone, self).__init__()

        if backbone in ["resnet18", "resnet34"]:
            max_channels = 512
        elif backbone in ["resnet50", "resnet101"]:
            max_channels = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}.")

        model_fn = getattr(models, backbone)
        self.backbone = model_fn(pretrained=True)
        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )

    def forward(self, input):
        x = self.backbone.conv1(input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]

        fpn_features = self.fpn(OrderedDict({f"c{i + 2}": features[i] for i in range(4)}))
        features = [v for k, v in fpn_features.items()]
        return cast(Tensor, features)

class featureFused_1(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(featureFused_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels*4, out_channels=outchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
    
    def forward(self, input):
        _, _, h, w = input[0].shape
        x2 = input[0]
        x3 = nn.functional.interpolate(input[1], size=(h, w),mode='bilinear', align_corners=True)
        x4 = nn.functional.interpolate(input[2], size=(h, w),mode='bilinear', align_corners=True)
        x5 = nn.functional.interpolate(input[3], size=(h, w),mode='bilinear', align_corners=True)
        out = self.conv(torch.cat((x2, x3, x4, x5), dim=1))
        return out
    
class featureFused(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(featureFused, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
    
    def forward(self, input):
        _, _, h, w = input[0].shape
        x2 = input[0]
        x3 = nn.functional.interpolate(input[1], size=(h, w),mode='bilinear', align_corners=True)
        x4 = nn.functional.interpolate(input[2], size=(h, w),mode='bilinear', align_corners=True)
        x5 = nn.functional.interpolate(input[3], size=(h, w),mode='bilinear', align_corners=True)
        out = self.conv(x2 + x3 + x4 + x5)
        return out

class changeNet(nn.Module):
    def __init__(self, backbone = 'resnet18'):
        super(changeNet, self).__init__()
        self.backbone = changeBackbone(backbone)

        # get change feature
        self.compare_c2 = Conv3x3ReLUBNs(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c3 = Conv3x3ReLUBNs(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c4 = Conv3x3ReLUBNs(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c5 = Conv3x3ReLUBNs(in_channels=256, inner_channels=256, num_convs=2)
        # feature fusion
        self.fused = featureFused(inchannels=256, outchannels=256)
        # change head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        feature_x1 = self.backbone(input1)
        feature_x2 = self.backbone(input2)

        compare_out2 = self.compare_c2(torch.abs(feature_x1[0]-feature_x2[0]))
        compare_out3 = self.compare_c3(torch.abs(feature_x1[1]-feature_x2[1]))
        compare_out4 = self.compare_c4(torch.abs(feature_x1[2]-feature_x2[2]))
        compare_out5 = self.compare_c5(torch.abs(feature_x1[3]-feature_x2[3]))
        compare_out = [compare_out2, compare_out3, compare_out4, compare_out5]

        fuse_out = self.fused(compare_out)
        change_out = self.head(fuse_out)
        change_out = torch.clamp(change_out, 1e-6, 1-1e-6)

        return change_out
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = changeNet(backbone='resnet101').to(device)
    print(model)
    image1 = torch.randn(1, 3, 1024, 1024).to(device)
    image2 = torch.randn(1, 3, 1024, 1024).to(device)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
    with torch.no_grad():
        change_out = model.forward(image1, image2)
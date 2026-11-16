# Adapted from the original CENet implementation by Hui-Xian Cheng et al.:
#   https://github.com/huixiancheng/CENet
#   ("CENet: Toward Concise and Efficient LiDAR Semantic Segmentation for
#    Autonomous Driving", ICME 2022). All credit for the architecture and the
#   pretrained weights goes to the original authors; see their repository for
#   the license.

"""
CENet (ResNet_34 backbone) for adversarial LiDAR diffusion experiments.

Modified from the original: nn.LeakyReLU is replaced by nn.Hardswish in BasicConv2d
and BasicBlock, matching the activation the official SemanticKITTI checkpoints were
trained with. Without this change the weights still load, but predictions are wrong.

Input contract:
    [B, 5, 64, W]    channels = [range, x, y, z, signal/remission]
                     normalized with the CENet KITTI mean/std (per-channel).

Output contract:
    if aux_loss.use is True   -> list [main, aux1, aux2, aux3]  (softmax probs)
    if aux_loss.use is False  -> single tensor                   (softmax probs)
    main shape: [B, nclasses, 64, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True):
        super().__init__()
        self.relu = relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.relu:
            self.relu = nn.Hardswish()  # PATCHED: was nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, if_BN=None):
        super().__init__()
        self.if_BN = if_BN
        if self.if_BN:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        if self.if_BN:
            self.bn1 = norm_layer(planes)
        self.relu = nn.Hardswish()  # PATCHED: was nn.LeakyReLU()
        self.conv2 = conv3x3(planes, planes)
        if self.if_BN:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.if_BN:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.if_BN:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet_34(nn.Module):
    def __init__(self, nclasses, params, block=BasicBlock, layers=[3, 4, 6, 3], if_BN=True,
                 zero_init_residual=False, norm_layer=None, groups=1, width_per_group=64):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.if_BN = if_BN
        self.dilation = 1
        self.aux = params["train"]["aux_loss"]["use"]

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = BasicConv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(128, 128, kernel_size=3, padding=1)

        self.inplanes = 128

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv_1 = BasicConv2d(640, 256, kernel_size=3, padding=1)
        self.conv_2 = BasicConv2d(256, 128, kernel_size=3, padding=1)
        self.semantic_output = nn.Conv2d(128, nclasses, 1)

        if self.aux:
            self.aux_head1 = nn.Conv2d(128, nclasses, 1)
            self.aux_head2 = nn.Conv2d(128, nclasses, 1)
            self.aux_head3 = nn.Conv2d(128, nclasses, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.if_BN:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, if_BN=self.if_BN))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                if_BN=self.if_BN))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        res_2 = F.interpolate(x_2, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_3 = F.interpolate(x_3, size=x.size()[2:], mode='bilinear', align_corners=True)
        res_4 = F.interpolate(x_4, size=x.size()[2:], mode='bilinear', align_corners=True)
        res = [x, x_1, res_2, res_3, res_4]

        out = torch.cat(res, dim=1)
        out = self.conv_1(out)
        out = self.conv_2(out)
        out = self.semantic_output(out)
        out = F.softmax(out, dim=1)

        if self.aux:
            res_2 = self.aux_head1(res_2)
            res_2 = F.softmax(res_2, dim=1)
            res_3 = self.aux_head2(res_3)
            res_3 = F.softmax(res_3, dim=1)
            res_4 = self.aux_head3(res_4)
            res_4 = F.softmax(res_4, dim=1)
            return [out, res_2, res_3, res_4]
        else:
            return out

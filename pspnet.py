import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict

affine_par = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv to reduce number of feature maps
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias = False, dilation= dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        # 1x1 conv to increase number of feature maps
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU()
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
        out = self.relu(out)

        # if downsample is not None, then x need change it's size
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PSP_Module(nn.Module):
    def __init__(self, size_series):
        super(PSP_Module, self).__init__()
        self.pool2d_list = nn.ModuleList([self._make_pool(size) for size in size_series])

    def _make_pool(self, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(2048, 512, kernel_size=1, stride=1, bias=False)
        # we get some trouble in bn about affine
        bn = nn.BatchNorm2d(512)
        relu = nn.ReLU()
        interp_layer = nn.Upsample(size= size, mode='bilinear')
        return nn.Sequential(pool, conv, bn, relu)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool1 = F.upsample(input=self.pool2d_list[0](x), size=(h, w), mode='bilinear')
        pool2 = F.upsample(input=self.pool2d_list[1](x), size=(h, w), mode='bilinear')
        pool3 = F.upsample(input=self.pool2d_list[2](x), size=(h, w), mode='bilinear')
        pool6 = F.upsample(input=self.pool2d_list[3](x), size=(h, w), mode='bilinear')
        out = torch.cat((pool1, pool2, pool3, pool6, x), dim=1)
        return out


class Classification_Module(nn.Module):
    def __init__(self, num_classes):
        super(Classification_Module, self).__init__()
        self.conv1 = nn.Conv2d(4096, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(512, affine=affine_par)
        for i in self.bn.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class Classification(nn.Sequential):
    def __init__(self, num_feature, num_classes, input_size):
        super(Classification, self).__init__()
        self.classificiation = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_feature, (num_feature/8), kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn0', nn.BatchNorm2d(num_feature/8)),
            ('relu0', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout2d(p=0.1)),
            ('conv', nn.Conv2d(num_feature/8, num_classes, kernel_size=1, stride=1, bias=True)),
            ('interp', nn.Upsample(size=input_size, mode='bilinear'))
        ]))


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, size = 64):
        # number of input feature map in initial block
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_psp_layer(PSP_Module, [1, 2, 3, 6])
        self.layer6 = self._make_pre_layer(Classification_Module, num_classes)
        self.conv_aux = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1)
        self.conv_aux_interp = nn.Upsample(size=(512, 512), mode='bilinear')
        self.interp = nn.Upsample(size=(512, 512), mode='bilinear')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine= affine_par))

        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        # number of featur maps has changed
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_psp_layer(self, block, size_series):
        return block(size_series)

    def _make_pre_layer(self, block, num_classes):
        return block(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_aux = self.conv_aux(x)
        x_aux = self.conv_aux_interp(x_aux)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.interp(x)

        return x, x_aux


def PSPNet(num_classes=5):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


if __name__ == '__main__':
    model = PSPNet(5)
    name = model.state_dict().copy()
    for i in name:
        print(i)

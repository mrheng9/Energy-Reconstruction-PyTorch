"""
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1, use_residual=False):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion
        self.use_residual = use_residual and in_channels == out_channels and stride == 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out

def repetitive_bottleneck(x, num_blocks, in_channels, out_channels, expansion):
    layers = []
    layers.append(Bottleneck(in_channels, out_channels, expansion, stride=1, use_residual=False))
    for _ in range(num_blocks - 1):
        layers.append(Bottleneck(out_channels, out_channels, expansion, stride=1, use_residual=True))
    return nn.Sequential(*layers)

class SubNet(nn.Module):
    def __init__(self, filter_number):
        super(SubNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, filter_number, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_number),
            nn.ReLU6(inplace=True)
        )
        self.b1 = Bottleneck(filter_number, 16, expansion=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.b2 = repetitive_bottleneck(None, num_blocks=2, in_channels=16, out_channels=24, expansion=6)

    def forward(self, x):
        x = self.initial(x)
        x = self.b1(x)
        x = self.pool(x)
        x = self.b2(x)
        return x

class MobileNetRegression(nn.Module):
    def __init__(self, input_scaling=1.0, dropout=0.4):# May 7th, change the dropout from 0.3 to 0.4
        super(MobileNetRegression, self).__init__()
        self.input_scaling = input_scaling

        self.subnet_x = SubNet(filter_number=32)
        self.subnet_y = SubNet(filter_number=32)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone1 = repetitive_bottleneck(None, 3, 24, 32, expansion=6)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone2 = repetitive_bottleneck(None, 4, 32, 48, expansion=6)
        self.backbone3 = repetitive_bottleneck(None, 3, 48, 64, expansion=6)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone4 = repetitive_bottleneck(None, 3, 64, 96, expansion=6)
        self.bottleneck_last = Bottleneck(96, 160, expansion=6)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 1024),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )

    def forward(self, input_x, input_y):
        if self.input_scaling != 1.0:
            input_x = input_x * self.input_scaling
            input_y = input_y * self.input_scaling

        # ✅ 跳过输入中为全零（空事件）的样本
        mask = torch.logical_or(input_x.abs().sum(dim=(1, 2, 3)) > 0, input_y.abs().sum(dim=(1, 2, 3)) > 0)
        if mask.sum() == 0:
            return torch.zeros((0, 1), device=input_x.device)

        input_x = input_x[mask]
        input_y = input_y[mask]

        x_feat = self.subnet_x(input_x)
        y_feat = self.subnet_y(input_y)

        top = torch.max(x_feat, y_feat)
        top = self.pool1(top)
        top = self.backbone1(top)
        top = self.pool2(top)
        top = self.backbone2(top)
        top = self.backbone3(top)
        top = self.pool3(top)
        top = self.backbone4(top)
        top = self.bottleneck_last(top)

        top = self.global_avgpool(top)
        out = self.regressor(top)
        return out
"""

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride=1, use_residual=False):
        super(Bottleneck, self).__init__()
        mid_channels = in_channels * expansion
        self.use_residual = use_residual and in_channels == out_channels and stride == 1

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride,
                      padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out

def repetitive_bottleneck(x, num_blocks, in_channels, out_channels, expansion):
    layers = []
    layers.append(Bottleneck(in_channels, out_channels, expansion, stride=1, use_residual=False))
    for _ in range(num_blocks - 1):
        layers.append(Bottleneck(out_channels, out_channels, expansion, stride=1, use_residual=True))
    return nn.Sequential(*layers)

class SubNet(nn.Module):
    def __init__(self, filter_number):
        super(SubNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(1, filter_number, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_number),
            nn.ReLU6(inplace=True)
        )
        self.b1 = Bottleneck(filter_number, 16, expansion=1, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.b2 = repetitive_bottleneck(None, num_blocks=2, in_channels=16, out_channels=24, expansion=6)

    def forward(self, x):
        x = self.initial(x)
        x = self.b1(x)
        x = self.pool(x)
        x = self.b2(x)
        return x

class MobileNetRegression(nn.Module):
    def __init__(self, input_scaling=1.0, dropout=0.4):
        super(MobileNetRegression, self).__init__()
        self.input_scaling = input_scaling

        self.subnet_x = SubNet(filter_number=32)
        self.subnet_y = SubNet(filter_number=32)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone1 = repetitive_bottleneck(None, 3, 24, 32, expansion=6)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone2 = repetitive_bottleneck(None, 4, 32, 48, expansion=6)
        self.backbone3 = repetitive_bottleneck(None, 3, 48, 64, expansion=6)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.backbone4 = repetitive_bottleneck(None, 3, 64, 96, expansion=6)
        self.bottleneck_last = Bottleneck(96, 160, expansion=6)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 1024),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )

    def forward(self, input_x, input_y):
        if self.input_scaling != 1.0:
            input_x = input_x * self.input_scaling
            input_y = input_y * self.input_scaling

        mask = torch.logical_or(input_x.abs().sum(dim=(1, 2, 3)) > 0, input_y.abs().sum(dim=(1, 2, 3)) > 0)
        if mask.sum() == 0:
            return torch.zeros((0, 1), device=input_x.device)

        input_x = input_x[mask]
        input_y = input_y[mask]

        x_feat = self.subnet_x(input_x)
        y_feat = self.subnet_y(input_y)

        top = torch.max(x_feat, y_feat)
        top = self.pool1(top)
        top = self.backbone1(top)
        top = self.pool2(top)
        top = self.backbone2(top)
        top = self.backbone3(top)
        top = self.pool3(top)
        top = self.backbone4(top)
        top = self.bottleneck_last(top)

        top = self.global_avgpool(top)
        out = self.regressor(top)
        return out

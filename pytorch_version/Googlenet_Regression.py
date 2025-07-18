import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red5x5, out5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], 1)

class SubNetGoogLeNet(nn.Module):
    def __init__(self, in_channels=1):
        super(SubNetGoogLeNet, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        return x

class GoogLeNetRegression(nn.Module):
    def __init__(self, input_scaling=1.0, dropout=0.4):
        super(GoogLeNetRegression, self).__init__()
        self.input_scaling = input_scaling

        self.subnet_x = SubNetGoogLeNet()
        self.subnet_y = SubNetGoogLeNet()

        self.shared_inception = nn.Sequential(
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
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
        top = self.shared_inception(top)
        out = self.regressor(top)
        return out

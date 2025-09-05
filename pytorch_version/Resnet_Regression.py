import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, skipconnection=True):
        super(ResidualBlock, self).__init__()
        self.skipconnection = skipconnection
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Handle dimension changes for the shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.skipconnection:
            out += self.shortcut(residual)
            
        out = self.relu(out)
        
        return out

class SubNetResNet(nn.Module):
    def __init__(self, in_channels=1, filter_number=64, num_blocks=2, pooling='max', skipconnection=True):
        super(SubNetResNet, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, filter_number, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(filter_number),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Create pooling layers based on parameter
        if pooling == 'average':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'fullyconv':
            self.pool1 = nn.Conv2d(filter_number, filter_number, kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.Conv2d(filter_number*2, filter_number*2, kernel_size=3, stride=2, padding=1)
        else:  # default is max pooling
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Create residual blocks for each level
        self.layer1 = self._make_layer(filter_number, filter_number, num_blocks, skipconnection)
        self.layer2 = self._make_layer(filter_number, filter_number*2, num_blocks, skipconnection)
        self.layer3 = self._make_layer(filter_number*2, filter_number*4, num_blocks, skipconnection)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, skipconnection):
        layers = []
        # First block may need to handle dimension changes
        layers.append(ResidualBlock(in_channels, out_channels, stride=1, skipconnection=skipconnection))
        
        # Subsequent blocks maintain dimensions
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, skipconnection=skipconnection))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        
        x = self.layer1(x)
        x = self.pool1(x)
        
        x = self.layer2(x)
        x = self.pool2(x)
        
        x = self.layer3(x)
        
        return x

class ResNetRegression(nn.Module):
    def __init__(self, filter_number=64, num_blocks=2, num_top_blocks=2, pooling='max', 
                 skipconnection=True, input_scaling=1.0, dropout=0.4):
        super(ResNetRegression, self).__init__()
        self.input_scaling = input_scaling
        
        # Create subnets for processing x and y inputs
        self.subnet_x = SubNetResNet(in_channels=1, filter_number=filter_number, 
                                    num_blocks=num_blocks, pooling=pooling,
                                    skipconnection=skipconnection)
        self.subnet_y = SubNetResNet(in_channels=1, filter_number=filter_number,
                                    num_blocks=num_blocks, pooling=pooling,
                                    skipconnection=skipconnection)
        
        # Create top shared layers
        top_layers = []
        in_channels = filter_number * 4  # Output channels from subnet's last layer
        out_channels = filter_number * 8
        
        for _ in range(num_top_blocks):
            top_layers.append(ResidualBlock(in_channels, out_channels, 
                                           skipconnection=skipconnection))
            in_channels = out_channels
        
        top_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.shared_layers = nn.Sequential(*top_layers)
        
        # Create regressor
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1)
        )
    
    def forward(self, input_x, input_y):
        # Scale inputs if needed
        if self.input_scaling != 1.0:
            input_x = input_x * self.input_scaling
            input_y = input_y * self.input_scaling
        
        # Filter out empty samples
        mask = torch.logical_or(input_x.abs().sum(dim=(1, 2, 3)) > 0, 
                               input_y.abs().sum(dim=(1, 2, 3)) > 0)
        if mask.sum() == 0:
            return torch.zeros((0, 1), device=input_x.device)
        
        input_x = input_x[mask]
        input_y = input_y[mask]
        
        # Process through subnets
        x_feat = self.subnet_x(input_x)
        y_feat = self.subnet_y(input_y)
        
        # Combine features (using max as in GoogLeNet implementation)
        top = torch.max(x_feat, y_feat)
        
        # Process through shared layers
        top = self.shared_layers(top)
        
        # Final regression
        out = self.regressor(top)
        
        return out
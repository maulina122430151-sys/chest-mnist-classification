# regnet_model.py

import torch
import torch.nn as nn
from math import log

class SE(nn.Module):
    """Squeeze-and-Excitation module."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, se_ratio=0.25):
        super().__init__()
        self.has_shortcut = (in_channels != out_channels) or (stride != 1)
        group_width = out_channels // groups
        
        # Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Conv 3x3 grouped
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Conv 1x1
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # SE module
        self.se = SE(out_channels, reduction=int(out_channels * se_ratio))
        
        if self.has_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        
        if self.has_shortcut:
            identity = self.shortcut(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class RegNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, init_channels=32, slope=0.25, groups=16):
        super().__init__()
        self.init_channels = init_channels
        
        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        # RegNet stages
        stage_depths = [2, 4, 6, 4]  # number of blocks in each stage
        channels = init_channels
        stages = []
        
        for i, num_blocks in enumerate(stage_depths):
            stage = []
            for j in range(num_blocks):
                out_channels = int(channels * (1 + slope))
                stride = 2 if j == 0 and i > 0 else 1
                stage.append(RegNetBlock(channels, out_channels, stride=stride, groups=groups))
                channels = out_channels
            stages.append(nn.Sequential(*stage))
            
        self.stages = nn.Sequential(*stages)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'RegNet' ---")
    
    model = RegNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'RegNet' berhasil.")
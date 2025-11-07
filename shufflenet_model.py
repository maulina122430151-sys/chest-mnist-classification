# shufflenet_model.py

import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.stride = stride
        assert stride in [1, 2]

        branch_features = out_channels // 2
        
        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(in_channels, in_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if self.stride > 1 else branch_features, branch_features, 
                     kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                        groups=in_channels, bias=bias)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, width_mult=2.0):
        super().__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        self.stage_out_channels = [int(x * width_mult) for x in self.stage_out_channels]
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.stage_out_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Building stages
        self.stages = []
        input_channels = self.stage_out_channels[1]
        
        stage_names = []
        for i, repeats in enumerate(self.stage_repeats):
            stage_name = f'stage{i + 2}'
            stage_names.append(stage_name)
            output_channels = self.stage_out_channels[i + 2]
            
            # First module uses stride=2
            self.stages.append(InvertedResidual(input_channels, output_channels, stride=2))
            
            # Remaining modules use stride=1
            for _ in range(repeats - 1):
                self.stages.append(InvertedResidual(output_channels, output_channels, stride=1))
                
            input_channels = output_channels
            
        self.stages = nn.Sequential(*self.stages)
        
        # Final conv
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, self.stage_out_channels[-1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.stage_out_channels[-1], 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'ShuffleNetV2 X2.0' ---")
    
    model = ShuffleNetV2(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, width_mult=2.0)
    print("Arsitektur Model:")
    print(model)
    
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print("Pengujian model 'ShuffleNetV2 X2.0' berhasil.")
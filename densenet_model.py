# densenet_model.py

import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # Similar pattern to ResNet's BasicBlock, but with dense connections
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        return torch.cat([x, out], 1)  # Dense connection - concatenate instead of add

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # Create dense layers similar to ResNet's make_layer pattern
        self.layers = self._make_layers()
        
    def _make_layers(self):
        layers = []
        channels = self.in_channels
        for _ in range(self.num_layers):
            layers.append(DenseLayer(channels, self.growth_rate))
            channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        # Similar pattern to ResNet's downsample
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        out = torch.relu(self.bn(x))
        out = self.conv(out)
        out = self.pool(out)
        return out

class DenseNet121(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, growth_rate=32):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution (similar to ResNet)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Dense layers with transitions (similar pattern to ResNet's layers)
        num_channels = 64
        
        # Dense block 1
        self.dense1 = DenseBlock(num_channels, num_layers=6, growth_rate=growth_rate)
        num_channels += 6 * growth_rate
        self.trans1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        # Dense block 2
        self.dense2 = DenseBlock(num_channels, num_layers=12, growth_rate=growth_rate)
        num_channels += 12 * growth_rate
        self.trans2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        # Dense block 3
        self.dense3 = DenseBlock(num_channels, num_layers=24, growth_rate=growth_rate)
        num_channels += 24 * growth_rate
        self.trans3 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        # Dense block 4 (final)
        self.dense4 = DenseBlock(num_channels, num_layers=16, growth_rate=growth_rate)
        num_channels += 16 * growth_rate
        
        # Final layers (similar to ResNet)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        # Initial convolution (similar to ResNet)
        out = torch.relu(self.bn1(self.conv1(x)))
        
        # Dense blocks with transitions
        out = self.dense1(out)
        out = self.trans1(out)
        
        out = self.dense2(out)
        out = self.trans2(out)
        
        out = self.dense3(out)
        out = self.trans3(out)
        
        out = self.dense4(out)
        
        # Final layers (similar to ResNet)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Menguji Model 'DenseNet121' ---")
    
    model = DenseNet121(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)
    
    # Pengujian dengan input dummy (sama seperti ResNet)
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    
    # Tambahan pengujian untuk memastikan dense connections
    print("\nPengujian Dense Connections:")
    test_input = torch.randn(1, IN_CHANNELS, 28, 28)
    with torch.no_grad():
        # Cek ukuran output setiap blok
        x = torch.relu(model.bn1(model.conv1(test_input)))
        print(f"Initial conv output shape: {x.shape}")
        
        x = model.dense1(x)
        print(f"After dense1 shape: {x.shape}")
        x = model.trans1(x)
        print(f"After trans1 shape: {x.shape}")
        
        x = model.dense2(x)
        print(f"After dense2 shape: {x.shape}")
        x = model.trans2(x)
        print(f"After trans2 shape: {x.shape}")
        
        x = model.dense3(x)
        print(f"After dense3 shape: {x.shape}")
        x = model.trans3(x)
        print(f"After trans3 shape: {x.shape}")
        
        x = model.dense4(x)
        print(f"After dense4 shape: {x.shape}")
    
    print("\nPengujian model 'DenseNet121' berhasil.")
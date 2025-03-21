import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet.model import resnet18, resnet34, resnet50

class CenterNetHead(nn.Module):
    def __init__(self, in_channels, num_classes, skip_channels=[512, 256, 128]):
        super().__init__()
        
        # Upsampling layers to increase feature map resolution
        self.deconv1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        
        # Lateral connections for skip features
        self.lateral1 = nn.Conv2d(skip_channels[0], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(skip_channels[1], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(skip_channels[2], 256, kernel_size=1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Head for heatmap prediction (object centers)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Head for size regression (width and height)
        self.wh_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 2, kernel_size=1)
        )
        
        # Head for local offset regression
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 2, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, skip_features):
        # First deconv + skip connection
        x = self.deconv1(x)
        skip1 = self.lateral1(skip_features[0])
        x = x + skip1
        x = F.relu(self.bn1(x), inplace=True)
        
        # Second deconv + skip connection
        x = self.deconv2(x)
        skip2 = self.lateral2(skip_features[1])
        x = x + skip2
        x = F.relu(self.bn2(x), inplace=True)
        
        # Third deconv + skip connection
        x = self.deconv3(x)
        skip3 = self.lateral3(skip_features[2])
        x = x + skip3
        x = F.relu(self.bn3(x), inplace=True)
        
        # Generate predictions with intermediate ReLUs
        x_hm = F.relu(self.heatmap_head[0](x), inplace=True)
        heatmap = self.heatmap_head[1](x_hm)
        
        x_wh = F.relu(self.wh_head[0](x), inplace=True)
        wh = self.wh_head[1](x_wh)
        
        x_off = F.relu(self.offset_head[0](x), inplace=True)
        offset = self.offset_head[1](x_off)
        
        return {
            'heatmap': torch.sigmoid(heatmap),  # Apply sigmoid for heatmap
            'wh': wh,
            'offset': offset
        }

class CenterNet(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=80):
        super().__init__()
        
        # Initialize backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(num_classes=1000)  # Using ImageNet pretrained settings
            backbone_channels = 512
            skip_channels = [256, 128, 64]  # Channels from different stages
        elif backbone == 'resnet34':
            self.backbone = resnet34(num_classes=1000)
            backbone_channels = 512
            skip_channels = [256, 128, 64]
        elif backbone == 'resnet50':
            self.backbone = resnet50(num_classes=1000)
            backbone_channels = 2048
            skip_channels = [1024, 512, 256]  # Channels from different stages
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Get the layers from ResNet
        self.conv1 = self.backbone.conv
        self.bn1 = self.backbone.bn
        self.maxpool = self.backbone.maxpool
        
        # Get the four stages of ResNet
        self.stage1 = self.backbone.resnet_layers[0]  # 1/4
        self.stage2 = self.backbone.resnet_layers[1]  # 1/8
        self.stage3 = self.backbone.resnet_layers[2]  # 1/16
        self.stage4 = self.backbone.resnet_layers[3]  # 1/32
        
        # Initialize CenterNet head
        self.head = CenterNetHead(backbone_channels, num_classes, skip_channels)
    
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # ResNet stages with skip connections
        s1 = self.stage1(x)      # 1/4 resolution
        s2 = self.stage2(s1)     # 1/8 resolution
        s3 = self.stage3(s2)     # 1/16 resolution
        s4 = self.stage4(s3)     # 1/32 resolution
        
        # Skip features from different stages (from higher to lower resolution)
        skip_features = [s3, s2, s1]
        
        # Generate predictions using head
        return self.head(s4, skip_features)

if __name__ == "__main__":
    # Create a dummy input tensor (batch_size=1, channels=3, height=512, width=512)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Create CenterNet model with ResNet18 backbone
    # Using 80 classes as in COCO dataset
    model = CenterNet(backbone='resnet18', num_classes=80)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Print output shapes
    print("Input shape:", dummy_input.shape)
    print("\nOutput shapes:")
    print("- Heatmap shape:", outputs['heatmap'].shape)  # Should be [1, 80, 128, 128]
    print("- Width/Height shape:", outputs['wh'].shape)  # Should be [1, 2, 128, 128]
    print("- Offset shape:", outputs['offset'].shape)  # Should be [1, 2, 128, 128]
    
    # The output is 128x128 because:
    # 1. Input is 512x512
    # 2. ResNet backbone has 5 stride-2 operations (one initial conv + 4 stages)
    # 3. CenterNet head has 3 upsampling layers (stride-2 each)
    # So final stride is: 2^5 / 2^3 = 2^2 = 4
    # Therefore 512/4 = 128 
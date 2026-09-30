import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class DualStreamEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # PyTorch 0.15+ uses weights= instead of pretrained=
        from torchvision.models import MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        
        self.rgb_backbone = mobilenet_v3_large(weights=weights)
        self.rgb_backbone.classifier = nn.Identity()
        
        self.ela_backbone = mobilenet_v3_large(weights=weights)
        self.ela_backbone.classifier = nn.Identity()
        
        self.rgb_cbam = CBAM(960)
        self.ela_cbam = CBAM(960)

    def forward(self, rgb, ela):
        rgb_feat = self.rgb_backbone(rgb)
        ela_feat = self.ela_backbone(ela)
        
        # Reshape for CBAM (batch_size, 960, 1, 1) to apply attention properly
        # Note: MobileNetV3 with Identity classifier returns (B, 960)
        # So we need to unsqueeze to make it a spatial tensor for CBAM
        rgb_feat = rgb_feat.unsqueeze(-1).unsqueeze(-1)
        ela_feat = ela_feat.unsqueeze(-1).unsqueeze(-1)
        
        rgb_feat = self.rgb_cbam(rgb_feat)
        ela_feat = self.ela_cbam(ela_feat)
        
        # Flatten back out
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        ela_feat = ela_feat.view(ela_feat.size(0), -1)
        
        return rgb_feat, ela_feat

class ODD2F(nn.Module):
    def __init__(self, num_classes=2): # Output size is 2
        super().__init__()
        self.encoder = DualStreamEncoder(pretrained=True)
        
        # The paper says 2 units (sigmoid) for binary classification
        self.fusion = nn.Sequential(
            # Since the output of the encoder is flattened, we don't need AdaptiveAvgPool2d or Flatten here
            nn.Linear(960 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Increased from 0.3 to stop overfitting
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # Increased from 0.3 to stop overfitting
            nn.Linear(256, num_classes),
            # nn.Sigmoid() - Usually handled by CrossEntropyLoss or BCEWithLogitsLoss
        )

    def forward(self, rgb, ela):
        rgb_feat, ela_feat = self.encoder(rgb, ela)
        fused = torch.cat([rgb_feat, ela_feat], dim=1)
        fused = self.fusion(fused)
        out = self.classifier(fused)
        return out

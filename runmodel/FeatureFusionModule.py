"""Legacy feature fusion block used by runmodel-local SPANet implementation."""

import torch.nn as nn

class FeatureFusionModule(nn.Module):
    """Fuse low-level features using attention generated from high-level features."""

    def __init__(self, low_channels, high_channels):
        """Initialize fusion block channel projections.

        Args:
            low_channels (int): Low-level feature channels.
            high_channels (int): High-level feature channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(low_channels // 4, 1)

        self.conv1 = nn.Conv2d(high_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, low_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        """Apply pooled high-level attention to low-level feature map.

        Args:
            low (torch.Tensor): Low-level feature tensor.
            high (torch.Tensor): High-level feature tensor.

        Returns:
            torch.Tensor: Fused feature tensor.
        """
        B, C_l, H_l, W_l = low.shape
        h = self.pool(high)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.sigmoid(self.conv2(h))
        attn = h.expand(-1, -1, H_l, W_l)
        fused = low * attn
        return fused
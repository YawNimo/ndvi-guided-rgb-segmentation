"""Feature fusion block used by SPANetFull decoder path."""

import torch.nn as nn


class FeatureFusionModule(nn.Module):
    """Fuse low-level features with high-level attention weights.

    The high-resolution low-level map is reweighted using a channel-attention signal
    produced from pooled high-level features.
    """

    def __init__(self, low_channels, high_channels):
        """Initialize the feature fusion module.

        Args:
            low_channels (int): Channel count of the low-level feature map.
            high_channels (int): Channel count of the high-level feature map.
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
        """Apply high-level channel attention to low-level features.

        Args:
            low (torch.Tensor): Low-level tensor of shape ``(N, C_low, H, W)``.
            high (torch.Tensor): High-level tensor of shape ``(N, C_high, h, w)``.

        Returns:
            torch.Tensor: Fused tensor with shape ``(N, C_low, H, W)``.
        """
        _, _, h_l, w_l = low.shape
        h = self.pool(high)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.sigmoid(self.conv2(h))
        attn = h.expand(-1, -1, h_l, w_l)
        fused = low * attn
        return fused

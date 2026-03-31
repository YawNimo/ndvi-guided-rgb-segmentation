"""Legacy SPAM attention block for runmodel-local SPANet implementation."""

import torch.nn as nn

class SPAMBlock(nn.Module):
    """Apply multi-scale pooled attention to feature maps."""

    def __init__(self, in_channels, pool_sizes=(1, 2, 4), reduction=4):
        """Initialize SPAM block parameters.

        Args:
            in_channels (int): Input feature channels.
            pool_sizes (tuple[int, ...]): Adaptive pooling scales.
            reduction (int): Channel reduction factor.
        """
        super().__init__()
        self.pool_sizes = pool_sizes
        mid_channels = max(in_channels // reduction, 1)

        self.conv_reduce = nn.Conv2d(in_channels * len(pool_sizes), mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_attn = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Compute attention map from pooled context and apply to input tensor."""
        B, C, H, W = x.shape
        pooled_list = []
        for ps in self.pool_sizes:
            p = F.adaptive_avg_pool2d(x, output_size=(ps, ps))
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            pooled_list.append(p)

        multi_scale = torch.cat(pooled_list, dim=1)
        h = self.relu(self.bn_reduce(self.conv_reduce(multi_scale)))
        attn = self.sigmoid(self.conv_attn(h))
        return x * attn
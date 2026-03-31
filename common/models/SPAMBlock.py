"""Spatial pyramid attention module used in SPANetFull."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPAMBlock(nn.Module):
    """Apply multi-scale pooled attention to an input feature map."""

    def __init__(self, in_channels, pool_sizes=(1, 2, 4), reduction=4):
        """Initialize the SPAM block.

        Args:
            in_channels (int): Number of channels in the input tensor.
            pool_sizes (tuple[int, ...]): Pyramid pooling output sizes.
            reduction (int): Reduction factor for the bottleneck channels.
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
        """Compute and apply spatial attention from pooled multi-scale context.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Attention-weighted tensor with shape ``(N, C, H, W)``.
        """
        _, _, h, w = x.shape
        pooled_list = []
        for ps in self.pool_sizes:
            p = F.adaptive_avg_pool2d(x, output_size=(ps, ps))
            p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            pooled_list.append(p)

        multi_scale = torch.cat(pooled_list, dim=1)
        y = self.relu(self.bn_reduce(self.conv_reduce(multi_scale)))
        attn = self.sigmoid(self.conv_attn(y))
        return x * attn

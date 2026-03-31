"""ResNet-50 feature encoder that returns low and high feature maps."""

import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


class ResNet50Encoder(nn.Module):
    """Backbone encoder exposing intermediate features for SPANetFull."""

    def __init__(self, pretrained=True):
        """Initialize ResNet-50 encoder layers.

        Args:
            pretrained (bool): Whether to initialize with ImageNet pretrained weights.
        """
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        """Encode an input image into low/high feature tensors.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - ``low`` from layer2
                - ``high`` from layer4
        """
        x = self.stem(x)
        x = self.layer1(x)
        low = self.layer2(x)
        x = self.layer3(low)
        high = self.layer4(x)
        return low, high

import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
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
        x = self.stem(x)
        x = self.layer1(x)
        low = self.layer2(x)
        x = self.layer3(low)
        high = self.layer4(x)
        return low, high

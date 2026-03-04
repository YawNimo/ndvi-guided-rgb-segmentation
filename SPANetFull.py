import torch.nn as nn
from ResNet50Encoder import ResNet50Encoder
from SPAMBlock import SPAMBlock
from FeatureFusionModule import FeatureFusionModule


class SPANetFull(nn.Module):
    def __init__(self, num_classes=4, pretrained_backbone=True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained_backbone)

        self.low_ch = 512
        self.high_ch = 2048

        self.spam_low = SPAMBlock(self.low_ch)
        self.spam_high = SPAMBlock(self.high_ch)
        self.ffm = FeatureFusionModule(self.low_ch, self.high_ch)

        self.dec_conv1 = nn.Conv2d(self.low_ch, 256, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        low, high = self.encoder(x)

        low_enh = self.spam_low(low)
        high_enh = self.spam_high(high)

        fused_low = self.ffm(low_enh, high_enh)

        y = self.dec_bn1(F.relu(self.dec_conv1(fused_low)))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)

        y = self.dec_bn2(F.relu(self.dec_conv2(y)))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        logits = self.classifier(y)
        return logits

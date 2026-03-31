import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50

from .SPANetFull import SPANetFull


def build_unet(num_classes=4, encoder_name="resnet34", encoder_weights="imagenet"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


def build_deeplab(num_classes=4):
    model = deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )

    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    return model


def build_model(name: str, num_classes=4, unet_encoder="resnet34", unet_weights="imagenet"):
    name = name.lower()
    if name == "unet":
        return build_unet(
            num_classes=num_classes,
            encoder_name=unet_encoder,
            encoder_weights=unet_weights,
        )
    if name == "deeplab":
        return build_deeplab(num_classes=num_classes)
    if name == "spanetfull":
        return SPANetFull(num_classes=num_classes, pretrained_backbone=True)
    raise ValueError(f"Unknown model '{name}'. Choose from: unet, deeplab, spanetfull")

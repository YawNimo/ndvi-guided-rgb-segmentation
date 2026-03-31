"""Model factory helpers for segmentation architectures used in this repository."""

import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import deeplabv3_resnet50

from .SPANetFull import SPANetFull


def build_unet(
    num_classes: int = 4,
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """Construct a U-Net model via ``segmentation_models_pytorch``.

    Args:
        num_classes (int): Number of output segmentation classes.
        encoder_name (str): Backbone encoder name supported by SMP.
        encoder_weights (str): Pretrained weight spec for the encoder.

    Returns:
        nn.Module: Configured U-Net model with logits output.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    return model


def build_deeplab(num_classes: int = 4) -> nn.Module:
    """Construct DeepLabV3-ResNet50 and replace classifier heads.

    Args:
        num_classes (int): Number of output segmentation classes.

    Returns:
        nn.Module: DeepLabV3 model configured for ``num_classes`` logits.
    """
    model = deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )

    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    return model


def build_model(
    name: str,
    num_classes: int = 4,
    unet_encoder: str = "resnet34",
    unet_weights: str = "imagenet",
) -> nn.Module:
    """Build a model by name from the supported architecture set.

    Args:
        name (str): Model key. Supported: ``unet``, ``deeplab``, ``spanetfull``.
        num_classes (int): Number of output classes.
        unet_encoder (str): U-Net encoder when ``name`` is ``unet``.
        unet_weights (str): U-Net encoder weight spec when ``name`` is ``unet``.

    Returns:
        nn.Module: Instantiated segmentation model.

    Usage:
        Choose the architecture from CLI/config and pass it to this factory.
    """
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

"""Model factory for reviewed study backbones."""

from __future__ import annotations
from typing import Any


def _gray_conv(old: Any, nn: Any) -> Any:
    return nn.Conv2d(1,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=old.bias is not None)


def _dual_conv(old: Any, nn: Any) -> Any:
    return nn.Conv2d(2,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=old.bias is not None)


def _copy_rgb_mean(old: Any, new: Any, torch: Any) -> None:
    with torch.no_grad():
        new.weight.copy_(old.weight.mean(dim=1,keepdim=True))
        if old.bias is not None and new.bias is not None:new.bias.copy_(old.bias)


def build_model(architecture: str,num_classes: int,pretrained: str="none",image_size: int=224)->Any:
    import torch
    from torch import nn
    try:
        import torchvision.models as models
    except Exception as exc:raise RuntimeError("reviewed image backbones require torchvision") from exc
    weights_enabled=pretrained=="imagenet"
    if pretrained not in {"none","imagenet"}:raise ValueError("pretrained must be none or imagenet")
    if architecture in {"controlled_cnn_s_gray","controlled_cnn_m_gray"}:
        if weights_enabled:raise ValueError("controlled CNNs are scratch-only")
        channels=(8,16,32) if architecture=="controlled_cnn_s_gray" else (16,32,64)
        layers=[];in_channels=1
        for out_channels in channels:
            layers.extend([nn.Conv2d(in_channels,out_channels,3,padding=1),nn.ReLU(inplace=True),nn.MaxPool2d(2)])
            in_channels=out_channels
        layers.extend([nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(channels[-1],num_classes)])
        return nn.Sequential(*layers)
    if architecture=="resnet18_gray":
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_enabled else None);old=model.conv1;model.conv1=_gray_conv(old,nn)
        if weights_enabled:_copy_rgb_mean(old,model.conv1,torch)
        model.fc=nn.Linear(model.fc.in_features,num_classes);return model
    if architecture=="resnet34_gray":
        model=models.resnet34(weights=models.ResNet34_Weights.DEFAULT if weights_enabled else None);old=model.conv1;model.conv1=_gray_conv(old,nn)
        if weights_enabled:_copy_rgb_mean(old,model.conv1,torch)
        model.fc=nn.Linear(model.fc.in_features,num_classes);return model
    if architecture=="resnet50_gray":
        model=models.resnet50(weights=models.ResNet50_Weights.DEFAULT if weights_enabled else None);old=model.conv1;model.conv1=_gray_conv(old,nn)
        if weights_enabled:_copy_rgb_mean(old,model.conv1,torch)
        model.fc=nn.Linear(model.fc.in_features,num_classes);return model
    if architecture=="resnet18_dual":
        model=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights_enabled else None);old=model.conv1;model.conv1=_dual_conv(old,nn)
        if weights_enabled:
            with torch.no_grad():model.conv1.weight.copy_(old.weight[:,:2])
        model.fc=nn.Linear(model.fc.in_features,num_classes);return model
    if architecture=="densenet121_gray":
        model=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if weights_enabled else None);old=model.features.conv0;model.features.conv0=_gray_conv(old,nn)
        if weights_enabled:_copy_rgb_mean(old,model.features.conv0,torch)
        if image_size<=32:model.features.pool0=nn.Identity()
        model.classifier=nn.Linear(model.classifier.in_features,num_classes);return model
    if architecture=="vgg16_gray":
        model=models.vgg16(weights=models.VGG16_Weights.DEFAULT if weights_enabled else None);old=model.features[0];model.features[0]=_gray_conv(old,nn)
        if weights_enabled:_copy_rgb_mean(old,model.features[0],torch)
        model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,num_classes);return model
    if architecture=="swin_tiny_gray":
        base=models.swin_t(weights=models.Swin_T_Weights.DEFAULT if weights_enabled else None);base.head=nn.Linear(base.head.in_features,num_classes)
        class GraySwin(nn.Module):
            def __init__(self):super().__init__();self.net=base
            def forward(self,x):return self.net(x.repeat(1,3,1,1) if x.shape[1]==1 else x)
        return GraySwin()
    raise ValueError(f"unsupported reviewed architecture {architecture!r}")

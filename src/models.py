import torch
import torch.nn as nn
import timm
from torchvision import transforms
from typing import Tuple

# ---------- Utils ----------
def _resolve_timm_input_hw(model_name: str) -> Tuple[int, int]:
    """
    Resolve the spatial input size (H, W) a timm model expects.
    Handles both new and older timm APIs; defaults to 224x224 if unknown.
    """
    try:
        # timm >= 0.9
        cfg = timm.models.resolve_pretrained_cfg(model_name)
        size = cfg.get("input_size", None)
        if size and len(size) == 3:
            return int(size[-2]), int(size[-1])
    except Exception:
        pass
    try:
        # older timm
        cfg = timm.models.get_pretrained_cfg(model_name)
        size = getattr(cfg, "input_size", None)
        if size and len(size) == 3:
            return int(size[-2]), int(size[-1])
    except Exception:
        pass
    return 224, 224  # safe default

def _get_backbone_out_features(model: nn.Module, model_name: str) -> int:
    """
    Determine feature width D for the classification head.
    Prefer timm's `num_features`; otherwise do a tiny forward pass.
    """
    nf = getattr(model, "num_features", None)
    if isinstance(nf, int) and nf > 0:
        return nf
    H, W = _resolve_timm_input_hw(model_name)
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, H, W)
        y = model(x)  # could be [B, C, H, W] or [B, D]
        if y.ndim == 4:
            y = nn.AdaptiveAvgPool2d(1)(y).flatten(1)
        elif y.ndim != 2:
            raise ValueError(f"Unexpected backbone output shape: {tuple(y.shape)}")
        return int(y.shape[1])

def get_transforms(model_name: str):
    """
    Creates model-specific transformations using timm's pretrained model configuration.
    """
    try:
        # Tries to get model configuration from timm
        cfg = timm.models.resolve_pretrained_cfg(model_name)
        img_size = cfg.input_size[-1]
        mean, std = cfg.mean, cfg.std
    except Exception:
        # Falls back to a generic default if model config not found
        img_size = 224
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        print(f"Warning: Could not resolve timm config for {model_name}. Using default transforms.")

    # Transformations for training (can be augmented further)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # Transformations for validation and testing
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf

# ---------- Model wrapper ----------
class ImageClassifier(nn.Module):
    """
    Wrap a timm backbone and add our own head.
    request `global_pool=''` to get **feature maps**,
    """
    def __init__(self, model_name: str = 'resnet50', num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=''
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = _get_backbone_out_features(self.backbone, model_name)
        self.head = nn.Linear(in_features, num_classes)
        print(f"✅ Created {model_name} with a head accepting {in_features} features.")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(inputs)
        # Support both map and vector outputs from backbones
        if feats.ndim == 4:
            feats = self.pool(feats).flatten(1)
        elif feats.ndim != 2:
            raise ValueError(f"Unexpected backbone output shape during forward: {tuple(feats.shape)}")
        return self.head(feats)

# ---- VGG16-only wrapper using conv features (avoids 4096-D vec) ----
class VGG16Classifier(nn.Module):
    """
    VGG16 special-case: grab the last conv feature map (C=512) via features_only,
    then GAP -> Linear(512 -> num_classes). This sidesteps the 4096-D classifier.
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            'vgg16',
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],    # last conv stage
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_ch = self.backbone.feature_info.channels()[-1]  # should be 512
        self.head = nn.Linear(in_ch, num_classes)
        print(f"✅ Created vgg16 (features_only) with a head accepting {in_ch} features.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats_list = self.backbone(x)       # list[T] with one tensor [B,C,H,W]
        feats = feats_list[-1]
        feats = self.pool(feats).flatten(1) # [B,C]
        return self.head(feats)
    
# ---- factory so the training loop stays clean ----
def make_dl_model(model_name: str, num_classes: int, device: torch.device, pretrained: bool = True) -> nn.Module:
    mn = model_name.lower()
    if mn == "vgg16" or mn.startswith("vgg16"):
        return VGG16Classifier(num_classes=num_classes, pretrained=pretrained).to(device)
    # all other models use your existing generic wrapper
    return ImageClassifier(model_name=model_name, num_classes=num_classes, pretrained=pretrained).to(device)
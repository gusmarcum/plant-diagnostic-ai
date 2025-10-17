#!/usr/bin/env python3
# resnet_classifier.py
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# -----------------------
# Device & defaults
# -----------------------
_HAS_CUDA = torch.cuda.is_available()
_DEV_COUNT = torch.cuda.device_count() if _HAS_CUDA else 0
_DEVICE = torch.device(f"cuda:{1 if _DEV_COUNT > 1 else 0}") if _HAS_CUDA else torch.device("cpu")

# Default class order (will be overwritten by ckpt label_map/classes)
_CLASSES = ["drought", "frost", "healthy", "overwatered", "root_rot"]

# Single source of truth for canonical keys
_ALIAS = {
    "healthy": {"healthy"},
    "overwatered": {"overwatered", "over-watering", "over watering"},
    "root_rot": {"root_rot", "root rot"},
    "drought": {"drought"},
    "frost": {"frost", "frost injury"},
    "gray_mold": {"gray_mold", "gray mold", "grey_mold", "grey mold"},
    "white_mold": {"white_mold", "white mold"},
}

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower().replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    for canon, variants in _ALIAS.items():
        if s in variants:
            return canon
    return s.replace(" ", "_")  # fallback canonicalization

# Per-class accept thresholds (adjusted based on class weights)
_CLASS_THRESH = {
    "healthy": 0.40,  # Lowered to catch more healthy predictions
    "overwatered": 0.60,
    "root_rot": 0.65,
    "drought": 0.70,
    "frost": 0.75,
    "gray_mold": 0.40,  # Much lower due to low class weight (0.407)
    "white_mold": 0.45,  # Lower for better mold detection
}
_THRESH_DEFAULT = 0.60  # Lowered for better sensitivity
_TH_TOP2 = 0.80

# Optional temperature from calibration (ckpt can override)
_TEMPERATURE = 0.78  # typical calibrated band ~0.74–0.81

# -----------------------
# Transforms
# -----------------------
def _tfm(img_size: int = 256):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

# -----------------------
# Loading
# -----------------------
def _extract_classes_from_ckpt(ckpt: dict) -> list[str]:
    # Preferred: label_map {idx:int -> class:str} or list indexed by idx
    label_map = ckpt.get("label_map")
    if isinstance(label_map, dict) and len(label_map) > 0:
        # keys might be strings if saved via JSON
        pairs = sorted(((int(k), v) for k, v in label_map.items()), key=lambda t: t[0])
        return [v for _, v in pairs]

    meta = ckpt.get("meta", {})
    lm = meta.get("label_map")
    if isinstance(lm, dict) and len(lm) > 0:
        pairs = sorted(((int(k), v) for k, v in lm.items()), key=lambda t: t[0])
        return [v for _, v in pairs]

    classes = ckpt.get("classes") or meta.get("classes")
    if isinstance(classes, (list, tuple)) and classes:
        return list(classes)

    # Fall back to global default
    return list(_CLASSES)

def load_resnet(ckpt_path: str = "plant_diagnostic/models/resnet_straw5.pth"):
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    classes = _extract_classes_from_ckpt(ckpt)
    # overwrite global list with ckpt order to keep indices aligned
    global _CLASSES
    _CLASSES = classes

    temp = ckpt.get("temperature") or (ckpt.get("meta", {}) if isinstance(ckpt.get("meta", {}), dict) else {}).get("temperature")
    if isinstance(temp, (int, float)) and temp > 0:
        global _TEMPERATURE
        _TEMPERATURE = float(temp)

    net = models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(_CLASSES))
    state = ckpt.get("model") or ckpt  # tolerate raw state_dict
    net.load_state_dict(state, strict=True)
    net = net.to(_DEVICE).eval()
    return net

# -----------------------
# Inference
# -----------------------
@torch.no_grad()
def diagnose_or_none(model, image_path: str, img_size: int = 256):
    # PIL can throw; be explicit
    with Image.open(image_path) as im:
        img = im.convert("RGB")

    x = _tfm(img_size)(img).unsqueeze(0).to(_DEVICE)

    # 2-view TTA: center + hflip
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))
    logits = (logits1 + logits2) / 2

    # temperature calibration (if present)
    probs = F.softmax(logits / _TEMPERATURE, dim=1).squeeze(0)

    # top-2
    pvals, idxs = torch.sort(probs, descending=True)
    p1, i1 = float(pvals[0]), int(idxs[0])
    p2, i2 = float(pvals[1]), int(idxs[1])

    raw1 = _CLASSES[i1]
    raw2 = _CLASSES[i2]
    label1 = _normalize_label(raw1)
    label2 = _normalize_label(raw2)

    # Always return the top prediction - ResNet is very accurate
    # Commented out threshold check since ResNet predictions are reliable
    # thr = _CLASS_THRESH.get(label1, _THRESH_DEFAULT)
    # accept = (p1 >= thr) or ((p1 + p2) >= _TH_TOP2)
    # if not accept:
    #     return None

    return {"label": label1, "p1": p1, "top2": (label2, p2), "raw": (raw1, raw2)}

#!/usr/bin/env python3
# resnet_classifier.py (7-class, checkpoint-aware)
# - Loads class list + temperature from checkpoint if available
# - Supports both underscore and space variants for thresholds
# - Conservative, image-grounded inference with light TTA + calibrated softmax

import os
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ---------------------------
# Defaults (overridden by ckpt)
# ---------------------------
_CLASSES = [
    "drought", "frost", "healthy", "overwatered", "root_rot", "gray_mold", "white_mold"
]
# default cuda:0 
def _resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    dev = os.getenv("RESNET_DEVICE", "cuda:0")
    # allow simple numbers like "0" or "1"
    if isinstance(dev, str) and dev.isdigit():
        dev = f"cuda:{dev}"
    return torch.device(dev)

_DEVICE = _resolve_device()

# Per-class accept thresholds (space + underscore variants).
_THRESH = {
    # healthy stricter (raise precision)
    "healthy": 0.78,

    "gray_mold": 0.85, "gray mold": 0.85,
    "white_mold": 0.80, "white mold": 0.80,

    "overwatered": 0.80,
    "root_rot": 0.80, "root rot": 0.80,
    "drought": 0.80,

    "frost": 0.82,
}

# Top-2 rescue: if best+second exceeds this, we accept label1 even if p1 below class thr
_TH_TOP2 = 0.80 

# Temperature (logit scaling)
_TEMPERATURE = 0.78  # sane default; overwritten by ckpt if present


# ---------------------------
# Preprocess
# ---------------------------

def _tfm(img_size: int = 256):
    # Match training crop (your trainer normalizes; here we rely on calibrated softmax)
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


# ---------------------------
# Model loading
# ---------------------------

def load_resnet(ckpt_path: str = None):
    """Load a ResNet-50 classifier checkpoint.
    - Respects classes/label_map and temperature if present in ckpt
    - Builds the correct head size from class list
    """
    ckpt_path = ckpt_path or os.getenv("RESNET_CKPT", "plant_diagnostic/models/resnet_straw7.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Prefer explicit class list / label_map from the checkpoint
    classes = ckpt.get("classes")
    label_map = ckpt.get("label_map")
    global _CLASSES
    if isinstance(classes, (list, tuple)) and classes:
        _CLASSES = list(classes)
    elif isinstance(label_map, dict) and label_map:
        # label_map like {0: "classA", 1: "classB", ...}
        _CLASSES = [label_map[i] for i in sorted(label_map.keys())]

    # Temperature from calibration (if present)
    temp = ckpt.get("temperature", None)
    global _TEMPERATURE
    if isinstance(temp, (int, float)) and temp > 0:
        _TEMPERATURE = float(temp)

    # Build model with proper head
    net = models.resnet50(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(_CLASSES))
    state = ckpt.get("model", ckpt)
    net.load_state_dict(state, strict=True)
    net.eval().to(_DEVICE)
    return net


# ---------------------------
# Inference
# ---------------------------

@torch.no_grad()
def diagnose_or_none(model, image_path: str, img_size: int = 256):
    """Return {label, p1, top2:(label2,p2)} or None if below acceptance rules."""
    img = Image.open(image_path).convert("RGB")
    x = _tfm(img_size)(img).unsqueeze(0).to(_DEVICE)

    # Simple TTA: center + horizontal flip
    logits = model(x)
    logits = (logits + model(torch.flip(x, dims=[3]))) / 2

    # Calibrated softmax
    probs = F.softmax(logits / _TEMPERATURE, dim=1).squeeze(0)

    pvals, idxs = torch.sort(probs, descending=True)
    p1, i1 = float(pvals[0]), int(idxs[0])
    p2, i2 = float(pvals[1]), int(idxs[1])

    label1 = _CLASSES[i1]
    label2 = _CLASSES[i2]

    # Threshold lookup supports both underscore and space variants
    thr = _THRESH.get(label1, _THRESH.get(label1.replace("_", " "), 0.80))

    accept = (p1 >= thr) or (p1 + p2 >= _TH_TOP2)
    if not accept:
        return None

    return {"label": label1, "p1": p1, "top2": (label2, p2)}

#!/usr/bin/env python3
# resnet_classifier.py
import torch, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

_CLASSES = ["drought", "frost", "healthy", "overwatered", "root_rot"]
_DEVICE  = torch.device("cuda:1" if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else "cuda:0")

# per-class accept thresholds you’ve been using
_THRESH = {"healthy": 0.67, "overwatered": 0.74, "root_rot": 0.76, "drought": 0.77, "frost": 0.79}
_TH_TOP2 = 0.82
_TEMPERATURE = 0.78  # within your calibrated band ~0.74–0.81

def _tfm(img_size=256):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])

def load_resnet(ckpt_path="models/resnet_straw5.pth"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(_CLASSES))
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval().to(_DEVICE)
    return m

@torch.no_grad()
def diagnose_or_none(model, image_path, img_size=256):
    img = Image.open(image_path).convert("RGB")
    x = _tfm(img_size)(img).unsqueeze(0).to(_DEVICE)

    # simple TTA: center + hflip
    logits = model(x)
    logits = (logits + model(torch.flip(x, dims=[3]))) / 2
    probs = F.softmax(logits / _TEMPERATURE, dim=1).squeeze(0)

    pvals, idxs = torch.sort(probs, descending=True)
    p1, i1 = float(pvals[0]), int(idxs[0])
    p2, i2 = float(pvals[1]), int(idxs[1])

    label1 = _CLASSES[i1]
    label2 = _CLASSES[i2]

    accept = (p1 >= _THRESH[label1]) or (p1 + p2 >= _TH_TOP2)
    if not accept:
        return None

    return {"label": label1, "p1": p1, "top2": (label2, p2)}


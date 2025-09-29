#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

TH_ACCEPT = 0.78
TH_TOP2   = 0.86
IMG = 256

TH_ACCEPT_PER_CLASS = {
    "healthy":      0.70,
    "overwatered":  0.72,
    "root_rot":     0.78,
    "drought":      0.75,
    "frost":        0.80,
}

TTA = [
    transforms.Compose([
        transforms.Resize(int(IMG*1.15)), transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize(int(IMG*1.15)), transforms.CenterCrop(IMG),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([   # slightly tighter crop
        transforms.Resize(int(IMG*1.05)), transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([   # slightly looser crop
        transforms.Resize(int(IMG*1.25)), transforms.CenterCrop(IMG),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
]

EXTS={".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # keep weights_only=False for your own ckpts
    classes = ckpt["classes"]
    T = float(ckpt.get("temperature", 1.0))
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["model"])
    m.eval()
    return m, classes, T

@torch.inference_mode()
def predict_tta(model, pil, T: float = 1.0):
    ps = []
    for tf in TTA:
        x = tf(pil).unsqueeze(0)
        logits = model(x) / T           # apply temperature
        p = logits.softmax(1)[0].cpu()  # softmax AFTER temp
        ps.append(p)
    p = torch.stack(ps).mean(0)
    vals, idxs = torch.topk(p, k=2)
    return idxs.tolist(), vals.tolist(), p.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/resnet_straw5_prod.pth")
    ap.add_argument("--images", required=True, help="file or folder")
    ap.add_argument("--out", default="predictions.json")
    args = ap.parse_args()

    model, classes, T = load_model(args.ckpt)
    print(f"Loaded classes={classes}  temperature={T:.3f}")

    p = Path(args.images)
    if p.is_file() and p.suffix.lower() in EXTS:
        images = [p]
    else:
        images = [q for q in p.rglob("*") if q.suffix.lower() in EXTS]

    results = []
    for img_path in images:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("skip:", img_path, e); continue

        idxs, vals, _ = predict_tta(model, pil, T)
        i1, i2 = idxs[0], idxs[1]
        p1, p2 = float(vals[0]), float(vals[1])
        c1, c2 = classes[i1], classes[i2]

        th_accept = TH_ACCEPT_PER_CLASS.get(c1, TH_ACCEPT)
        if p1 >= th_accept:
            decision, label = "top1", c1
        elif (p1 + p2) >= TH_TOP2:
            decision, label = "top2", f"{c1}|{c2}"
        else:
            decision, label = "unsure", None

        results.append({
            "image": str(img_path),
            "top1": c1, "p1": round(p1,4),
            "top2": c2, "p2": round(p2,4),
            "decision": decision, "label": label
        })
        print(f"{img_path.name:28s}  {c1:<12s} {p1:.2f}  {c2:<12s} {p2:.2f}  -> {decision}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} rows -> {args.out}")

if __name__ == "__main__":
    main()


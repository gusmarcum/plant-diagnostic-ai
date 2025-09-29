#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torchvision import models, transforms, datasets

def load_model(ckpt_path, num_classes=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes) if num_classes is None else num_classes)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()
    return m, classes, ckpt

def get_loader(root, img_size=256, bs=64, nw=4):
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root=root, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return dl, ds

def build_label_map(ds_classes, ckpt_classes):
    """Map dataset class index -> ckpt class index; assert all ckpt classes exist."""
    name_to_ckpt_idx = {c:i for i,c in enumerate(ckpt_classes)}
    if set(ds_classes) != set(ckpt_classes):
        missing = set(ckpt_classes) - set(ds_classes)
        extra   = set(ds_classes) - set(ckpt_classes)
        raise ValueError(f"Class set mismatch.\nMissing in dataset: {missing}\nExtra in dataset: {extra}")
    return torch.tensor([name_to_ckpt_idx[c] for c in ds_classes], dtype=torch.long)

@torch.inference_mode()
def evaluate(model, loader, device, y_map, n_classes):
    cm = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # remap dataset label -> ckpt label index so rows match printed names
        y_ckpt = y_map[y].to(device, non_blocking=True)

        logits = model(x)
        pred = logits.argmax(1)

        for t, p in zip(y_ckpt, pred):
            cm[t.long(), p.long()] += 1

        correct += (pred == y_ckpt).sum().item()
        total   += y_ckpt.numel()
    return correct / total, cm.cpu().numpy()

def per_class_metrics(cm):
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)!=0)
    sup  = cm.sum(1)
    return prec, rec, f1, sup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/resnet_straw5.pth")
    ap.add_argument("--data_root", required=True, help="Folder with class subdirs (true holdout preferred)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt_classes, _ = load_model(args.ckpt)
    model.to(device)

    loader, ds = get_loader(args.data_root, img_size=args.img_size, bs=args.bs, nw=args.num_workers)
    y_map = build_label_map(ds.classes, ckpt_classes)  # ds idx -> ckpt idx

    acc, cm = evaluate(model, loader, device, y_map=y_map, n_classes=len(ckpt_classes))
    prec, rec, f1, sup = per_class_metrics(cm)

    print(f"Overall accuracy: {acc:.3f}")
    for i, c in enumerate(ckpt_classes):
        print(f"{c:12s}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  support={int(sup[i])}")
    print("Confusion matrix (rows=truth, cols=pred in ckpt class order):")
    for i, row in enumerate(cm):
        print(ckpt_classes[i].ljust(12), row.tolist())

if __name__ == "__main__":
    main()


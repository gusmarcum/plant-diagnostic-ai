#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from PIL import Image

def build_val_tf(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/resnet_straw5.pth")
    ap.add_argument("--data_root", default="data/holdout")
    ap.add_argument("--out_dir", default="misclassified")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--include_correct", action="store_true",
                    help="Also copy correct preds into <true>_to_<true>/")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (class order used during training)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_classes = ckpt["classes"]

    # Build model
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(ckpt_classes))
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval().to(device)

    # Dataset (class order may differ from ckpt; handle mapping)
    tf = build_val_tf(args.img_size)
    ds = datasets.ImageFolder(args.data_root, transform=tf)
    ds_classes = ds.classes
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    paths = [p for p, _ in ds.samples]

    # Map dataset label indices -> ckpt label indices
    # (model outputs are in ckpt_classes order)
    name_to_ckpt = {name: i for i, name in enumerate(ckpt_classes)}
    try:
        ds_to_ckpt = np.array([name_to_ckpt[name] for name in ds_classes], dtype=np.int64)
    except KeyError as e:
        missing = str(e).strip("'")
        raise SystemExit(f"Class '{missing}' present in dataset but not in checkpoint.\n"
                         f"ckpt classes: {ckpt_classes}\n ds classes: {ds_classes}")

    # For confusion matrix in dataset (human-readable) order:
    ckpt_to_ds = {name_to_ckpt[name]: i for i, name in enumerate(ds_classes)}

    # Confusion matrix rows=true(ds order), cols=pred(ds order)
    C = len(ds_classes)
    cm = np.zeros((C, C), dtype=np.int64)

    copied = 0
    i0 = 0
    for xb, yb_ds in dl:
        xb = xb.to(device)
        logits = m(xb)
        pred_ckpt = logits.argmax(1).cpu().numpy()     # indices in ckpt order
        yb_ds = yb_ds.numpy()                          # indices in ds order
        yb_ckpt = ds_to_ckpt[yb_ds]                    # true labels in ckpt order

        # update cm in dataset order
        pred_ds = np.array([ckpt_to_ds[int(i)] for i in pred_ckpt], dtype=np.int64)
        for t_ds, p_ds in zip(yb_ds, pred_ds):
            cm[t_ds, p_ds] += 1

        # copy misclassifications (and optional corrects)
        for j in range(len(yb_ds)):
            true_idx_ds = int(yb_ds[j])
            pred_idx_ckpt = int(pred_ckpt[j])
            pred_name = ckpt_classes[pred_idx_ckpt]
            true_name = ds_classes[true_idx_ds]
            correct = (pred_idx_ckpt == int(yb_ckpt[j]))

            if correct and not args.include_correct:
                continue

            sub = f"{true_name}_to_{pred_name}"
            dst_dir = out_dir / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            src = Path(paths[i0 + j])
            dst = dst_dir / src.name
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
            except Exception as e:
                print("Copy failed:", src, "->", dst, e)
        i0 += len(yb_ds)

    # Report
    print(f"\nWrote {copied} files under '{out_dir}/<true>_to_<pred>/'")

    # Pretty print confusion matrix & per-class metrics
    print("\nConfusion matrix (rows=truth, cols=pred):")
    col_header = " ".join([f"{n:12s}" for n in ds_classes])
    print(" " * 13 + col_header)
    for i, name in enumerate(ds_classes):
        row = " ".join([f"{cm[i, j]:12d}" for j in range(C)])
        print(f"{name:12s} {row}")

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) != 0)
    print("\nPer-class (P/R/F1/support):")
    for i, name in enumerate(ds_classes):
        print(f"{name:12s} P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  support={cm[i].sum()}")

if __name__ == "__main__":
    main()


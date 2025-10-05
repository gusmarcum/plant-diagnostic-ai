#!/usr/bin/env python3
"""
K-fold cross-validation trainer for the 7-class strawberry ResNet anchor.

Features
- Stratified K-fold split from a single ImageFolder root (train-only root). No leakage.
- Two-phase finetune (head-only → selective unfreeze with per-layer LRs).
- Class imbalance handling: class-weighted loss or WeightedRandomSampler.
- Optional per-class weight boosts (e.g., "white_mold:1.15,gray_mold:0.95").
- AMP for speed, EMA for stability, MixUp (light) support.
- Saves best checkpoint per fold and a summary JSON of metrics.
- Optional warm-start from an older checkpoint (backbone only; drops fc to avoid size mismatch).
- Temperature calibration per-fold (optional) using LBFGS on the fold's val set.

Example launch
-------------
python /data/kiriti/MiniGPT-4/plant_diagnostic/scripts/resnetscripts/train_resnet_kfold.py \
  --data_root /data/kiriti/MiniGPT-4/plant_diagnostic/data/train_aug \
  --out_dir   /data/kiriti/MiniGPT-4/plant_diagnostic/models/kfold_resnet \
  --k 5 \
  --epochs_head 3 --epochs_full 18 \
  --batch_size 32 --img_size 256 \
  --balance weights \
  --boost_classes "frost:1.3,white_mold:1.1,gray_mold:0.95" \
  --label_smoothing 0.05 --mixup 0.05 --ema 0.998 \
  --resume /data/kiriti/MiniGPT-4/plant_diagnostic/models/resnet_straw7.pth \
  --calibrate

Notes
- Device selection honors env RESNET_DEVICE (e.g., "0" or "cuda:1"), default cuda:0.
- Output files: out_dir/fold{j}.pth and out_dir/summary.json
"""
from __future__ import annotations
import argparse, json, math, os, random, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# -----------------------------
# Device & Seeding
# -----------------------------

def resolve_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    dev = os.getenv("RESNET_DEVICE", "cuda:0")
    if isinstance(dev, str) and dev.isdigit():
        dev = f"cuda:{dev}"
    return torch.device(dev)


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# -----------------------------
# Transforms
# -----------------------------

def build_transforms(img_size: int):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0)),
        transforms.RandomRotation(degrees=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.15),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAutocontrast(p=0.20),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.5), ratio=(0.3, 3.3)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf_train, tf_val


# -----------------------------
# Dataset & Splits
# -----------------------------

def load_imagefolder(root: str, tf) -> datasets.ImageFolder:
    rootp = Path(root)
    if not any(rootp.rglob("*"+ext) for ext in VALID_EXTS):
        raise SystemExit(f"No images under {root}")
    return datasets.ImageFolder(root=root, transform=tf)


def stratified_kfold_indices(targets: List[int], num_classes: int, k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(targets):
        per_class[y].append(idx)
    # shuffle within class
    for c in range(num_classes):
        rng.shuffle(per_class[c])
    # slice per class into k roughly equal chunks
    per_class_slices: List[List[List[int]]] = []
    for c in range(num_classes):
        idxs = per_class[c]
        folds = [idxs[i::k] for i in range(k)]  # round-robin split
        per_class_slices.append(folds)
    # build folds by concatenating class slices
    folds: List[List[int]] = []
    for i in range(k):
        fold = []
        for c in range(num_classes):
            fold.extend(per_class_slices[c][i])
        rng.shuffle(fold)
        folds.append(fold)
    # create (train_idx, val_idx) per fold
    splits: List[Tuple[List[int], List[int]]] = []
    all_idx = set(range(len(targets)))
    for i in range(k):
        val_idx = folds[i]
        train_idx = list(all_idx.difference(val_idx))
        splits.append((train_idx, val_idx))
    return splits


# -----------------------------
# Sampler, Weights, MixUp, EMA
# -----------------------------

def make_sampler(train_targets: List[int], num_classes: int) -> WeightedRandomSampler:
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)
    inv = np.zeros_like(counts)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    weights = np.array([inv[t] for t in train_targets], dtype=np.float64)
    weights[weights <= 0.0] = 1e-8
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )


def parse_boosts(boost_str: str, classes: List[str]) -> np.ndarray:
    boosts_map = {c: 1.0 for c in classes}
    if boost_str:
        for tok in boost_str.split(","):
            tok = tok.strip()
            if not tok or ":" not in tok:
                continue
            name, val = tok.split(":", 1)
            name = name.strip()
            try:
                f = float(val)
            except ValueError:
                continue
            if name in boosts_map:
                boosts_map[name] = f
    return np.array([boosts_map[c] for c in classes], dtype=float)


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    x2 = x[index]
    y2 = y[index]
    return lam * x + (1 - lam) * x2, (y, y2), lam


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.0):
        self.decay = decay
        self.shadow = None if decay <= 0.0 else {
            k: p.detach().clone() for k,p in model.state_dict().items() if p.dtype.is_floating_point
        }
    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.shadow is None: return
        for k,p in model.state_dict().items():
            if k in self.shadow and p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        if self.shadow is None: return
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)


# -----------------------------
# Eval helpers
# -----------------------------
@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval(); correct=total=0
    for x,y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return correct/total if total else 0.0


@torch.inference_mode()
def evaluate_cm(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    for x,y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu().numpy()
        y_np = y.cpu().numpy()
        for t,p in zip(y_np, pred):
            cm[t,p] += 1
    tp = cm.diagonal()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp)!=0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp, dtype=float), where=(tp+fn)!=0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec, dtype=float), where=(prec+rec)!=0)
    macro_f1 = float(np.mean(f1)) if len(f1) else 0.0
    return cm, prec, rec, f1, macro_f1


# -----------------------------
# Temperature calibration (optional)
# -----------------------------
@torch.inference_mode(False)
def fit_temperature(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    T = torch.nn.Parameter(torch.ones([], device=device))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_n = 0
        for x,y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x) / T.clamp_min(1e-3)
            total_loss += F.cross_entropy(logits, y, reduction="sum")
            total_n += y.size(0)
        loss = total_loss / max(1, total_n)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().cpu())


# -----------------------------
# Main training per fold
# -----------------------------

def train_fold(fold_id: int, splits, ds_train_full, ds_val_full, classes: List[str], args, device: torch.device):
    train_idx, val_idx = splits[fold_id]
    train_targets_all = getattr(ds_train_full, "targets", [c for _,c in ds_train_full.samples])

    ds_train = Subset(ds_train_full, train_idx)
    ds_val   = Subset(ds_val_full,   val_idx)

    # Sampler or plain shuffle
    sampler = None
    if args.balance == "sampler":
        train_targets = [train_targets_all[i] for i in train_idx]
        sampler = make_sampler(train_targets, len(classes))

    train_loader = DataLoader(ds_train, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Class-weighted loss
    if args.balance == "weights":
        tr_counts = np.bincount([train_targets_all[i] for i in train_idx], minlength=len(classes)).astype(float)
        weights = tr_counts.sum() / (tr_counts + 1e-6)
        weights = weights / weights.mean()
        boosts = parse_boosts(args.boost_classes, classes)
        weights *= boosts
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device).to(memory_format=torch.channels_last)

    # Warm-start (backbone only)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt.get("model", ckpt)
        state = {k:v for k,v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[fold {fold_id}] resume loaded: missing={len(missing)} unexpected={len(unexpected)}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda" if device.type=="cuda" else None)
    ema = EMA(model, decay=args.ema)

    # Phase 1: head-only
    best_acc = 0.0
    if args.epochs_head > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        # Freeze BN
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=0.05)
        for e in range(args.epochs_head):
            model.train()
            for x,y in train_loader:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                x_mix, y_mix, lam = mixup_batch(x, y, args.mixup)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
                    logits = model(x_mix)
                    if lam is None:
                        loss = loss_fn(logits, y)
                    else:
                        y1,y2 = y_mix
                        loss = lam*loss_fn(logits,y1) + (1-lam)*loss_fn(logits,y2)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(opt); scaler.update(); ema.update(model)
            acc = evaluate(model, val_loader, device)
            if acc > best_acc:
                best_acc = acc
                outp = Path(args.out_dir)/f"fold{fold_id}.pth"
                torch.save({"model": model.state_dict(), "classes": classes}, outp)
            print(f"[fold {fold_id}][head] {e+1}/{args.epochs_head} val_acc={acc:.3f}")

    # Phase 2: selective unfreeze
    for p in model.parameters():
        p.requires_grad = False
    for name,p in model.named_parameters():
        if any(k in name for k in ["layer1","layer2","layer3","layer4","fc"]):
            p.requires_grad = True
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def param_groups(model, base_lr, wd):
        buckets = {"layer1":[],"layer2":[],"layer3":[],"layer4":[],"fc":[]}
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            key = "fc" if "fc" in n else next((k for k in ["layer1","layer2","layer3","layer4"] if k in n), None)
            if key is None: continue
            no_decay = (p.ndim==1) or n.endswith(".bias")
            buckets[key].append((p,no_decay))
        lrs = {"layer1":base_lr*0.25, "layer2":base_lr*0.50, "layer3":base_lr*0.75, "layer4":base_lr*1.00, "fc":base_lr*1.00}
        groups = []
        for k,params in buckets.items():
            if not params: continue
            decay_params   = [p for p,nd in params if not nd]
            nodecay_params = [p for p,nd in params if nd]
            if decay_params:
                groups.append({"params":decay_params,"lr":lrs[k],"weight_decay":wd})
            if nodecay_params:
                groups.append({"params":nodecay_params,"lr":lrs[k],"weight_decay":0.0})
        return groups

    pg = param_groups(model, base_lr=args.lr_full, wd=0.05)
    opt = torch.optim.AdamW(pg)
    warmup_epochs = 2
    if args.epochs_full > warmup_epochs:
        warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=max(1, args.epochs_full - warmup_epochs))
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs_full))

    for e in range(args.epochs_full):
        model.train()
        for x,y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x_mix, y_mix, lam = mixup_batch(x, y, args.mixup)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=("cuda" if device.type=="cuda" else "cpu")):
                logits = model(x_mix)
                if lam is None:
                    loss = loss_fn(logits, y)
                else:
                    y1,y2 = y_mix
                    loss = lam*loss_fn(logits,y1) + (1-lam)*loss_fn(logits,y2)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(opt); scaler.update(); ema.update(model)
        sched.step()

        # Evaluate (EMA-applied)
        snapshot = model.state_dict()
        if ema.shadow is not None:
            ema.apply_to(model)
        acc = evaluate(model, val_loader, device)
        cm, prec, rec, f1, macro_f1 = evaluate_cm(model, val_loader, device, num_classes=len(classes))
        if ema.shadow is not None:
            model.load_state_dict(snapshot)

        if acc > best_acc:
            best_acc = acc
            outp = Path(args.out_dir)/f"fold{fold_id}.pth"
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "label_map": {i:c for i,c in enumerate(classes)},
                "meta": {
                    "img_size": args.img_size,
                    "label_smoothing": args.label_smoothing,
                    "class_weights": (class_weights.detach().cpu().tolist() if class_weights is not None else None),
                    "ema_decay": args.ema,
                    "mixup_alpha": args.mixup,
                }
            }, outp)
        print(f"[fold {fold_id}][full] {e+1}/{args.epochs_full} lr={opt.param_groups[0]['lr']:.2e} val_acc={acc:.3f} macro_f1={macro_f1:.3f}")
        # Per-class quick print (rounded)
        pcs = {classes[i]: (round(float(prec[i]),3), round(float(rec[i]),3), round(float(f1[i]),3)) for i in range(len(classes))}
        print(f"[fold {fold_id}] per-class: {pcs}")

    # Temperature calibration on best model (optional)
    temp = None
    if args.calibrate:
        best_ckpt = torch.load(Path(args.out_dir)/f"fold{fold_id}.pth", map_location="cpu")
        model.load_state_dict(best_ckpt["model"], strict=False)
        model.to(device)
        temp = fit_temperature(model, val_loader, device)
        best_ckpt["temperature"] = float(temp)
        best_ckpt.setdefault("meta", {})["temperature"] = float(temp)
        torch.save(best_ckpt, Path(args.out_dir)/f"fold{fold_id}.pth")
        print(f"[fold {fold_id}] calibrated temperature = {temp:.3f}")

    # Final metrics on (EMA-applied) best model loaded above
    return {"fold": fold_id, "best_val_acc": round(float(best_acc), 4), "temperature": (None if temp is None else round(float(temp), 4))}


# -----------------------------
# Entry
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stratified K-fold trainer for strawberry ResNet anchor")
    ap.add_argument("--data_root", required=True, help="Root with class subdirs (single tree; we split K folds)")
    ap.add_argument("--out_dir", required=True, help="Directory to write fold checkpoints & summary.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs_head", type=int, default=3)
    ap.add_argument("--epochs_full", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="", help="Optional .pth to warm start from (backbone only)")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--balance", choices=["weights","sampler","none"], default="weights")
    ap.add_argument("--boost_classes", default="", help="e.g. 'white_mold:1.1,gray_mold:0.95' (weights mode only)")
    ap.add_argument("--mixup", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    device = resolve_device()
    seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_train, tf_val = build_transforms(args.img_size)
    ds_train_full = load_imagefolder(args.data_root, tf_train)
    ds_val_full   = load_imagefolder(args.data_root, tf_val)   # same ordering, different tf
    classes = ds_train_full.classes
    targets = getattr(ds_train_full, "targets", [c for _,c in ds_train_full.samples])

    splits = stratified_kfold_indices(targets, len(classes), args.k, args.seed)

    # Print fold sizes
    print("Classes:", classes)
    for i,(tr,va) in enumerate(splits):
        tr_counts = np.bincount([targets[j] for j in tr], minlength=len(classes))
        va_counts = np.bincount([targets[j] for j in va], minlength=len(classes))
        print(f"[fold {i}] train per-class:", dict(zip(classes, tr_counts.tolist())))
        print(f"[fold {i}]   val per-class:", dict(zip(classes, va_counts.tolist())))

    summary = {"folds": [], "classes": classes}

    for i in range(args.k):
        res = train_fold(i, splits, ds_train_full, ds_val_full, classes, args, device)
        summary["folds"].append(res)
        # flush partial summary
        (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))

    # Aggregate best_val_acc across folds
    accs = [f["best_val_acc"] for f in summary["folds"]]
    summary["mean_best_val_acc"] = round(float(np.mean(accs)), 4) if accs else 0.0
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE. Summary ->", out_dir/"summary.json")


if __name__ == "__main__":
    main()

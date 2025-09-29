#!/usr/bin/env python3
import argparse, random
import os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import math, random
from PIL import Image

IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}


def build_transforms(img_size):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0)),  # slightly tighter
        transforms.RandomRotation(degrees=3),                        # was 5
        transforms.RandomHorizontalFlip(),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.15), # was 0.3@0.2
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomAutocontrast(p=0.20),

        # --- tensor-level ---
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.8), ratio=(0.3, 3.3)),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf_train, tf_val


def load_imagefolder(root, tf):
    if not any(Path(root).rglob("*"+ext) for ext in IMG_EXTS):
        raise SystemExit(f"No images under {root}")
    return datasets.ImageFolder(root=root, transform=tf)

def stratified_split_indices(targets, num_classes, val_frac=0.1, seed=42):
    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(targets):
        per_class[y].append(idx)
    train_idx, val_idx = [], []
    for c in range(num_classes):
        idxs = per_class[c]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs)*val_frac)) if len(idxs)>0 else 0
        val_idx.extend(idxs[:n_val]); train_idx.extend(idxs[n_val:])
    # ensure no empty train class
    train_counts = np.bincount([targets[i] for i in train_idx], minlength=num_classes)
    for c in range(num_classes):
        if train_counts[c] == 0:
            swap = [i for i in val_idx if targets[i]==c]
            if swap:
                i = swap[0]
                val_idx.remove(i); train_idx.append(i)
    return train_idx, val_idx

def make_sampler(train_targets, num_classes):
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)
    inv = np.zeros_like(counts, dtype=np.float64)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    weights = np.array([inv[t] for t in train_targets], dtype=np.float64)
    weights[weights <= 0.0] = 1e-8
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True
    )

def parse_boosts(boost_str: str, classes):
    """
    Parse a string like "healthy:1.6,overwatered:1.6,drought:1.0" into
    an array aligned with `classes`. Missing classes default to 1.0.
    Invalid tokens are ignored.
    """
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
    import numpy as np
    return np.array([boosts_map[c] for c in classes], dtype=float)


def get_loaders(data_root, img_size=256, batch_size=32, num_workers=4,
                val_frac=0.1, seed=42, balance="weights"):
    tf_train, tf_val = build_transforms(img_size)
    ds_full = load_imagefolder(data_root, tf_train)
    classes = ds_full.classes
    targets = getattr(ds_full, "targets", [c for _, c in ds_full.samples])

    train_idx, val_idx = stratified_split_indices(targets, len(classes), val_frac, seed)
    from torch.utils.data import Subset
    ds_train = Subset(ds_full, train_idx)
    ds_val_full = load_imagefolder(data_root, tf_val)
    ds_val = Subset(ds_val_full, val_idx)

    sampler = None
    if balance == "sampler":
        train_targets = [targets[i] for i in train_idx]
        sampler = make_sampler(train_targets, len(classes))

    train_loader = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    tr_counts = np.bincount([targets[i] for i in train_idx], minlength=len(classes))
    va_counts = np.bincount([targets[i] for i in val_idx], minlength=len(classes))
    print("Train per-class:", dict(zip(classes, tr_counts.tolist())))
    print("Val   per-class:", dict(zip(classes, va_counts.tolist())))

    # also return what we need to compute class weights
    return train_loader, val_loader, classes, targets, train_idx

@torch.inference_mode()
def evaluate_cm(model, loader, device, num_classes):
    import numpy as np
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
    return cm, prec, rec, f1

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); correct=total=0
    for x,y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return correct/total if total else 0.0

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   # speed
    torch.backends.cudnn.deterministic = False 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/train")
    ap.add_argument("--out", default="models/resnet_straw5.pth")
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_full", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_balance", action="store_true")
    ap.add_argument("--resume", default="", help="optional .pth to warm start from")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--balance", choices=["weights","sampler","none"], default="weights",
                help="Use class-weighted loss, or WeightedRandomSampler, or none")
    ap.add_argument("--boost_classes", default="", 
                help='Optional weights boost: e.g. "healthy:1.6,overwatered:1.6,drought:1.0" (only used with --balance weights)')
    ap.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0=off, try 0.1)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0=off, try 0.998)")

    args = ap.parse_args()
    
    seed_everything(args.seed)
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loaders + split metadata
    train_loader, val_loader, classes, full_targets, train_idx = get_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers,
        args.val_frac, args.seed, args.balance
    )

    # class weights
    if args.balance == "weights":
        train_targets = [full_targets[i] for i in train_idx]
        counts = np.bincount(train_targets, minlength=len(classes)).astype(float)
        weights = counts.sum() / (counts + 1e-6)                 # inverse freq
        weights = weights / weights.mean()
        boosts = parse_boosts(args.boost_classes, classes)
        weights *= boosts
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)
    model.to(memory_format=torch.channels_last) 

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)

    # common loss (weights + smoothing) for both phases
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Phase 1: head only
    if args.epochs_head > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=0.05)
        best = 0.0

        for e in range(args.epochs_head):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                scaler.step(opt)
                scaler.update()

            acc = evaluate(model, val_loader, device)
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "classes": classes}, args.out)
            print(f"[head] {e+1}/{args.epochs_head}  val_acc={acc:.3f}")
    else:
        best = 0.0

    # --- Phase 2: selective unfreeze with per-layer LRs and clean BN handling ---

    # 0) Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 1) Unfreeze layer1–4 + fc
    for name, p in model.named_parameters():
        if any(k in name for k in ["layer1", "layer2", "layer3", "layer4", "fc"]):
            p.requires_grad = True

    # 2) Freeze BN stats/params *before* building param groups
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # 3) Build param groups: lower LR for early layers, no weight decay on 1D/bias
    def _make_param_groups(model, base_lr, wd):
        buckets = {"layer1": [], "layer2": [], "layer3": [], "layer4": [], "fc": []}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = "fc" if "fc" in n else next((k for k in ["layer1","layer2","layer3","layer4"] if k in n), None)
            if key is None:
                continue
            no_decay = (p.ndim == 1) or n.endswith(".bias")
            buckets[key].append((p, no_decay))

        lrs = {
            "layer1": base_lr * 0.25,
            "layer2": base_lr * 0.50,
            "layer3": base_lr * 0.75,
            "layer4": base_lr * 1.00,
            "fc":     base_lr * 1.00,
        }
        groups = []
        for k, params in buckets.items():
            if not params:
                continue
            decay_params   = [p for p, nd in params if not nd]
            nodecay_params = [p for p, nd in params if nd]
            if decay_params:
                groups.append({"params": decay_params,   "lr": lrs[k], "weight_decay": wd})
            if nodecay_params:
                groups.append({"params": nodecay_params, "lr": lrs[k], "weight_decay": 0.0})
        return groups

    pg = _make_param_groups(model, base_lr=args.lr_full, wd=0.05)
    opt = torch.optim.AdamW(pg)

    # (optional) sanity print
    print({i: (len(g["params"]), g["lr"], g["weight_decay"]) for i, g in enumerate(pg)})


    warmup_epochs = 2
    if args.epochs_full > warmup_epochs:
        warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=max(1, args.epochs_full - warmup_epochs))
        sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs_full))

    for e in range(args.epochs_full):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0
            )
            scaler.step(opt)
            scaler.update()

        sched.step()  # step LR once per epoch

        acc = evaluate(model, val_loader, device)
        cm, prec, rec, f1 = evaluate_cm(model, val_loader, device, num_classes=len(classes))
        print("val per-class:", {
            classes[i]: (round(float(prec[i]),3), round(float(rec[i]),3), round(float(f1[i]),3))
            for i in range(len(classes))
        })
        if acc > best:
            best = acc
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "meta": {
                    "img_size": args.img_size,
                    "val_frac": args.val_frac,
                    "label_smoothing": args.label_smoothing,
                    "class_weights": class_weights.detach().cpu().tolist(),
                }
            }, args.out)

        print(f"[full] {e+1}/{args.epochs_full}  lr={opt.param_groups[0]['lr']:.2e}  val_acc={acc:.3f}")

    # ---- after training: reload BEST and calibrate ONCE ----
    best_ckpt = torch.load(args.out, map_location="cpu")
    model.load_state_dict(best_ckpt["model"], strict=False)
    model.to(device)
    temperature = fit_temperature(model, val_loader, device)
    best_ckpt["temperature"] = float(temperature)
    torch.save(best_ckpt, args.out)
    print(f"Saved temperature={temperature:.3f} into {args.out}")
    print(f"Saved best to {args.out}  (best_val_acc={best:.3f})")


@torch.inference_mode(False)
def fit_temperature(model, loader, device):
    model.eval()
    T = torch.nn.Parameter(torch.ones([], device=device))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure():
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_n = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            logits = model(x) / T.clamp_min(1e-3)
            total_loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum")
            total_n += y.size(0)
        loss = total_loss / max(1, total_n)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().cpu())



if __name__ == "__main__":
    main()

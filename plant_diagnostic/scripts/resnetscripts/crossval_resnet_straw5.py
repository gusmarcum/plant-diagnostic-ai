#!/usr/bin/env python3
import argparse, random, math, numpy as np
from pathlib import Path
from collections import defaultdict

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models

IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}

# ---------------------- transforms ----------------------
def build_transforms(img_size):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf_train, tf_val

# ---------------------- dataset utils -------------------
def load_imagefolder(root, tf):
    if not any(Path(root).rglob("*"+ext) for ext in IMG_EXTS):
        raise SystemExit(f"No images under {root}")
    return datasets.ImageFolder(root=root, transform=tf)

def make_stratified_kfold_indices(targets, num_classes, k, seed=42):
    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    for i, y in enumerate(targets):
        per_class[y].append(i)
    for lst in per_class: rng.shuffle(lst)

    folds = [list() for _ in range(k)]
    # round-robin distribute each class across k folds
    for c in range(num_classes):
        for i, idx in enumerate(per_class[c]):
            folds[i % k].append(idx)
    # now build (train_idx, val_idx) per fold
    out = []
    all_idx = set(range(len(targets)))
    for i in range(k):
        val_idx = sorted(folds[i])
        train_idx = sorted(all_idx - set(val_idx))
        # guard: ensure no empty train class
        cls_counts = np.bincount([targets[j] for j in train_idx], minlength=num_classes)
        for c in range(num_classes):
            if cls_counts[c] == 0 and len(folds[i])>0:
                # move one from some other fold into train (rare)
                for j in range(k):
                    if j==i: continue
                    cand = next((x for x in folds[j] if targets[x]==c), None)
                    if cand is not None:
                        train_idx.append(cand)
                        folds[j].remove(cand)
                        break
        out.append((sorted(train_idx), val_idx))
    return out

def make_sampler(train_targets, num_classes):
    counts = np.bincount(train_targets, minlength=num_classes).astype(np.float64)
    inv = np.zeros_like(counts); nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    weights = np.array([inv[t] for t in train_targets], dtype=np.float64)
    weights[weights <= 0.0] = 1e-8
    return WeightedRandomSampler(torch.from_numpy(weights).double(),
                                 num_samples=len(weights),
                                 replacement=True)

# ---------------------- metrics -------------------------
def confusion_matrix(num_classes):
    return torch.zeros((num_classes, num_classes), dtype=torch.int64)

@torch.inference_mode()
def evaluate_probs(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).softmax(1).cpu()
        all_p.append(p); all_y.append(y)
    P = torch.cat(all_p, 0); Y = torch.cat(all_y, 0)
    return P, Y

def cm_from_preds(P, Y, num_classes):
    pred = P.argmax(1)
    cm = confusion_matrix(num_classes)
    for t, p in zip(Y, pred):
        cm[t, p] += 1
    return cm.numpy()

def prf_from_cm(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
    rec  = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
    f1   = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec)!=0)
    return prec, rec, f1

def sweep_threshold(P, Y, focus_idxs, lo=0.5, hi=0.8, steps=31):
    # choose acceptance threshold that maximizes macro-F1 on focus classes
    best_f1, best_th = 0.0, 0.65
    for th in np.linspace(lo, hi, steps):
        tp=fp=fn=0
        for i in range(len(Y)):
            p1 = float(P[i].max()); c1 = int(P[i].argmax()); y = int(Y[i])
            if p1 < th:  # abstain
                continue
            for c in focus_idxs:
                if y==c and c1==c: tp+=1
                elif y!=c and c1==c: fp+=1
                elif y==c and c1!=c: fn+=1
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        if f1 > best_f1: best_f1, best_th = f1, th
    return best_f1, best_th

# ---------------------- training ------------------------
def train_one_fold(args, classes, train_idx, val_idx, ds_train_full, ds_val_full, device):
    # subsets
    ds_tr = Subset(ds_train_full, train_idx)   # transform = train
    ds_va = Subset(ds_val_full,   val_idx)     # transform = val

    # sampler (optional)
    targets_all = getattr(ds_train_full, "targets", [c for _, c in ds_train_full.samples])
    train_targets = [targets_all[i] for i in train_idx]
    sampler = make_sampler(train_targets, len(classes)) if args.sample_balance else None

    train_loader = DataLoader(ds_tr, batch_size=args.batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ds_va, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # class weights from this fold's train set
    counts = np.bincount(train_targets, minlength=len(classes)).astype(float)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    # optional manual boosts for weak classes
    if args.boost:
        for name, mult in args.boost.items():
            if name in classes:
                w[classes.index(name)] *= mult
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # Phase 1: head only
    if args.epochs_head > 0:
        for p in model.parameters(): p.requires_grad = False
        for p in model.fc.parameters(): p.requires_grad = True
        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=0.05)
        best_acc = 0.0
        for e in range(args.epochs_head):
            model.train()
            for x,y in train_loader:
                x,y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward(); opt.step()
            acc = evaluate(model, val_loader, device)
            best_acc = max(best_acc, acc)

    # Phase 2: unfreeze layer2–4
    for name, p in model.named_parameters():
        p.requires_grad = False
        if any(k in name for k in ["layer2","layer3","layer4","fc"]):
            p.requires_grad = True
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr_full, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,args.epochs_full))

    best_acc = 0.0
    best_state = None
    for e in range(args.epochs_full):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {"model": model.state_dict(), "classes": classes}

    # final eval with best weights
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    P, Y = evaluate_probs(model, val_loader, device)
    cm = cm_from_preds(P, Y, len(classes))
    prec, rec, f1 = prf_from_cm(cm)

    return {
        "best_val_acc": float(best_acc),
        "cm": cm, "prec": prec, "rec": rec, "f1": f1,
        "P": P, "Y": Y,  # for threshold sweep
        "state": best_state
    }

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); correct=total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return correct/total if total else 0.0

# ---------------------- main ----------------------------
def parse_boost(s):
    # "healthy:1.5,overwatered:1.7"
    if not s: return {}
    out={}
    for part in s.split(","):
        name, mult = part.split(":")
        out[name.strip()] = float(mult)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/train")
    ap.add_argument("--out_dir", default="models/cv_resnet")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=256)

    ap.add_argument("--epochs_head", type=int, default=3)
    ap.add_argument("--epochs_full", type=int, default=10)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--sample_balance", action="store_true")
    ap.add_argument("--boost_classes", default="", help='e.g. "healthy:1.5,overwatered:1.7"')

    args = ap.parse_args()
    args.boost = parse_boost(args.boost_classes)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_train, tf_val = build_transforms(args.img_size)
    ds_train_full = load_imagefolder(args.data_root, tf_train)
    ds_val_full   = load_imagefolder(args.data_root, tf_val)   # same files, val transform
    classes = ds_train_full.classes
    targets = getattr(ds_train_full, "targets", [c for _, c in ds_train_full.samples])

    folds = make_stratified_kfold_indices(targets, len(classes), args.k, args.seed)

    # Run folds
    fold_results = []
    for i,(tr_idx, va_idx) in enumerate(folds):
        print(f"\n=== Fold {i+1}/{args.k} ===")
        # print per-class counts for sanity
        tr_counts = np.bincount([targets[j] for j in tr_idx], minlength=len(classes))
        va_counts = np.bincount([targets[j] for j in va_idx], minlength=len(classes))
        print("Train per-class:", dict(zip(classes, tr_counts.tolist())))
        print("Val   per-class:", dict(zip(classes, va_counts.tolist())))

        res = train_one_fold(args, classes, tr_idx, va_idx, ds_train_full, ds_val_full, device)
        fold_results.append(res)

        # save best checkpoint per fold
        fold_dir = Path(args.out_dir)/f"fold_{i+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if res["state"] is not None:
            torch.save(res["state"], fold_dir/"best.pth")

        # print fold summary
        acc = res["best_val_acc"]
        cm  = res["cm"]; prec, rec, f1 = res["prec"], res["rec"], res["f1"]
        print(f"Fold {i+1} best_acc={acc:.3f}")
        for ci, name in enumerate(classes):
            print(f"{name:12s}  P={prec[ci]:.3f}  R={rec[ci]:.3f}  F1={f1[ci]:.3f}  support={cm[ci].sum()}")

        # threshold sweep for healthy+overwatered
        focus = [c for c in ["healthy","overwatered"] if c in classes]
        if focus:
            idxs = [classes.index(n) for n in focus]
            best_f1, best_th = sweep_threshold(res["P"], res["Y"], idxs)
            print(f"Fold {i+1}: best_macroF1({'+'.join(focus)})={best_f1:.3f} at threshold={best_th:.2f}")

    # Aggregate
    accs = [r["best_val_acc"] for r in fold_results]
    mean_acc = float(np.mean(accs)); std_acc = float(np.std(accs))
    print("\n=== Cross-val summary ===")
    print(f"acc: mean={mean_acc:.3f}  std={std_acc:.3f}  over {args.k} folds")

    # Per-class recall mean±std
    rec_stack = np.stack([r["rec"] for r in fold_results], axis=0)  # [k, C]
    for ci, name in enumerate(classes):
        m = rec_stack[:,ci].mean(); s = rec_stack[:,ci].std()
        print(f"recall[{name}]: {m:.3f} ± {s:.3f}")

if __name__ == "__main__":
    main()

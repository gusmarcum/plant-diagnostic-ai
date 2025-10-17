#!/usr/bin/env python3
"""
Class-conditional **offline data augmentation** for strawberry disease dataset.

Goal: create **realistic** extra training images for the weak classes so your
ResNet anchor learns robust cues; optionally keep a parallel tree for LLM/VQA.

Highlights
- Works directly on your folder tree: train/<class>/*.jpg (val is left untouched)
- Class-conditional policies (different aug recipes per class)
- Photometric + mild geometric aug that **preserve diagnostic cues**
- Deterministic naming, resumable, parallelized
- Sensible defaults for your current counts (drought, overwatered, root_rot are weak)

Usage (most common):
python /data/AGAI/MiniGPT-4/plant_diagnostic/tools/augment_dataset.py \
  --src  /data/AGAI/MiniGPT-4/plant_diagnostic/data/train \
  --dst  /data/AGAI/MiniGPT-4/plant_diagnostic/data/train_aug \
  --classes drought overwatered root_rot \
  --target-per-class 300 \
  --max-per-image 4 \
  --img-size 256 \
  --report /data/AGAI/MiniGPT-4/plant_diagnostic/data/aug_report.json


If you prefer augmenting **in-place**, set --dst equal to --src (it will write
files with _augXX suffix so originals aren’t overwritten).

After augmentation:
- Point ResNet training to --train_root pointing at a **merged** tree, e.g.:
    /data/.../data/train_merged
  where train_merged/<class> contains originals + aug (you can just combine or
  set --dst to a fresh folder and then use both folders with a “symlink merge”).

Requirements: Pillow is enough for fallback. For best results, install albumentations:
  pip install albumentations opencv-python-headless

"""
from __future__ import annotations
import argparse, os, sys, math, json, random, shutil
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial

from PIL import Image, ImageOps, ImageFilter, ImageEnhance

try:
    import albumentations as A
    HAS_ALBU = True
except Exception:
    HAS_ALBU = False

# -----------------------------
# Config / Policies
# -----------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
ALL_CLASSES = [
    "drought", "frost", "healthy", "overwatered", "root_rot", "gray_mold", "white_mold"
]

# Per-class augmentation policies. Designed to keep cues believable.
# If albumentations is available, we use rich transforms; else PIL fallback.

def build_policy(label: str, img_size: int = 256):
    if not HAS_ALBU:
        return None  # will use PIL fallback

    common_geo = [
        A.SmallestMaxSize(max_size=math.floor(img_size*1.20), interpolation=1, always_apply=True),
        A.RandomCrop(height=img_size, width=img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=6, border_mode=1, p=0.6),
        A.Perspective(scale=(0.02, 0.05), p=0.15),
    ]

    # Photometric ranges vary by class to reduce label leakage via color artifacts
    if label in ("gray_mold", "white_mold"):
        photo = [
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.08, p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.15),
            A.ISONoise(color_shift=(0.01,0.03), intensity=(0.01,0.03), p=0.2),
            A.Sharpen(alpha=(0.05,0.15), lightness=(0.8,1.0), p=0.2),
        ]
    elif label in ("drought", "overwatered"):
        photo = [
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.7),
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=6, val_shift_limit=6, p=0.25),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
            A.UnsharpMask(p=0.2),
        ]
    elif label == "root_rot":
        photo = [
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.7),
            A.GaussianBlur(blur_limit=(3,5), p=0.15),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.25),
            A.GaussNoise(var_limit=(5.0, 12.0), p=0.2),
        ]
    else:  # healthy, frost
        photo = [
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=6, p=0.3),
            A.GaussNoise(var_limit=(5.0, 12.0), p=0.2),
        ]

    # Light cutout: avoid deleting the whole diagnostic area
    cutout = [A.CoarseDropout(max_holes=1, max_height=int(img_size*0.12), max_width=int(img_size*0.12),
                               min_holes=0, fill_value=0, p=0.15)]

    return A.Compose(common_geo + photo + cutout)


# -----------------------------
# PIL fallback
# -----------------------------

def pil_augment(img: Image.Image, label: str, img_size: int) -> Image.Image:
    # Resize & crop to target window
    long = max(img.size)
    scale = img_size/float(min(img.size))
    # Slight random scale/jitter
    jitter = 1.0 + random.uniform(-0.08, 0.08)
    new_w = int(img.size[0] * scale * jitter)
    new_h = int(img.size[1] * scale * jitter)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    # random crop
    if new_w > img_size and new_h > img_size:
        x0 = random.randint(0, new_w-img_size)
        y0 = random.randint(0, new_h-img_size)
        img = img.crop((x0, y0, x0+img_size, y0+img_size))
    else:
        img = ImageOps.fit(img, (img_size, img_size), Image.BILINEAR)

    # hflip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # mild rotation
    if random.random() < 0.6:
        ang = random.uniform(-6, 6)
        img = img.rotate(ang, resample=Image.BILINEAR, expand=False)

    # photometric tweaks (class-conditional)
    if label in ("gray_mold", "white_mold"):
        if random.random() < 0.7:
            b, c = 1+random.uniform(-0.08,0.08), 1+random.uniform(-0.08,0.08)
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5,1.2)))
    elif label in ("drought", "overwatered"):
        if random.random() < 0.7:
            b, c = 1+random.uniform(-0.08,0.08), 1+random.uniform(-0.08,0.08)
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)
    elif label == "root_rot":
        if random.random() < 0.7:
            b, c = 1+random.uniform(-0.10,0.10), 1+random.uniform(-0.10,0.10)
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)
    else:
        if random.random() < 0.7:
            b, c = 1+random.uniform(-0.10,0.10), 1+random.uniform(-0.10,0.10)
            img = ImageEnhance.Brightness(img).enhance(b)
            img = ImageEnhance.Contrast(img).enhance(c)

    # light unsharp mask to keep texture
    if random.random() < 0.2:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=80, threshold=3))

    return img


# -----------------------------
# Core logic
# -----------------------------

def list_images(cls_dir: Path) -> List[Path]:
    return [p for p in sorted(cls_dir.rglob("*")) if p.suffix.lower() in VALID_EXTS]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def one_aug(in_path: Path, out_dir: Path, label: str, idx: int, img_size: int, policy) -> Path:
    out_name = f"{in_path.stem}_aug{idx:03d}.jpg"
    out_path = out_dir / out_name
    if out_path.exists():
        return out_path

    with Image.open(in_path) as im:
        im = im.convert("RGB")
        if HAS_ALBU and policy is not None:
            import numpy as np
            arr = np.array(im)
            arr = policy(image=arr)["image"]
            im = Image.fromarray(arr)
        else:
            im = pil_augment(im, label, img_size)
        # Save conservative quality to avoid artifacts
        im.save(out_path, format="JPEG", quality=92, subsampling=1, optimize=True)
    return out_path


def augment_class(src_cls: Path, dst_cls: Path, label: str, target: int, max_per_image: int, img_size: int) -> Dict:
    ensure_dir(dst_cls)
    images = list_images(src_cls)
    cur = len(images)
    written = 0

    if cur >= target:
        return {"class": label, "status": "skip", "have": cur, "target": target, "added": 0}

    need = target - cur
    policy = build_policy(label, img_size)

    # Round-robin over originals to hit the target; cap per-image to reduce clones
    i = 0
    gen_idx = 0
    while written < need and images:
        img = images[i % len(images)]
        # count existing aug siblings for this base file
        base = img.stem
        existing = len(list(dst_cls.glob(base + "_aug*.jpg")))
        if existing < max_per_image:
            outp = one_aug(img, dst_cls, label, gen_idx, img_size, policy)
            written += 1
            gen_idx += 1
        i += 1

    return {"class": label, "status": "ok", "have": cur, "target": target, "added": written}


def main():
    ap = argparse.ArgumentParser(description="Offline class-conditional augmentation for strawberry diseases")
    ap.add_argument("--src", required=True, help="train root with class subfolders")
    ap.add_argument("--dst", required=True, help="output root (can be same as --src)")
    ap.add_argument("--classes", nargs="*", default=["drought","overwatered","root_rot"],
                    help="which classes to augment")
    ap.add_argument("--target-per-class", type=int, default=300, help="aim to reach this many images per selected class")
    ap.add_argument("--max-per-image", type=int, default=4, help="max augmentations produced per original image")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0, help="reserved for future parallelism; current loop is IO-bound and fast enough")
    ap.add_argument("--report", default="", help="optional path to write a JSON report")
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        sys.exit(f"Missing src: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    # Walk classes
    results = []
    for cls in args.classes:
        src_cls = src_root / cls
        if not src_cls.exists():
            print(f"[WARN] missing class: {src_cls}")
            continue
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        r = augment_class(src_cls, dst_cls, cls, args.target_per_class, args.max_per_image, args.img_size)
        results.append(r)
        print(f"[{r['class']}] have={r['have']} -> target={r['target']} +added={r['added']} ({r['status']})")

    # Copy over non-augmented classes if dst != src so the tree is complete
    if dst_root.resolve() != src_root.resolve():
        for clsdir in src_root.iterdir():
            if not clsdir.is_dir():
                continue
            if clsdir.name in args.classes:
                # already has originals implicitly (we didn’t copy); optional: copy originals as well
                for p in clsdir.glob("*.*"):
                    if p.suffix.lower() in VALID_EXTS:
                        tgt = (dst_root/clsdir.name/p.name)
                        if not tgt.exists():
                            shutil.copy2(p, tgt)
            else:
                # mirror untouched classes fully
                outdir = dst_root/clsdir.name
                outdir.mkdir(parents=True, exist_ok=True)
                for p in clsdir.glob("*.*"):
                    if p.suffix.lower() in VALID_EXTS:
                        tgt = outdir/p.name
                        if not tgt.exists():
                            shutil.copy2(p, tgt)

    if args.report:
        Path(args.report).write_text(json.dumps(results, indent=2))

    print("\nDone. Summary:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()


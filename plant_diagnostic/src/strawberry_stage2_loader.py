# /data/kiriti/MiniGPT-4/plant_diagnostic/scripts/strawberry_stage2_loader.py
import json
import collections
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


class StrawberryStage2Dataset(Dataset):
    """
    Expects Stage-2 chat JSON:
      [
        {
          "image": "data/train/<class>/<file>.jpg",
          "conversations": [
             {"from":"human","value":"<Img><ImageHere></Img> [condition: X] ..."},
             {"from":"gpt","value":"..."},
             ...
          ]
        }, ...
      ]
    """
    def __init__(self, ann_path: str, image_root: str = ".", image_size: int = 224):
        self.items = json.load(open(ann_path, "r"))
        self.root = Path(image_root)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),  # BLIP-ish default
        ])
        # class from parent directory of the image path
        self.labels = [Path(it["image"]).parts[-2] for it in self.items]
        self.class_counts = collections.Counter(self.labels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img_path = self.root / it["image"]
        with Image.open(img_path).convert("RGB") as im:
            img = self.transform(im)
        return {
            "image": img,                               # tensor CxHxW
            "image_path": str(img_path),
            "label": Path(it["image"]).parts[-2],       # class name
            "conversations": it["conversations"],       # list of turns
        }


def make_weighted_sampler(labels):
    """
    Inverse-frequency sampling so minority classes show up more.
    """
    freq = collections.Counter(labels)
    n_cls = len(freq)
    n_total = len(labels)
    class_w = {c: n_total / (n_cls * cnt) for c, cnt in freq.items()}
    weights = [class_w[c] for c in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


def build_loader(
    ann_path: str,
    image_root: str = ".",
    batch_size: int = 8,
    image_size: int = 224,
    num_workers: int = 4,
    weighted: bool = True,
    drop_last: bool = True,
):
    """
    Convenience factory. Returns (loader, dataset).
    Set weighted=False if you want plain shuffled sampling.
    """
    ds = StrawberryStage2Dataset(ann_path, image_root, image_size)
    if weighted:
        sampler = make_weighted_sampler(ds.labels)
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=sampler, shuffle=False,
            num_workers=num_workers, drop_last=drop_last, pin_memory=True
        )
    else:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=drop_last, pin_memory=True
        )
    return loader, ds


if __name__ == "__main__":
    # quick smoke test
    ann = "datasets/strawberry_stage2.json"   # your converted file
    img_root = "."                            # repo root containing data/train/...
    loader, ds = build_loader(ann, img_root, batch_size=8, weighted=True)

    print("Total samples:", len(ds))
    print("Class counts:", ds.class_counts)

    for i, batch in enumerate(loader):
        print(f"[{i}] images:", batch["image"].shape, "labels:", list(batch["label"])[:8])
        if i == 1:
            break


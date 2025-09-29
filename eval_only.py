#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.tasks import setup_task

def build_single_loader(ds, batch_size, num_workers=0):
    if isinstance(batch_size, (list, tuple)):
        batch_size = int(batch_size[0] if batch_size else 2)
    if batch_size is None or int(batch_size) <= 0:
        batch_size = 2
    batch_size = int(batch_size)
    collate_fn = getattr(ds, "collater", None)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=int(num_workers or 0), pin_memory=True,
                      collate_fn=collate_fn)

def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:    print("  missing[:10]:", missing[:10])
    if unexpected: print("  unexpected[:10]:", unexpected[:10])
    return model

def resolve_split(datasets, want):
    if want in datasets:
        return want, datasets[want]
    keys = list(datasets.keys())
    if len(keys) == 1:
        print(f"[eval] requested '{want}' not found; using '{keys[0]}'")
        return keys[0], datasets[keys[0]]
    raise SystemExit(f"[eval] split '{want}' not found. available: {keys}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-path", required=True, dest="cfg_path")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("options", nargs="*")
    args = ap.parse_args()
    if not hasattr(args, "options"): args.options = None

    cfg = Config(args)
    task = setup_task(cfg)
    datasets = task.build_datasets(cfg)
    print("[eval] dataset keys:", list(datasets.keys()))
    split, ds = resolve_split(datasets, args.split)

    # batch size from cfg if available; else fallback
    bsz = 2
    try:
        # many configs store under the dataset name (e.g., 'llava_conversation')
        bsz = getattr(cfg.datasets_cfg, list(datasets.keys())[0]).get("batch_size", bsz)
    except Exception:
        pass

    loader = build_single_loader(ds, bsz, num_workers=0)

    model = registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device).eval()
    load_ckpt(model, args.ckpt, device)

    task.before_evaluation(model=model, dataset=ds)
    with torch.no_grad():
        print("[eval] running task.evaluation …")
        results = task.evaluation(model, loader)

    log = None
    try:
        log = task.after_evaluation(val_result=results, split_name=split, epoch="eval_only")
    except Exception as e:
        print("[eval] after_evaluation skipped/failed:", e)

    payload = {}
    if isinstance(results, dict): payload.update(results)
    elif results is not None:      payload["result"] = results
    if isinstance(log, dict):      payload.update(log)
    payload.setdefault("split", split)
    try: payload.setdefault("num_samples", len(loader.dataset))
    except Exception: pass

    try:
        print(json.dumps(payload, indent=2))
    except Exception as e:
        print("[eval] could not JSON-serialize payload:", e)
        print(str(payload))

    try:
        # prefer result_dir if registry populated during Config init
        result_dir = registry.get_path("result_dir") or registry.get_path("output_dir") or "./result"
    except Exception:
        result_dir = "./result"
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    out = Path(result_dir) / f"eval_{split}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[eval] wrote {out}")

if __name__ == "__main__":
    main()

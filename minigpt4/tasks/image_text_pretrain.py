import json
import io
import os
import logging
import torch
from torch.utils.data import DataLoader

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask


def _make_indexable(ds):
    """Force any mapping/generator/custom iterable into something indexable for DataLoader."""
    if hasattr(ds, "values") and callable(getattr(ds, "values", None)):
        try:
            ds = list(ds.values())
        except Exception:
            ds = list(ds)
    elif hasattr(ds, "__iter__") and not hasattr(ds, "__getitem__"):
        ds = list(ds)
    try:
        _ = ds[0]
    except Exception:
        try:
            ds = list(ds)
        except Exception:
            ds = list(iter(ds))
    return ds


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.current_step = 0
        # stash cfg if someone passes it
        self.config = kwargs.get("config", None)

    @torch.no_grad()
    def evaluation(self, model, data_loader, cuda_enabled: bool = True):
        """
        Minimal, robust evaluation that won’t crash demo:
        - Accepts DataLoader, Dataset, or nested dict; unwraps to an indexable sequence
        - Uses batch_size=1 + unwrap collate to avoid custom collate issues
        - Returns a tiny dict; demo UIs generally don’t use this anyway
        """
        # 1) device
        try:
            device = getattr(model, "device", None) or next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _to(x):
            if hasattr(x, "to"):
                return x.to(device) if cuda_enabled else x
            return x

        # 2) unwrap dicts
        ds = data_loader
        if isinstance(ds, dict):
            if "val" in ds:
                ds = ds["val"]
            elif len(ds) == 1:
                _only = next(iter(ds.values()))
                if isinstance(_only, dict) and "val" in _only:
                    ds = _only["val"]
                else:
                    ds = _only

        # 3) force indexable
        ds = _make_indexable(ds)

        # 4) loader
        loader = ds if isinstance(ds, DataLoader) else DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b[0]
        )

        # 5) quick pass to prove it iterates (demo doesn’t need full eval)
        n = 0
        for batch in loader:
            # move tensors if present (kept minimal)
            if isinstance(batch, dict):
                batch = {k: _to(v) for k, v in batch.items()}
            else:
                batch = _to(batch)
            n += 1
            break

        # 6) return a small dict so callers don’t choke
        return {"num_samples": n, "agg_metrics": 0.0}

    def after_evaluation(self, val_result, split_name, epoch):
        # keep keys present for any logger expecting them
        if not isinstance(val_result, dict):
            val_result = {}
        val_result.setdefault("agg_metrics", 0.0)
        val_result.setdefault("num_samples", 0)
        return val_result

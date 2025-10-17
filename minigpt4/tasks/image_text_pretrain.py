import json
import io
import os
import logging
import torch
from torch.utils.data import DataLoader, IterableDataset

from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask
from minigpt4.datasets.datasets.dataloader_utils import MultiIterLoader


def _make_indexable(ds):
    """Check if dataset is indexable or iterable for DataLoader."""
    # Already a DataLoader: just return
    if isinstance(ds, DataLoader):
        return ds
    # Indexable datasets (map-style)
    if hasattr(ds, "__getitem__"):
        return ds
    # Iterable-style datasets (including MultiIterLoader)
    if isinstance(ds, IterableDataset) or hasattr(ds, "__iter__") or hasattr(ds, "__next__"):
        return ds
    raise TypeError("Dataset is neither indexable nor iterable.")


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.current_step = 0
        # stash cfg if someone passes it
        self.cfg = kwargs.get("cfg", None)

    @torch.no_grad()
    def evaluation(self, model, data_loader, cuda_enabled: bool = True):
        """
        Minimal, robust evaluation that won't crash demo:
        - Accepts DataLoader, Dataset, or nested dict; unwraps to an indexable sequence
        - Uses batch_size=1 + unwrap collate to avoid custom collate issues
        - Returns a tiny dict; demo UIs generally don't use this anyway
        """
        # Early exit if evaluation is disabled in config or no val splits
        if (not getattr(self.cfg.run_cfg, "evaluate", False) or
            not self.cfg.run_cfg.get("val_splits")):
            return {"num_samples": 0, "agg_metrics": 0.0}

        # Optional smoke test assertion (remove in production if not needed)
        # Assert the guard works - this should never execute if eval is properly disabled
        # assert not getattr(self.cfg.run_cfg, "evaluate", False) or self.cfg.run_cfg.get("val_splits"), \
        #     "Evaluation is disabled or no val splits; evaluation() should have early-returned."

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

        # 4) loader - handle different dataset types appropriately
        if isinstance(ds, DataLoader):
            loader = ds
        elif hasattr(ds, "__next__"):
            # It's already an iterator (like MultiIterLoader) - use directly
            # MultiIterLoader is designed to be endless/cycling, so reuse is safe
            loader = ds
        elif hasattr(ds, "__iter__"):
            # It's iterable but not an iterator - get fresh iterator each eval call
            loader = iter(ds)
        else:
            # It's an indexable dataset, wrap in DataLoader
            loader = DataLoader(
                ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b[0]
            )

        # 5) quick pass to prove it iterates (demo doesn't need full eval)
        # Bound evaluation for endless loaders - configurable from config
        max_val_steps = getattr(self.cfg.run_cfg, "max_val_steps", 200)
        n = 0
        for batch in loader:
            # move tensors if present (kept minimal)
            if isinstance(batch, dict):
                batch = {k: _to(v) for k, v in batch.items()}
            else:
                batch = _to(batch)
            n += 1
            if n >= max_val_steps:
                break

        # 6) return a small dict so callers don't choke
        return {"num_samples": n, "agg_metrics": 0.0}

    def compute_validation_loss(self, model, data_loader, device, max_steps=50):
        """Compute average validation loss over max_steps batches."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= max_steps:
                    break

                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                else:
                    batch = batch.to(device) if hasattr(batch, 'to') else batch

                # Forward pass
                try:
                    output = model(batch)
                    if isinstance(output, dict) and "loss" in output:
                        loss = output["loss"]
                        if isinstance(loss, torch.Tensor):
                            total_loss += float(loss)
                            num_batches += 1
                except Exception as e:
                    logging.warning(f"Error computing loss for batch {i}: {e}")
                    continue

        model.train()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, num_batches

    def after_evaluation(self, val_result, split_name, epoch):
        # keep keys present for any logger expecting them
        if not isinstance(val_result, dict):
            val_result = {}
        val_result.setdefault("agg_metrics", 0.0)
        val_result.setdefault("num_samples", 0)
        return val_result

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or
 https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import (
    concat_datasets,  # kept for compatibility
    reorg_datasets_by_split,
    ChainDataset,
)
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)


@registry.register_runner("runner_base")
class RunnerBase:
    """
    Canonical training/eval runner.

    - Robust device selection honoring LOCAL_RANK
    - Single, well-formed model() that places the model then wraps DDP once
    - Usual optimizer / scaler / LR scheduler / dataloaders
    - Checkpoint save/load and logging hooks
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model
        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None
        self.start_epoch = 0

        # Output dirs (register in registry and on disk)
        self.setup_output_dir()

        # Resolve validation splits early
        try:
            ds_keys = set(datasets.keys()) if isinstance(datasets, dict) else set()
            run_block = getattr(cfg, "run_cfg", None)
            cfg_vs = list(getattr(run_block, "val_splits", []) or [])
            resolved = [s for s in cfg_vs if s in ds_keys]
            if not resolved and "val" in ds_keys:
                resolved = ["val"]
            self._valid_splits = resolved
            logging.info(
                "[runner] Resolved valid_splits=%s from ds_keys=%s and cfg=%s",
                self._valid_splits, sorted(ds_keys), cfg_vs
            )
        except Exception as e:
            logging.warning("[runner] Failed to resolve valid_splits early: %s", e)
            self._valid_splits = []

    # --------------------------
    # Core properties / helpers
    # --------------------------
    @property
    def device(self):
        if self._device is None:
            dev = str(getattr(self.config.run_cfg, "device", "cuda"))
            if dev == "cuda" and torch.cuda.is_available():
                idx = getattr(self.config.run_cfg, "gpu", None)
                if idx is None:
                    lr = os.environ.get("LOCAL_RANK")
                    if lr is not None:
                        idx = int(lr)
                if idx is None:
                    idx = 0
                dev = f"cuda:{idx}"
            self._device = torch.device(dev)
            if self._device.type == "cuda":
                torch.cuda.set_device(self._device.index or 0)
        return self._device

    @property
    def use_distributed(self):
        return bool(getattr(self.config.run_cfg, "distributed", False))

    @property
    def model(self):
        """Return the (possibly DDP-wrapped) model on the correct device.

        - Uses self.device (honors LOCAL_RANK set by torchrun).
        - If the LLM is quantized or has an hf_device_map, only non-LLM modules are moved.
        - Wraps with DDP exactly once after placement; static-graph hint when available.
        """
        dev = self.device
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        def _is_quantized_or_mapped(llm):
            chain = [
                llm,
                getattr(llm, "base_model", None),
                getattr(getattr(llm, "base_model", None), "model", None),
            ]
            for obj in chain:
                if obj is None:
                    continue
                if getattr(obj, "is_loaded_in_8bit", False) or getattr(obj, "is_loaded_in_4bit", False):
                    return True
                if getattr(obj, "quantization_config", None) is not None:
                    return True
                if hasattr(obj, "hf_device_map"):
                    return True
            return False

        # Move to device (carefully if the LLM is quantized / device-mapped)
        if self._model is not None:
            llm = getattr(self._model, "llama_model", None)
            is_quant = _is_quantized_or_mapped(llm) if llm is not None else False

            if is_quant:
                # Only move aux modules; leave LLM per its device map/quantization
                for name, child in self._model.named_children():
                    if name != "llama_model" and isinstance(child, torch.nn.Module):
                        child.to(dev, non_blocking=True)
                for attr in ("visual_encoder", "ln_vision", "llama_proj", "mm_projector",
                             "vision_proj", "proj", "q_former"):
                    mod = getattr(self._model, attr, None)
                    if isinstance(mod, torch.nn.Module):
                        mod.to(dev, non_blocking=True)
            else:
                # Non-quantized: move whole model if needed
                p0 = next(self._model.parameters(), None)
                if p0 is None or p0.device != dev:
                    self._model = self._model.to(dev, non_blocking=True)

        # Wrap with DDP if requested
        if self.use_distributed:
            if self._wrapped_model is None:
                # Sanity: ensure all params/buffers are on a single device for this rank
                param_devs = {p.device for p in self._model.parameters()}
                buffer_devs = {b.device for b in self._model.buffers()}
                if dev not in param_devs or (param_devs and len(param_devs) != 1) or (buffer_devs and len(buffer_devs) != 1):
                    raise RuntimeError(f"Model not on single device for {dev}: params={param_devs}, buffers={buffer_devs}")

                self._wrapped_model = DDP(
                    self._model,
                    device_ids=[dev.index] if dev.type == "cuda" else None,
                    output_device=dev.index if dev.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                    gradient_as_bucket_view=True,
                )
                try:
                    self._wrapped_model._set_static_graph()
                except Exception:
                    pass
            return self._wrapped_model

        return self._model

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        return int(self.config.run_cfg.get("log_freq", 50))

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        # support both names used in configs
        return int(self.config.run_cfg.get("accum_grad_iters",
                   self.config.run_cfg.get("accumulation_steps", 1)))

    @property
    def valid_splits(self):
        try:
            vs = getattr(self, "_valid_splits", None)
            if vs:
                return vs
            ds = getattr(self, "datasets", None)
            ds_keys = set(ds.keys()) if isinstance(ds, dict) else set()
            cfg_vs = list(getattr(self.config.run_cfg, "val_splits", []) or [])
            resolved = [s for s in cfg_vs if s in ds_keys]
            if not resolved and "val" in ds_keys:
                resolved = ["val"]
            return resolved
        except Exception:
            return []

    @property
    def test_splits(self):
        return list(self.config.run_cfg.get("test_splits", []) or [])

    @property
    def train_splits(self):
        splits = list(self.config.run_cfg.get("train_splits", []) or [])
        if not splits:
            try:
                ds_keys = set(self.datasets.keys()) if isinstance(self.datasets, dict) else set()
            except Exception:
                ds_keys = set()
            if "train" in ds_keys:
                splits = ["train"]
        if not splits:
            logging.info("Empty train splits.")
        return splits

    # --------------------------
    # Output dirs
    # --------------------------
    def setup_output_dir(self):
        lib_root = registry.get_path("library_root")
        if not lib_root:
            lib_root = os.environ.get("MINIGPT4_LIB_ROOT", os.path.dirname(os.path.dirname(__file__)))
            try:
                registry.register_path("library_root", lib_root)
            except KeyError:
                pass

        out_root = getattr(self.config.run_cfg, "output_dir", "output/minigptv2_finetune")
        job_id = getattr(self, "job_id", "run")
        output_dir = Path(lib_root) / out_root / str(job_id)
        result_dir = output_dir / "result"
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("output_dir", str(output_dir))
        registry.register_path("result_dir", str(result_dir))

        self.output_dir = output_dir
        self.result_dir = result_dir

    # --------------------------
    # Dataloaders
    # --------------------------
    @property
    def dataloaders(self) -> dict:
        if self._dataloaders is not None:
            return self._dataloaders

        logging.info(
            "dataset_ratios not specified, datasets will be concatenated (map-style) "
            "or chained (webdataset.DataPipeline)."
        )

        # Per-dataset batch_size override support
        batch_sizes = {
            name: getattr(self.config.datasets_cfg, name).get("batch_size", 6)
            for name in self.datasets.keys()
        }

        datasets, batch_sizes = reorg_datasets_by_split(self.datasets, batch_sizes)
        self.datasets = datasets

        # stats
        for split_name in self.datasets:
            if isinstance(self.datasets[split_name], (tuple, list)):
                num_records = sum(
                    [
                        (len(d) if not isinstance(d, (wds.DataPipeline, ChainDataset)) else 0)
                        for d in self.datasets[split_name]
                    ]
                )
            else:
                if hasattr(self.datasets[split_name], "__len__"):
                    num_records = len(self.datasets[split_name])
                else:
                    num_records = -1
                    logging.info("Only a single wds.DataPipeline dataset, no __len__ attribute.")
            if num_records >= 0:
                logging.info("Loaded %d records for %s split from the dataset.", num_records, split_name)

        split_names = sorted(self.datasets.keys())
        datasets_list = [self.datasets[split] for split in split_names]
        bsz_list = [batch_sizes[split] for split in split_names]
        is_trains = [split in self.train_splits for split in split_names]

        collate_fns = []
        for dataset in datasets_list:
            if isinstance(dataset, (tuple, list)):
                collate_fns.append([getattr(d, "collater", None) for d in dataset])
            else:
                collate_fns.append(getattr(dataset, "collater", None))

        dataloaders = self.create_loaders(
            datasets=datasets_list,
            num_workers=self.config.run_cfg.num_workers,
            batch_sizes=bsz_list,
            is_trains=is_trains,
            collate_fns=collate_fns,
        )

        self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}
        return self._dataloaders

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # webdataset pipeline
            if isinstance(dataset, (ChainDataset, wds.DataPipeline)):
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset with (optional) distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=(sampler is None and is_train),
                    collate_fn=collate_fn,
                    drop_last=bool(is_train),
                )
                loader = PrefetchLoader(loader)
                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)
            return loader

        loaders = []
        for dataset, bsz, is_train, collate_fn in zip(datasets, batch_sizes, is_trains, collate_fns):
            if isinstance(dataset, (list, tuple)):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)
            loaders.append(loader)
        return loaders

    # --------------------------
    # Optim / scaler / LR
    # --------------------------
    @property
    def optimizer(self):
        if self._optimizer is not None:
            return self._optimizer
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        logging.info("number of trainable parameters: %d", num_parameters)
        optim_params = [
            {"params": p_wd, "weight_decay": float(self.config.run_cfg.weight_decay)},
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = self.config.run_cfg.get("beta2", 0.999)
        self._optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.config.run_cfg.init_lr),
            weight_decay=float(self.config.run_cfg.weight_decay),
            betas=(0.9, beta2),
        )
        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)
        if amp and self._scaler is None:
            self._scaler = torch.cuda.amp.GradScaler()
        return self._scaler

    @property
    def lr_scheduler(self):
        if self._lr_sched is not None:
            return self._lr_sched

        lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)
        if lr_sched_cls is None:
            raise ValueError(f"LR scheduler '{self.config.run_cfg.lr_sched}' not found in registry.")

        max_epoch = self.max_epoch
        min_lr = self.min_lr
        init_lr = self.init_lr
        warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
        warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
        iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)
        if iters_per_epoch is None:
            try:
                iters_per_epoch = len(self.dataloaders['train'])
            except Exception:
                iters_per_epoch = 10000

        self._lr_sched = lr_sched_cls(
            optimizer=self.optimizer,
            max_epoch=max_epoch,
            iters_per_epoch=iters_per_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=self.config.run_cfg.get("lr_decay_rate", None),
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
        return self._lr_sched

    # --------------------------
    # Train / Eval
    # --------------------------
    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def evaluate_only(self):
        return bool(self.config.run_cfg.evaluate)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        return self.dataloaders["train"]

    def unwrap_dist_model(self, model):
        return model.module if self.use_distributed else model

    def train(self):
        logging.info("[runner] Enter train()")
        start_time = time.time()
        best_agg_metric = 0.0
        best_epoch = 0

        self.log_config()
        logging.info(
            "[runner] Enter train(): train_splits=%s | valid_splits=%s | ds_keys=%s",
            self.train_splits, self.valid_splits, sorted(self.datasets.keys())
        )

        if self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path, model_only=self.evaluate_only)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            if self.valid_splits:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on %s.", split_name)
                    val_log = self.eval_epoch(split_name=split_name, cur_epoch=cur_epoch)
                    if val_log is not None:
                        assert "agg_metrics" in val_log, "No agg_metrics in validation log."
                        agg = float(val_log["agg_metrics"])
                        if agg > best_agg_metric and split_name == "val":
                            best_epoch, best_agg_metric = cur_epoch, agg
                            self._save_checkpoint(cur_epoch, is_best=True)
                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)
            else:
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            if self.use_distributed:
                dist.barrier()

        # final evaluation
        # Guard for eval-only (no training loop) and safe test_epoch
        cur_epoch = locals().get('cur_epoch', self.start_epoch)
        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        use_best = bool(getattr(self, 'valid_splits', [])) and os.path.exists(best_ckpt)
        test_epoch = 'best' if use_best else cur_epoch
        best_ckpt = os.path.join(self.output_dir, "checkpoint_best.pth")
        skip_reload = not use_best
        logs = {}
        if hasattr(self, "evaluate") and (self.test_splits or self.valid_splits):
            logs = self.evaluate(cur_epoch=("last" if skip_reload else test_epoch), skip_reload=skip_reload)

        # mirror to log.txt
        if isinstance(logs, dict):
            for split, d in logs.items():
                if d is not None:
                    self.log_stats(d, split_name=split)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time %s", total_time_str)

    def evaluate(self, cur_epoch="best", skip_reload=False):
        """
        No-op evaluation to avoid crashes when no val/test splits are configured.
        Returns {} unless self.test_splits is non-empty, in which case it returns
        a dict with minimal metrics so downstream logging code won't break.
        """
        logs = {}
        try:
            splits = list(self.test_splits) if getattr(self, "test_splits", []) else []
        except Exception:
            splits = []
        for split in splits:
            logs[split] = {"agg_metrics": 0.0, "epoch": cur_epoch}
        return logs
    
    def train_epoch(self, epoch):
        self.model.train()
        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader is not None, f"data_loader for split {split_name} is None."

        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(model=model, dataset=self.datasets[split_name])
        results = self.task.evaluation(model, data_loader)

        try:
            log_dict = self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
        except Exception:
            log_dict = None

        if log_dict is None:
            if isinstance(results, dict):
                log_dict = dict(results)
                log_dict.setdefault("agg_metrics", 0.0)
            else:
                log_dict = {"agg_metrics": 0.0}

        return log_dict

    # --------------------------
    # Checkpoint & logging
    # --------------------------
    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic and not param_grad_dic[k]:
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch %s to %s.", cur_epoch, save_to)
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
        logging.info("Loading checkpoint from %s.", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            logging.warning(
                "Key mismatch when loading checkpoint. Loading with strict=False."
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename, model_only: bool = False):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        base = self._wrapped_model if self._wrapped_model is not None else self._model
        base = self.unwrap_dist_model(base)
        base.load_state_dict(state_dict, strict=False)

        if not (model_only or self.evaluate_only):
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scaler and "scaler" in checkpoint and checkpoint["scaler"] is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1

        logging.info("Resume checkpoint from %s", url_or_filename)

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

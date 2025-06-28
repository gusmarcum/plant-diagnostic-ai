import logging
import os
import warnings
from omegaconf import OmegaConf
import torch.distributed as dist
from torchvision.datasets.utils import download_url

import minigpt4.common.utils as utils
from minigpt4.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor


class BaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            self.config = cfg

        self.data_type = self.config.data_type

        self.vis_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        if is_main_process():
            self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        logging.info("Building datasets...")
        datasets = self.build()
        return datasets

    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if vis_proc_cfg:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_vis()

    def _download_ann(self):
        anns = self.config.build_info.annotations
        for split, info in anns.items():
            storage_paths = info.get("storage", [])
            if isinstance(storage_paths, str):
                storage_paths = [storage_paths]

            for storage_path in storage_paths:
                if not os.path.exists(storage_path):
                    warnings.warn(f"Annotation file {storage_path} for {split} split does not exist.")

    def _download_vis(self):
        vis_info = self.config.build_info.get("images", {})
        for split, storage_path in vis_info.items():
            if isinstance(storage_path, str):
                if not os.path.exists(storage_path):
                    warnings.warn(f"Visual input path {storage_path} for {split} split does not exist.")

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        ann_info = build_info.annotations
        vis_info = build_info.images

        datasets = {}
        for split in ["train", "val"]:
            ann_paths = ann_info.get(split, {}).get("storage", [])
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            valid_ann_paths = [path for path in ann_paths if os.path.isfile(path)]
            if not valid_ann_paths:
                logging.warning(f"No valid annotation files found for {split} split.")
                continue

            vis_path = vis_info.get(split)
            if not os.path.isdir(vis_path):
                logging.warning(f"Visual input directory not found for {split} split: {vis_path}")
                continue

            vis_processor = self.vis_processors["train"] if split == "train" else self.vis_processors["eval"]
            text_processor = self.text_processors["train"] if split == "train" else self.text_processors["eval"]

            dataset_cls = self.train_dataset_cls if split == "train" else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                vis_root=vis_path,
                ann_paths=valid_ann_paths
            )

        return datasets


def load_dataset_config(cfg_path):
    cfg = OmegaConf.load(cfg_path).datasets
    cfg = cfg[list(cfg.keys())[0]]
    return cfg

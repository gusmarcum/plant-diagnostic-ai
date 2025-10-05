import logging
import os
from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.llava_dataset import LlavaConversationDataset


@registry.register_builder("strawberry_diagnostic")
class StrawberryDiagnosticBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaConversationDataset
    eval_dataset_cls = LlavaConversationDataset

    DATASET_CONFIG_DICT = {
        "default": "minigpt4/configs/datasets/strawberry_diagnostic.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        for split in ["train", "val"]:
            ann_paths = build_info.annotations.get(split, {}).get("storage", [])
            if not ann_paths:
                ann_paths = []
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            try:
                valid_ann_paths = [path for path in ann_paths if path and os.path.isfile(path)]
            except TypeError:
                valid_ann_paths = []
            if not valid_ann_paths:
                logging.warning(f"No valid annotation files found for {split} split.")
                continue

            vis_path = build_info.images.get(split)
            if not vis_path or not os.path.isdir(vis_path):
                logging.warning(f"Visual input directory not found for {split} split: {vis_path}")
                continue

            vis_processor = self.vis_processors["train"] if split == "train" else self.vis_processors["eval"]
            text_processor = self.text_processors["train"] if split == "train" else self.text_processors["eval"]

            dataset_cls = self.train_dataset_cls if split == "train" else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                vis_root=vis_path,
                ann_path=valid_ann_paths[0]  # LlavaConversationDataset expects ann_path, not ann_paths
            )

        return datasets

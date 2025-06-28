import logging
from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset


@registry.register_builder("coco_vqa")
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
    }

    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        datasets = dict()

        # Handle training split
        if "train" in self.config.build_info.annotations:
            storage_info = self.config.build_info.annotations.train.storage
            if isinstance(storage_info, list):
                ann_paths = storage_info
            else:
                ann_paths = [storage_info]

            logging.info(f"Building training dataset with paths: {ann_paths}")

            datasets["train"] = self.train_dataset_cls(
                vis_processor=self.vis_processors["train"],
                text_processor=self.text_processors["train"],
                ann_paths=ann_paths,
                vis_root=self.config.build_info.images.train
            )

        return datasets
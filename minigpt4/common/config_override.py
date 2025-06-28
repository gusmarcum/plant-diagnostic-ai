from minigpt4.common.config import Config
import os


class ConfigOverride(Config):
    @staticmethod
    def build_dataset_config(config):
        datasets = config.get("datasets", None)
        if datasets is None:
            raise KeyError(
                "Expecting 'datasets' as the root key for dataset configuration."
            )

        from omegaconf import OmegaConf
        dataset_config = OmegaConf.create()

        for dataset_name in datasets:
            from minigpt4.common.registry import registry
            builder_cls = registry.get_builder_class(dataset_name)

            # Use the correct path for dataset config
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "minigpt4/configs/datasets/coco/defaults_vqa.yaml"
            )

            # Merge configurations
            dataset_config = OmegaConf.merge(
                dataset_config,
                OmegaConf.load(config_path),
                {"datasets": {dataset_name: config["datasets"][dataset_name]}},
            )

        return dataset_config
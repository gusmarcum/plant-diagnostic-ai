from minigpt4.configs.config import Config
from minigpt4.datasets import build_datasets

config_path = "train_configs/minigptv2_finetune.yaml"
cfg = Config(config_path)
datasets = build_datasets(cfg)
print("Successfully loaded datasets:", datasets.keys())


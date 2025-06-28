import os
import sys
import argparse
from pathlib import Path
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# Import all necessary modules to register components
from minigpt4.models import *  # This will register all models
from minigpt4.datasets import *  # This will register all datasets
from minigpt4.tasks import *  # This will register all tasks
from minigpt4.processors import *  # This will register all processors
from minigpt4.runners import *  # This will register all runners

# Register the library root path
repo_root = Path(__file__).parent.absolute()
registry.register_path("library_root", str(repo_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Config test")
    parser.add_argument("--cfg-path", default="train_configs/minigptv2_finetune.yaml",
                        help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def test_config():
    print(f"Library root: {registry.get_path('library_root')}")

    args = parse_args()
    print(f"Loading config from: {args.cfg_path}")

    # Print registered models
    print("\nRegistered models:")
    print(registry.list_models())

    try:
        cfg = Config(args)
        print("Config loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_config()
    if success:
        print("\nNow you can try running the training command:")
        print("torchrun --nproc-per-node=1 train.py --cfg-path train_configs/minigptv2_finetune.yaml")
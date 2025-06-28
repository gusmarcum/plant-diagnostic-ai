import os
import sys
from pathlib import Path
import argparse

# Add the repository root to Python path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

from minigpt4.common.registry import registry
from minigpt4.common.config import Config
from minigpt4.models import *
from minigpt4.datasets import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Config test")
    parser.add_argument("--cfg-path", default="train_configs/minigptv2_finetune.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()

def main():
    # Register paths using os.path.join for proper path handling
    registry.register_path("library_root", repo_root)
    registry.register_path("configs", os.path.join(repo_root, "minigpt4", "configs"))
    
    print("Registered paths:")
    print(f"library_root: {registry.get_path('library_root')}")
    print(f"configs: {registry.get_path('configs')}")
    
    # Print registered models
    print("\nRegistered models:")
    print(registry.list_models())
    
    args = parse_args()
    print(f"\nLoading config from: {args.cfg_path}")
    
    try:
        cfg = Config(args)
        print("\nConfig loaded successfully!")
        return True
    except Exception as e:
        print(f"\nError loading config: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()

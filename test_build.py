import os
import sys
from pathlib import Path

# Add the repository root to Python path
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root))

# Now we can import from minigpt4
from minigpt4.datasets import build_datasets

def main():
    print("Python path:", sys.path)
    print("Current directory:", os.getcwd())
    
    # Load and build datasets
    config_path = "train_configs/minigptv2_finetune.yaml"
    print(f"\nTrying to build datasets from config: {config_path}")
    
    try:
        datasets = build_datasets(config_path)
        print("\nSuccessfully built datasets:")
        for name, dataset in datasets.items():
            print(f"- {name}: {len(dataset)} samples")
    except Exception as e:
        print(f"\nError building datasets: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

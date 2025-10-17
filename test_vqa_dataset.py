import os
import sys
from pathlib import Path

# Add the repository root to Python path
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root))

from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset
from minigpt4.processors import build_processor

def test_vqa_dataset():
    try:
        # Dataset paths
        vis_root = "/data/AGAI/MiniGPT-4/dataset/coco/images/train2014"
        ann_paths = [
            "/data/AGAI/MiniGPT-4/dataset/coco/annotations/v2_OpenEnded_mscoco_train2014_questions.json",
            "/data/AGAI/MiniGPT-4/dataset/coco/annotations/v2_mscoco_train2014_annotations.json"
        ]
        
        # Create simple processors
        vis_processor_cfg = {"name": "blip2_image_train", "image_size": 448}
        text_processor_cfg = {"name": "blip_caption"}
        
        vis_processor = build_processor(vis_processor_cfg)
        text_processor = build_processor(text_processor_cfg)
        
        print("Creating dataset...")
        dataset = COCOVQADataset(
            vis_processor=vis_processor,
            text_processor=text_processor,
            vis_root=vis_root,
            ann_paths=ann_paths
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test first item
        print("\nTesting first item...")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vqa_dataset()

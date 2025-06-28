import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Add the repository root to Python path
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root))

from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset

def test_vqa_dataset():
    try:
        # Dataset paths
        vis_root = "/data/kiriti/MiniGPT-4/dataset/coco/images/train2014"
        ann_paths = [
            "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_OpenEnded_mscoco_train2014_questions.json",
            "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_mscoco_train2014_annotations.json"
        ]
        
        # Create processor directly
        vis_processor = Blip2ImageEvalProcessor(image_size=448)
        text_processor = None  # We'll handle text processing in the dataset
        
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
        if 'image' in sample:
            print("Image shape:", sample['image'].shape)
        if 'question' in sample:
            print("Question:", sample['question'])
        if 'answers' in sample:
            print("Answers:", sample['answers'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vqa_dataset()

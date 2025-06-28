import os
import json
from PIL import Image
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from minigpt4.common.registry import registry

def process_vqa_annotations(question_file, answer_file):
    """Process VQA annotations into the expected format"""
    with open(question_file, 'r') as f:
        questions = json.load(f)
    with open(answer_file, 'r') as f:
        answers = json.load(f)
    
    # Create a mapping of question_id to answers
    answer_map = {a['question_id']: a for a in answers['annotations']}
    
    # Combine questions and answers
    processed_annotations = []
    for q in questions['questions']:
        q_id = q['question_id']
        if q_id in answer_map:
            ann = {
                'question_id': q_id,
                'image_id': q['image_id'],
                'image': f"COCO_train2014_{q['image_id']:012d}.jpg",  # Create image filename
                'question': q['question'],
                'answers': [ans['answer'] for ans in answer_map[q_id]['answers']]
            }
            processed_annotations.append(ann)
    
    return processed_annotations

def test_vqa_dataset_directly():
    # Register paths
    repo_root = os.path.dirname(os.path.abspath(__file__))
    registry.register_path("library_root", repo_root)
    
    # Paths
    vis_root = "/data/kiriti/MiniGPT-4/dataset/coco/images/train2014"
    question_path = "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_OpenEnded_mscoco_train2014_questions.json"
    answer_path = "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_mscoco_train2014_annotations.json"
    
    # Create simple processors
    vis_processor = Blip2ImageEvalProcessor(image_size=448)
    text_processor = None
    
    print("Creating dataset...")
    print(f"Using images from: {vis_root}")
    print(f"Using questions from: {question_path}")
    print(f"Using answers from: {answer_path}")
    
    try:
        # Verify files exist
        for path in [vis_root, question_path, answer_path]:
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                return None
        
        print("\nProcessing annotations...")
        annotations = process_vqa_annotations(question_path, answer_path)
        print(f"Processed {len(annotations)} QA pairs")
        
        # Create temporary file with processed annotations
        temp_ann_path = "temp_vqa_annotations.json"
        with open(temp_ann_path, 'w') as f:
            json.dump({'annotations': annotations}, f)
        
        dataset = COCOVQADataset(
            vis_processor=vis_processor,
            text_processor=text_processor,
            vis_root=vis_root,
            ann_paths=[temp_ann_path]
        )
        
        print(f"\nDataset created successfully with {len(dataset)} samples")
        
        # Test first item
        print("\nTesting first item...")
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        
        # Clean up temporary file
        os.remove(temp_ann_path)
        
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    dataset = test_vqa_dataset_directly()

import os
import json
from PIL import Image

def test_vqa_dataset():
    # Define paths
    base_path = "/data/kiriti/MiniGPT-4/dataset/coco"
    image_path = os.path.join(base_path, "images/train2014")
    question_path = os.path.join(base_path, "annotations/v2_OpenEnded_mscoco_train2014_questions.json")
    answer_path = os.path.join(base_path, "annotations/v2_mscoco_train2014_annotations.json")
    
    # Test file existence
    print("Checking file paths...")
    for path in [image_path, question_path, answer_path]:
        exists = os.path.exists(path)
        print(f"{path}: {'✓ exists' if exists else '✗ not found'}")
    
    # Try loading question and answer files
    print("\nTrying to load annotation files...")
    try:
        with open(question_path, 'r') as f:
            questions = json.load(f)
            print(f"Successfully loaded questions file. Found {len(questions['questions'])} questions")
            
            # Test first question
            first_q = questions['questions'][0]
            print(f"\nExample question: {first_q}")
            
            # Try loading corresponding image
            img_id = first_q['image_id']
            img_path = os.path.join(image_path, f"COCO_train2014_{img_id:012d}.jpg")
            print(f"\nTrying to load image: {img_path}")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                print(f"Successfully loaded image with size {img.size}")
            else:
                print("Image file not found!")
            
    except Exception as e:
        print(f"Error loading questions file: {e}")
        
    try:
        with open(answer_path, 'r') as f:
            answers = json.load(f)
            print(f"\nSuccessfully loaded answers file. Found {len(answers['annotations'])} answers")
    except Exception as e:
        print(f"Error loading answers file: {e}")

if __name__ == "__main__":
    test_vqa_dataset()

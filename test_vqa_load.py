import os
import sys
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

# Add the repository root to Python path
repo_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(repo_root))

def test_vqa_files():
    # Dataset paths
    question_path = "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_OpenEnded_mscoco_train2014_questions.json"
    answer_path = "/data/kiriti/MiniGPT-4/dataset/coco/annotations/v2_mscoco_train2014_annotations.json"
    
    print("\nTesting question file...")
    with open(question_path, 'r') as f:
        questions = json.load(f)
        print(f"Questions file keys: {questions.keys()}")
        print(f"Number of questions: {len(questions['questions'])}")
        print("\nSample question:", questions['questions'][0])
    
    print("\nTesting answer file...")
    with open(answer_path, 'r') as f:
        answers = json.load(f)
        print(f"Answers file keys: {answers.keys()}")
        print(f"Number of answers: {len(answers['annotations'])}")
        print("\nSample answer:", answers['annotations'][0])

    return questions, answers

def create_processed_annotation(questions, answers):
    """Create a processed annotation file that matches the expected format"""
    processed_path = "/data/kiriti/MiniGPT-4/dataset/coco/preprocessed"
    os.makedirs(processed_path, exist_ok=True)
    
    # Create a mapping of question_id to answers
    answer_map = {a['question_id']: a for a in answers['annotations']}
    
    processed_data = {
        'annotations': []
    }
    
    for q in questions['questions']:
        q_id = q['question_id']
        if q_id in answer_map:
            entry = {
                'question_id': q_id,
                'image_id': q['image_id'],
                'question': q['question'],
                'answers': answer_map[q_id]['answers']
            }
            processed_data['annotations'].append(entry)
    
    output_path = os.path.join(processed_path, 'vqa_train.json')
    with open(output_path, 'w') as f:
        json.dump(processed_data, f)
    print(f"\nCreated processed annotation file at: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Testing VQA dataset files...")
    questions, answers = test_vqa_files()
    
    print("\nCreating processed annotation file...")
    processed_path = create_processed_annotation(questions, answers)
    
    print("\nNow you can update your yaml config to use this processed annotation file")
    print("Update the annotation path in your config to:", processed_path)

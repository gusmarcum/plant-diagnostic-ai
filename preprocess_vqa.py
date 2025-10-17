import json
import os

def preprocess_vqa(questions_file, annotations_file, output_file):
    # Load questions
    with open(questions_file, 'r') as f:
        questions = json.load(f)["questions"]

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)["annotations"]

    # Create a mapping of question_id to answers
    answers_dict = {ann["question_id"]: ann for ann in annotations}

    # Merge questions and answers
    merged_data = []
    for q in questions:
        question_id = q["question_id"]
        if question_id in answers_dict:
            merged_data.append({
                "question_id": question_id,
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": [ans["answer"] for ans in answers_dict[question_id]["answers"]],
            })

    # Save the merged data
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Saved preprocessed VQA data to {output_file}")

# Define paths
datasets_dir = "/data/AGAI/MiniGPT-4/dataset/coco/annotations"
output_dir = "/data/AGAI/MiniGPT-4/dataset/coco/preprocessed"

os.makedirs(output_dir, exist_ok=True)

# Preprocess train split
preprocess_vqa(
    os.path.join(datasets_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
    os.path.join(datasets_dir, "v2_mscoco_train2014_annotations.json"),
    os.path.join(output_dir, "vqa_train.json")
)

# Preprocess val split
preprocess_vqa(
    os.path.join(datasets_dir, "v2_OpenEnded_mscoco_val2014_questions.json"),
    os.path.join(datasets_dir, "v2_mscoco_val2014_annotations.json"),
    os.path.join(output_dir, "vqa_val.json")
)

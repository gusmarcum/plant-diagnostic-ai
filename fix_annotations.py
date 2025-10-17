import json

input_file = "/data/AGAI/MiniGPT-4/dataset/coco/preprocessed/vqa_val.json"
output_file = "/data/AGAI/MiniGPT-4/dataset/coco/preprocessed/vqa_val_fixed.json"
image_prefix = "COCO_val2014"

def add_image_field(input_path, output_path, image_prefix):
    with open(input_path, "r") as f:
        annotations = json.load(f)

    for ann in annotations:
        if "image_id" in ann:
            image_id = ann["image_id"]
            ann["image"] = f"{image_prefix}_{image_id:012d}.jpg"
        else:
            raise KeyError(f"Missing 'image_id' in annotation: {ann}")

    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)

# Add "image" field to annotations
add_image_field(input_file, output_file, image_prefix)

print(f"Updated annotations written to {output_file}")

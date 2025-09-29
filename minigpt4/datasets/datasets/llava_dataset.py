import os
import json
from PIL import Image
from torch.utils.data import Dataset

# ---- Helpers ----------------------------------------------------------------

_IMG_TAGS = ("<image>", "<Img><ImageHere></Img>")

def _strip_image_tokens(text: str) -> str:
    """
    Remove any image placeholder tokens that may be embedded in the text.
    """
    if not isinstance(text, str):
        return text
    for t in _IMG_TAGS:
        text = text.replace(t, "")
    return text.replace("\n", " ").strip()

def _pick_image_file(info):
    """
    Resolve the image filename from a dataset record.
    Assumes 'image' is a string (relative path under vis_root).
    """
    if "image" not in info:
        raise KeyError("Dataset record missing 'image' field")
    return info["image"]

def _safe_image_id(info, image_relpath: str) -> str:
    """
    Prefer explicit 'id' if present; otherwise derive from filename.
    """
    if isinstance(info, dict) and "id" in info:
        return str(info["id"])
    base = os.path.basename(image_relpath)
    return os.path.splitext(base)[0]

# ---- Datasets ----------------------------------------------------------------

class LlavaDetailDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. /data/.../plant_diagnostic)
        ann_path (string): Path to annotation JSON
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, "r") as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_rel = _pick_image_file(info)
        image_path = os.path.join(self.vis_root, image_rel)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction_raw = info["conversations"][0]["value"]
        instruction_txt = _strip_image_tokens(instruction_raw)
        instruction_txt = self.text_processor(instruction_txt)
        instruction = f"<Img><ImageHere></Img> {instruction_txt} "

        answer = info["conversations"][1]["value"]

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": _safe_image_id(info, image_rel),
        }


class LlavaReasonDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. /data/.../plant_diagnostic)
        ann_path (string): Path to annotation JSON
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, "r") as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_rel = _pick_image_file(info)
        image_path = os.path.join(self.vis_root, image_rel)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction_raw = info["conversations"][0]["value"]
        instruction_txt = _strip_image_tokens(instruction_raw)
        instruction_txt = self.text_processor(instruction_txt)
        instruction = f"<Img><ImageHere></Img> {instruction_txt} "

        answer = info["conversations"][1]["value"]

        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": _safe_image_id(info, image_rel),
        }


class LlavaConversationDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. /data/.../plant_diagnostic)
        ann_path (string): Path to annotation JSON
        """
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.connect_sym = "!@#"

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_rel = _pick_image_file(info)
        image_path = os.path.join(self.vis_root, image_rel)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        first_instruction_raw = info["conversations"][0]["value"]
        first_instruction_txt = _strip_image_tokens(first_instruction_raw)
        first_instruction_txt = self.text_processor(first_instruction_txt)
        first_instruction = f"<Img><ImageHere></Img> {first_instruction_txt} "

        questions = [first_instruction]
        answers = []

        for i, item in enumerate(info["conversations"][1:]):
            if i % 2 == 0:
                # Assistant turn
                answers.append(item["value"])
            else:
                # Human turn (strip image tokens; do not prepend image token again)
                human_txt = _strip_image_tokens(item["value"])
                human_txt = self.text_processor(human_txt)
                questions.append(human_txt + " ")

        questions_joined = self.connect_sym.join(questions)
        answers_joined = self.connect_sym.join(answers)

        return {
            "image": image,
            "conv_q": questions_joined,
            "conv_a": answers_joined,
            "image_id": _safe_image_id(info, image_rel),
            "connect_sym": self.connect_sym,
        }

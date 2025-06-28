"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
from PIL import Image
from collections import OrderedDict
from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict({
            "file": ann["image"],
            "question": ann["question"],
            "question_id": ann["question_id"],
            "answers": "; ".join(ann["answer"]),
            "image": sample["image"],
        })


class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print(f"Initializing COCOVQADataset with ann_paths: {ann_paths}")

        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        self.annotation = [
            ann for ann in self.annotation
            if os.path.exists(os.path.join(self.vis_root, ann["image"].split('/')[-1]))
        ]

    def get_data(self, index):
        ann = self.annotation[index]
        image_filename = f"COCO_train2014_{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        answer = random.choice(ann["answers"])
        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }


class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = ["Question: {} Short answer:"]

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_filename = f"COCO_val2014_{ann['image_id']:012d}.jpg"
        image_path = os.path.join(self.vis_root, image_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            'image_path': image_path,
            "question": question,
            "question_id": ann["question_id"],
            "instruction_input": instruction,
            "instance_id": ann["instance_id"],
        }


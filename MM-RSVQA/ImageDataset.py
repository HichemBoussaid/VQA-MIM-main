# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import gc
import torch
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DistilBertTokenizer

print("Loading DistilBert...")
model_path = "distilbert/distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
print("Done.", flush=True)

MAX_ANSWERS  = 4332
LEN_QUESTION = 20


class ImageDataset(Dataset):
    def __init__(self,
                 split,                        # "train", "val", or "test"
                 pathS1S2,                     # local path to S1/S2 patches, structure: {dept}/{image_id}/S1.tif
                 train=True,
                 ratio_images_to_use=1,
                 transform=None,
                 transformS2=None,
                 number_outputs=1000,
                 selected_answers=None,
                 tokenizer=tokenizer,
                 activate_bdo=True,
                 activate_s1=True,
                 activate_s2=True,
                 hf_repo="HichemBoussaid/TAMMI"):

        self.pathS1S2       = pathS1S2
        self.train          = train
        self.transform      = transform
        self.transformS2    = transformS2
        self.activate_bdo   = activate_bdo
        self.activate_s1    = activate_s1
        self.activate_s2    = activate_s2
        self.tokenizer      = tokenizer

        if not (activate_bdo or activate_s1 or activate_s2):
            raise ValueError("You need at least one image modality")

        # Load dataset from HF
        print(f"Loading TAMMI {split} split from HF...")
        ds = load_dataset(hf_repo, split=split)

        # Apply ratio
        n = int(len(ds) * ratio_images_to_use)
        ds = ds.select(range(n))
        print(f"  {len(ds)} images loaded")

        # Build flat list of (image_idx, question, answer, type)
        print("Building Q&A pairs...")
        self.images = ds
        self.images_questions_answers = []

        if train:
            # Build frequency dict across all Q&A pairs
            freq_dict = {}
            for row in tqdm(ds):
                for ans in row["answers"]:
                    freq_dict[ans] = freq_dict.get(ans, 0) + 1

            freq_sorted = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
            self.selected_answers = [k for k, _ in freq_sorted[:number_outputs]]
            coverage = sum(v for _, v in freq_sorted[:number_outputs])
            total    = sum(freq_dict.values())
            print(f"Top {number_outputs} answers cover {coverage/total*100:.2f}% of total answers.")
        else:
            self.selected_answers = selected_answers

        for img_idx, row in enumerate(tqdm(ds)):
            for q, a, t in zip(row["questions"], row["answers"], row["types"]):
                if self.selected_answers is None or a in self.selected_answers:
                    tokens = tokenizer(
                        q,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=26,
                        add_special_tokens=True
                    )
                    tokens["input_ids"]      = tokens["input_ids"].squeeze()
                    tokens["attention_mask"] = tokens["attention_mask"].squeeze()

                    answer_encoded = self.selected_answers.index(a) if self.selected_answers else a
                    self.images_questions_answers.append((tokens, answer_encoded, img_idx, t))

        print(f"  {len(self.images_questions_answers)} Q&A pairs built")
        gc.collect()

    def __len__(self):
        return len(self.images_questions_answers)

    def __getitem__(self, index):
        tokens, answer, img_idx, type_str = self.images_questions_answers[index]
        row  = self.images[img_idx]
        dept = row["department"]
        iid  = row["image_id"]
        patch_dir = os.path.join(self.pathS1S2, dept, iid)

        if self.activate_bdo:
            imgBDO = np.array(row["bdo"])  # PIL image already embedded in HF dataset

        if self.activate_s2:
            imgS2 = tiff.imread(os.path.join(patch_dir, "S2.tif"))

        if self.activate_s1:
            imgS1 = tiff.imread(os.path.join(patch_dir, "S1.tif"))

        if self.transform:
            if self.activate_bdo:
                imgBDO = self.transform(imgBDO.copy())
            if self.activate_s1:
                imgS1  = self.transform(imgS1.copy())
        if self.transformS2:
            if self.activate_s2:
                imgS2 = self.transformS2(imgS2.copy())

        data = (
            tokens,
            answer,
            *([imgBDO] if self.activate_bdo else []),
            *([imgS2]  if self.activate_s2  else []),
            *([imgS1]  if self.activate_s1  else []),
            type_str,
        )

        return data, index


del tokenizer

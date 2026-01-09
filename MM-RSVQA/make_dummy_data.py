import json
import os
from pathlib import Path

import numpy as np
import tifffile as tiff

# Minimal dummy data generator for smoke tests
# Usage:
#   python MM-RSVQA/make_dummy_data.py
# Then set BDO_ROOT to the printed path (or edit MM-RSVQA.py),
# and optionally set DUMMY_RUN=1 before running MM-RSVQA/MM-RSVQA.py.


def main():
    root = Path("dummy_data/BDO")
    img_id = "01-0001"
    dept, img_num = img_id.split("-")
    img_dir = root / dept / dept / img_num
    img_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple 1000x1000 RGB TIFF with random bytes
    img = (np.random.rand(1000, 1000, 3) * 255).astype(np.uint8)
    tiff.imwrite(img_dir / "bdo.tif", img)

    # One question/answer entry reused for train/val/test
    images_entry = {
        "images": [
            {
                "id": img_id,
                "active": True,
                "BdOrthoname": "bdo.tif",
                "S2name": "",
                "S2centerpos": [0, 0],
                "S1namevh": "",
                "S1namevv": "",
                "S1centerpos": [0, 0],
                "orbit_directionswath": "A",
                "look_directionswath": "R",
                "questions_ids": [0],
            }
        ]
    }
    questions_entry = {"questions": [{"question": "Is this dummy?", "type": "bool"}]}
    answers_entry = {"answers": [{"answer": "yes"}]}

    splits = [
        ("split_train_images.json", images_entry),
        ("split_val_images.json", images_entry),
        ("split_test_images.json", images_entry),
        ("split_train_questions.json", questions_entry),
        ("split_val_questions.json", questions_entry),
        ("split_test_questions.json", questions_entry),
        ("split_train_answers.json", answers_entry),
        ("split_val_answers.json", answers_entry),
        ("split_test_answers.json", answers_entry),
    ]

    for filename, payload in splits:
        with open(Path(__file__).parent / filename, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    # selected_answers.pkl is expected by the main script; create a minimal list
    import pickle

    with open(Path(__file__).parent.parent / "selected_answers.pkl", "wb") as f:
        pickle.dump(["yes", "no"], f)

    print("Dummy data written.")
    print(f"Set BDO_ROOT={root.resolve()} and DUMMY_RUN=1 before running MM-RSVQA/MM-RSVQA.py")


if __name__ == "__main__":
    main()

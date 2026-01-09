# VQA-MIM

Minimal notes to get the code running and to understand required assets.

## Requirements
- Python 3.9+ with PyTorch (CUDA optional).
- torchvision, transformers, rasterio, tifffile, matplotlib, tqdm, numpy, scipy.
- GPU recommended for training; CPU works if CUDA is unavailable.

## External weights and data (not included)
- VisualBERT VQA checkpoint at `uclanlp/visualbert-vqa`.
- DistilBERT tokenizer at `distilbert/distilbert-base-uncased`.
- ResNet152 weights at `checkpoints/resnet152-f82ba261.pth`.
- SAR and optical ResNet50 weights at `weights/sar.tar` and `weights/opt.tar`.
- Answer vocabulary pickle at `selected_answers.pkl`.
- JSON splits: `split_train_images.json`, `split_train_questions.json`, `split_train_answers.json`, and the matching `validation`/`test` files under `MM-RSVQA/`.
- Imagery roots: set `pathBdO`, `pathS1`, `pathS2` in `MM-RSVQA/MM-RSVQA.py` to your local data locations.

## Quick start
1. Place the required weight files and transformer checkpoints at the paths above.
2. Fill in `pathBdO`, `pathS1`, and `pathS2` in `MM-RSVQA/MM-RSVQA.py`.
3. (Optional) adjust batch size, epochs, and `ratio_images_to_use` in `hyper_params`.
4. Run training and evaluation:
	```bash
	python MM-RSVQA/MM-RSVQA.py
	```

## Dummy smoke test (no real data)
1. Generate tiny dummy data and splits:
	```bash
	python MM-RSVQA/make_dummy_data.py
	```
2. Set environment for the dummy run:
	- `set BDO_ROOT=ABSOLUTE_PATH_PRINTED_BY_SCRIPT`
	- `set DUMMY_RUN=1`
3. Run:
	```bash
	python MM-RSVQA/MM-RSVQA.py
	```
This uses random weights for ResNet152 if the checkpoint is missing and disables Sentinel-1/2.

## Notes
- The model supports optional modalities (BDOrtho, Sentinel-1, Sentinel-2). Enable/disable them via the dataset and model flags in `MM-RSVQA/MM-RSVQA.py`.
- Checkpoints are saved under `MM-RSVQA/` after each epoch.
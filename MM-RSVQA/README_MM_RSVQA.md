# MM-RSVQA

Implementation of **MM-RSVQA** (Multi-Modal Multi-Resolution Remote Sensing Visual Question Answering), introduced at CVPR 2025 EarthVision Workshop.

> **Visual Question Answering on Multiple Remote Sensing Image Modalities**
> Hichem Boussaid\*, Lucrezia Tosato\*, Flora Weissgerber, Camille Kurtz, Laurent Wendling, Sylvain Lobry
> CVPR 2025 EarthVision Workshop
> [Paper](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/papers/Boussaid_Visual_Question_Answering_on_Multiple_Remote_Sensing_Image_Modalities_CVPRW_2025_paper.pdf) | [Dataset](https://huggingface.co/datasets/HichemBoussaid/TAMMI) | [Project Page](https://tammi.sylvainlobry.com/)

---

## Overview

MM-RSVQA is a VisualBERT-based model that fuses three remote sensing image modalities with natural language questions:

- **BDOrtho** — Very High Resolution (VHR) RGB orthophotos (IGN, 20cm)
- **Sentinel-2** — Multispectral imagery (10 channels: TCI + B05/06/07/08/8A/11/12)
- **Sentinel-1** — SAR imagery (3 channels: VV, VH, VV/VH ratio)

Each modality is encoded by a ResNet backbone, then fused with the question embedding through VisualBERT for joint visual-language reasoning.

---

## Requirements

```bash
pip install torch torchvision transformers datasets tifffile rasterio scipy tqdm huggingface_hub
```

Python 3.9+. GPU recommended for training.

---

## External Weights

The following pretrained weights are required and not included in this repository:

| File | Description | Source |
|------|-------------|--------|
| `checkpoints/resnet152-f82ba261.pth` | ResNet152 for BDOrtho encoding | [PyTorch model zoo](https://download.pytorch.org/models/resnet152-f82ba261.pth) |
| `weights/sar.tar` | ResNet50 pretrained on SAR for S1 encoding | Provided separately |
| `weights/opt.tar` | ResNet50 pretrained on optical for S2 encoding | Provided separately |
| `uclanlp/visualbert-vqa` | VisualBERT VQA checkpoint | Downloaded automatically via HuggingFace |
| `distilbert/distilbert-base-uncased` | DistilBERT tokenizer | Downloaded automatically via HuggingFace |

---

## Data Setup

### 1. TAMMI Dataset (BDO + Q&A)

The dataset is hosted on HuggingFace and loaded automatically — no manual download needed:

```python
from datasets import load_dataset
ds = load_dataset("HichemBoussaid/TAMMI", split="train")
```

### 2. S1/S2 Patches

S1/S2 patches must be pre-extracted from the mosaics using the provided script:

```bash
# Download mosaics
python -c "
from huggingface_hub import snapshot_download
snapshot_download('HichemBoussaid/TAMMI_S1S2', repo_type='dataset', local_dir='./S1S2_mosaics')
"

# Extract patches
python get_s1s2_patches.py
```

This produces patches at `./patches/{dept}/{image_id}/S1.tif` and `S2.tif`.

See [TAMMI_S1S2](https://huggingface.co/datasets/HichemBoussaid/TAMMI_S1S2) for full documentation.

---

## Training

Edit `MM-RSVQA/MM-RSVQA.py` to set:

```python
pathS1S2 = "path/to/patches"   # output of get_s1s2_patches.py
```

Or set the environment variable:

```bash
export S1S2_ROOT=path/to/patches
```

Then run:

```bash
python MM-RSVQA/MM-RSVQA.py
```

By default trains with BDO only (`activate_s1=False, activate_s2=False`). To enable all modalities:

```python
model = VQAModel(activate_bdo=True, activate_s1=True, activate_s2=True)
```

And update `trainFunction()` / `testFunction()` accordingly.

Checkpoints are saved after each epoch at `MM-RSVQA/BigModelV2_{epoch}.tar`.

---

## Configuration

Key hyperparameters in `MM-RSVQA.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 80 | Batch size |
| `num_epochs` | 3 | Number of training epochs |
| `learning_rate` | 3e-5 | AdamW learning rate |
| `number_outputs` | 1000 | Number of answer classes |
| `ratio_images_to_use` | 1 | Fraction of dataset to use (0-1) |
| `nm_workers` | 10 | DataLoader workers |

---

## Model Architecture

```
Question → DistilBERT tokenizer
BDOrtho  → ResNet152 → L2-normalized features (2048-d)
S1       → ResNet50(SAR)  → L2-normalized features
S2       → ResNet50(Opt)  → L2-normalized features (10-channel input)
                         ↓
          Concatenated visual embeddings
                         ↓
                    VisualBERT
                         ↓
                 1000-class classifier
```

---

## Results

Results on the TAMMI test set:

| Model | Overall Accuracy | Average Accuracy |
|-------|-----------------|-----------------|
| BDO only | 52.22% | 61.94% |
| S2 RGB + BDO | 54.41% | 64.72% |
| S2 (all bands) + BDO | 54.49% | 64.87% |
| S1 + BDO | 54.29% | 64.40% |
| S2 + S1 | 54.70% | 65.15% |
| **S2 + S1 + BDO (MM-RSVQA)** | **55.11%** | **65.56%** |

---

## File Structure

```
MM-RSVQA/
├── MM-RSVQA.py        # Main training and evaluation script
├── ImageDataset.py    # PyTorch Dataset loading from HuggingFace TAMMI
├── VQAModel.py        # MM-RSVQA model definition
├── sar_display.py     # SAR visualization and geometric transform utilities
├── make_dummy_data.py # Smoke test data generator
└── get_s1s2_patches.py # Extract S1/S2 patches from mosaics
```

---

## Smoke Test

To verify the setup without real data:

```bash
python MM-RSVQA/make_dummy_data.py
export BDO_ROOT=<path printed by script>
export DUMMY_RUN=1
python MM-RSVQA/MM-RSVQA.py
```

---

## Citation

```bibtex
@inproceedings{boussaid2025tammi,
  title     = {Visual Question Answering on Multiple Remote Sensing Image Modalities},
  author    = {Boussaid, Hichem and Tosato, Lucrezia and Weissgerber, Flora and Kurtz, Camille and Wendling, Laurent and Lobry, Sylvain},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year      = {2025}
}
```

*This work is supported by ANR under the ANR-21-CE23-0011 project. Experiments were performed using HPC/AI resources provided by GENCI-IDRIS (Grant 2023-AD011012735R2).*

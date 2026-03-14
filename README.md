# VetFractureAI

### Femoral Shaft Fracture Detection & AO-Based Severity Classification
<p align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvendherfrancisco/VetFractureAI/blob/main/fracture_detection_and_ao_classification.ipynb)
[![HuggingFace Demo](https://img.shields.io/badge/HuggingFace-Demo-yellow)](https://huggingface.co/spaces/alvendherfrancisco/Fracture-Detection-and-AO-Based-Severity-Classification-in-Dogs-and-Cats)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/alvendherfrancisco/VetFractureAI-Dataset)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/alvendherfrancisco/VetFractureAI-Model)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/)

</p>

---

## Overview

**VetFractureAI** is a deep learning system based on **Faster R-CNN** that detects femoral shaft fractures in veterinary X-ray images and classifies their severity using the **AO/OTA fracture classification system**.

The model performs:

- **Fracture localization** using bounding boxes
- **AO fracture severity classification**
- **Single-pass inference using Faster R-CNN**

> **Medical Disclaimer:** This project is intended for **research and educational purposes only** and is not a certified medical device. Always consult a licensed veterinarian for clinical diagnosis and treatment decisions.

---

## AO Fracture Classification

| Class | Label | Description | Surgical Implication |
|-------|-------|-------------|----------------------|
| 0 | No Fracture | Intact femoral shaft | No surgical intervention |
| 1 | 32-A | Simple fracture (two fragments) | Intramedullary pin or plate fixation |
| 2 | 32-B | Wedge fracture | More complex fixation required |
| 3 | 32-C | Complex / comminuted | Advanced fixation techniques |

---

## Dataset

- **Total images:** 1,264 veterinary radiographs
- **Dogs:** 690 images · **Cats:** 574 images
- **Views:** Anteroposterior (AP) and lateral
- **Annotation tool:** CVAT · **Format:** COCO JSON
- **Split:** 70% Train / 15% Validation / 15% Test

| Species | No Fracture | 32-A | 32-B | 32-C | Total |
|---------|-------------|------|------|------|-------|
| Dogs | 152 | 200 | 150 | 188 | 690 |
| Cats | 150 | 165 | 133 | 126 | 574 |
| **Total** | **302** | **365** | **283** | **314** | **1,264** |

### Dataset Access

Publicly available on Hugging Face — no request needed.

[huggingface.co/datasets/alvendherfrancisco/VetFractureAI-Dataset](https://huggingface.co/datasets/alvendherfrancisco/VetFractureAI-Dataset)

**Terms:** Academic and non-commercial use only. No redistribution. Citation required in any publication.

---

## Model Weights Access

Publicly available on Hugging Face — no request needed.

[huggingface.co/alvendherfrancisco/VetFractureAI-Model](https://huggingface.co/alvendherfrancisco/VetFractureAI-Model)

Download `best_model.pth`, place it in the `models/` directory, and update the path in the `Config` class:

```python
Config.MODEL_SAVE_PATH = "/content/drive/MyDrive/.../models/best_model.pth"
```

**Terms:** Academic and non-commercial use only. No redistribution. Citation required in any publication.

---

## Model Architecture

| Component | Details |
|-----------|---------|
| Architecture | Faster R-CNN |
| Backbone | ResNet-50 + FPN V2 |
| Pretrained weights | COCO |
| Input resolution | 1024 x 1024 px |
| Output classes | 4 (Background, 32-A, 32-B, 32-C) |
| Optimizer | SGD |
| Learning rate | 0.005 |
| Momentum / Weight decay | 0.9 / 0.0005 |
| Scheduler | CosineAnnealingLR (T_max=35) |
| Batch size / Epochs | 16 / 50 |
| Best checkpoint | Epoch 43 |
| Confidence thresholds | 32-A: 0.65, 32-B: 0.45, 32-C: 0.65 |

---

## Preprocessing Pipeline

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | Color conversion | BGR to RGB |
| 2 | Resize | LongestMaxSize — max 1024 px |
| 3 | Pad | Zero-padding to 1024 x 1024 px |
| 4 | CLAHE | clip_limit=3.0, tile_grid=8x8, p=1.0 |
| 5 | Normalize | mean=0.5, std=0.5 |

**Training augmentations:**

| Transform | Parameters |
|-----------|------------|
| HorizontalFlip | p=0.5 |
| ShiftScaleRotate | shift +-6.25%, scale +-10%, rotate +-20°, p=0.5 |
| RandomBrightnessContrast | +-0.3, p=0.3 |

---

## Model Performance

### Test Set Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | **88.36%** |
| Average IoU | **77.69%** |
| Accuracy | **77.89%** |
| Macro Precision | **77.64%** |
| Macro Sensitivity | **77.64%** |
| Macro Specificity | **97.83%** |
| Macro F1-Score | **77.52%** |

### Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| No Fracture | 0.918 | 0.978 | 0.947 |
| 32-A Simple | 0.804 | 0.774 | 0.789 |
| 32-B Wedge | 0.565 | 0.619 | 0.591 |
| 32-C Complex | 0.818 | 0.735 | 0.774 |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/alvendherfrancisco/VetFractureAI.git
cd VetFractureAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run in Google Colab

Click the **Open in Colab** badge at the top of this page and run all cells in order.

> Update `Config.ROOT_DIR` in the `Config` class to match your Google Drive folder structure.

### 4. Notebook Sections

| Section | Description |
|---------|-------------|
| 1 — Environment Setup | Install all required packages |
| 2 — Mount Google Drive | Connect to Drive for dataset and checkpoint access |
| 3 — Dataset Distribution | Inspect image counts, species, and AO class breakdown |
| 4 — Complete Training Pipeline | Configure, train, and save the best Faster R-CNN model |
| 5 — TensorBoard Monitoring | Visualize training metrics in real-time |
| 6 — Test Set Evaluation | Evaluate the best checkpoint on the held-out test set |
| 7 — Interactive Gradio Interface | Launch a web UI for real-time inference (optional) |

---

## Repository Structure

```
VetFractureAI/
│
├── images/ ← Veterinary radiograph dataset 
│   ├── train/                 
│   ├── val/                  
│   └── test/                  
│
├── annotations/ ← COCO JSON annotation files
│   ├── instances_train.json   
│   ├── instances_val.json
│   └── instances_test.json
│
├── models/ ← Saved model checkpoints                   
│   └── best_model.pth         
│
├── logs/ ← Training logs and evaluation outputs                      
│
├── source_code/ ← Main notebook and training scripts
│   └── fracture_detection_and_ao_classification.ipynb
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Contact

Email: [alvendherfrancisco01@gmail.com](mailto:alvendherfrancisco01@gmail.com)
GitHub: [github.com/alvendherfrancisco](https://github.com/alvendherfrancisco)

---

## License

Licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

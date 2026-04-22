# 🧠 Thyroid Cancer Classification Pipeline

Unified deep learning pipeline for classification of thyroid lesions using cytological and ultrasound images. The project is designed with a **single-entry architecture**, reproducibility, and **paper-level evaluation metrics**.

---

## 📌 Overview

This project implements an end-to-end pipeline:

build_metadata() → train_fold() → evaluate() → visualize_attention()

All logic is centralized in **`main.py`** — no duplication or inconsistencies between transforms, dataloaders, or model definitions.

---

## 📂 Dataset Structure

The dataset consists of four modalities:

- **dataset1** — Cytological images of benign thyroid lesions  
- **dataset2** — Ultrasound images of benign thyroid lesions  
- **dataset3** — Ultrasound images of papillary thyroid carcinoma  
- **dataset4** — Cytological images of papillary thyroid carcinoma  

### 🧾 Metadata

- Automatically generated as `metadata.csv` on first run  
- Contains:
  - `patient_id`
  - `modality` (US / Cytology)
  - `label` (benign / malignant)
  - image paths

---

## 🏗️ Project Structure

├── main.py - Training + evaluation pipeline <br>
├── model_inference.py - Inference script <br>
├── metadata.csv - Auto-generated dataset index <br>
├── checkpoints/ - Model weights <br>
├── graphs/ - Metrics & visualizations <br>
└── datasets/ - Raw data <br>

---

## ⚙️ Training Pipeline

All components are defined **once and reused**:

- transforms
- dataloaders
- model
- collate functions

---

## 🚨 Overfitting Fixes

**Problem:**  
~99% accuracy in early epochs due to overfitting (ConvNeXt-Tiny fully unfrozen on a small dataset ~385 patients).

### ✅ Solutions

**Backbone Freezing**

- Only last stage (stage 3) is trainable
- Remaining layers are frozen

**Regularization**

- Dropout: `0.1 → 0.4`
- Weight decay: `1e-4 → 1e-3`
- Label smoothing: `0.05`

**Model Simplification**

- Transformer depth: `2 → 1`
- Embedding dimension: `768 → 512`

**Optimization**

- Learning rate: `1e-4 → 3e-5`
- Warmup: `5 epochs`
- Gradient clipping: `norm=1.0`

**Training Strategy**

- Early stopping based on **validation AUROC**

---

## 📊 Evaluation Metrics (Paper-Level)

All outputs are saved in `graphs/`.

### Core Metrics

- AUROC + Bootstrap 95% CI (1000 iterations)
- AUPRC
- Accuracy
- Balanced Accuracy
- F1 Score
- MCC (Matthews Correlation Coefficient)
- Cohen’s Kappa

### Clinical Metrics

- Sensitivity (Recall)
- Specificity
- PPV (Precision)
- NPV
- Brier Score (calibration)

### Threshold Optimization

- Metrics computed at **Youden Index threshold**

---

## 📈 Visualizations

Automatically generated:

- ROC Curve  
- Precision-Recall Curve  
- Calibration Curve  
- Score Distribution per class  
- Sensitivity / Specificity / F1 vs Threshold  
- Patient-level prediction bar plot  
- Attention maps (TP / TN / FP / FN cases)

---

## 🧠 Attention Visualization

- Patch-level attention maps  
- Cosine similarity-based relevance scoring  
- Color-coded patches per patient  

---

## 🔍 Inference (`model_inference.py`)

### 1. Single Patient

```bash
python model_inference.py \
  --us_path path/to/image \
  --patch_dir path/to/patches \
  --patient_id ID
```

**Outputs:**

- Prediction
- Attention visualization

### 2. Batch Mode

```bash
python model_inference.py --csv metadata.csv
```

**Outputs:**

- Computes full metrics if labels are available

### 3. Demo Mode

```bash
python model_inference.py
```

## 💾 Outputs

### checkpoints/

- Trained model weights

### graphs/

- Metrics
- Plots
- Evaluation artifacts

### 🚀 Getting Started

```bash
python main.py
python model_inference.py
```

### Key Features

- Unified pipeline (single source of truth)
- Strong overfitting control
- Clinically relevant metrics
- Attention-based interpretability
- Reproducible design

### Notes

- Designed for small medical datasets
- Focus on generalization, not raw accuracy
- Metrics aligned with clinical decision-making

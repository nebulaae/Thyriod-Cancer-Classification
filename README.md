# 🧠 Thyroid Cancer Classification Pipeline

Unified deep learning pipeline for classification of thyroid lesions using cytological and ultrasound images. The project is designed with a **single-entry architecture**, reproducibility, and **paper-level evaluation metrics**.

---

## 📌 Overview

This project implements an end-to-end pipeline:

build_metadata() → train_fold() → evaluate() → visualize_attention()

The pipeline lives in **`main.py`**; the model architecture and evaluation transform live in **`model.py`**, which both `main.py` and `model_inference.py` import — no duplication or inconsistencies between transforms, dataloaders, or model definitions.

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
├── model.py - Shared model definition (encoders + transformer fusion) <br>
├── model_inference.py - Inference script <br>
├── experiments.py - CV modality experiments (E1–E5) <br>
├── ablation.py - Overfitting-control ablation A0–A3 + re-run check <br>
├── dedup.py - Near-duplicate image audit (dHash) for leakage-safe folds <br>
├── reliability.py - ECE + temperature scaling <br>
├── app.py - Gradio MVP web demo <br>
├── deploy/publish.py - Hugging Face model + Space publishing <br>
├── results/ - Experiment tables + out-of-fold predictions <br>
├── tests/ - Unit + system tests (`python -m pytest tests/`, run by GitHub Actions) <br>
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

- ConvNeXt `features[5:]` (stage 3, last downsample, stage 4) and the final LayerNorm are trainable
- Stem, stages 1–2 and their downsamples are frozen
- Controlled by `CFG['trainable_from_feature']`; trainable/total parameter counts are printed at model creation

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

- Early stopping based on **validation AUROC** (`patience=5` < `epochs=10`)
- `CFG['run_all_folds'] = True` trains every fold and writes `graphs/metrics_cv_summary.csv` (mean ± SD)

---

## 🧪 CV Experiments (missing modality)

`experiments.py` runs the modality matrix over all 5 patient-level folds:

| ID | Name | Config |
|----|------|--------|
| E1 | `E1_us_only` | `mode='us_only'` |
| E2 | `E2_cyto_only` | `mode='cyto_only'` |
| E3 | `E3_late_fusion` | mean of E1 + E2 probabilities (no training) |
| E4 | `E4_joint` | `mode='joint'`, `modality_dropout=None` |
| E5 | `E5_joint_mdrop` | `mode='joint'`, `modality_dropout=dict(p_drop_cyto=0.3, p_drop_us=0.15)` |

Each model is evaluated under the test-time scenarios `both`, `us_missing`, `cyto_missing` and `few_patches` (first 2 patches), where they apply. Metrics: AUROC with bootstrap 95% CI, AUPRC, sensitivity/specificity at the Youden threshold, Brier. By default the Youden threshold and the best checkpoint are picked on an inner 15% validation split of the training folds (`inner_val_frac`), not on the test fold.

```bash
python experiments.py                              # E1–E5, 5 folds, 1 seed
python experiments.py --n_seeds 3                  # seeds 42, 43, 44
python experiments.py --experiments E4 E5 --folds 0 1
python experiments.py --reuse_ckpt                 # only train missing checkpoints
python experiments.py --inner_val_frac 0           # select on the test fold (original protocol)
```

Outputs go to `results/`: `cv_experiments.csv` (per experiment × seed × fold × scenario), `cv_experiments_summary.csv` (mean ± SD), `oof_<experiment>.csv` (out-of-fold predictions) and `delta_auroc.csv` (paired-bootstrap ΔAUROC: E4 and E5 vs best single modality, E4/E5 vs late fusion, E5 vs E4).

The same options exist in `main.py`'s `CFG` (`mode`, `modality_dropout`, `inner_val_frac`). Their defaults reproduce the original joint pipeline.

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

`--ckpt` defaults to `checkpoints/model_fold0_best.pth` in all modes.

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

---

## 🧪 System Testing & Evaluation (Week 11)

```bash
# Environment (GPU)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Tests: unit + full-pipeline system test on synthetic data (CPU, no downloads)
python -m pytest tests/ -v

# 5-fold CV with inner-validation selection, calibration and deletion test,
# then the deployment model trained on all patients
python main.py --data_root <folder with dataset1..4> --all_folds --final

# Overfitting ablation A0–A3 (SO1) and same-seed re-run check (SO2)
python ablation.py --data_root <folder> --folds 0 1 2 3 4
python ablation.py --data_root <folder> --configs A3_full --repro --folds 0
```

What the pipeline now guarantees:

- **Leakage-safe splits:** near-duplicate images (256-bit dHash, Hamming ≤ 8) group patients into one fold
  (`StratifiedGroupKFold`); groups that mix benign and PTC labels are dropped and logged to
  `graphs/duplicate_pairs.csv`.
- **Unbiased selection:** checkpoint, Youden threshold and temperature are chosen on an inner, group-aware
  validation split (15%) — never on the test fold.
- **Calibration:** temperature scaling with ECE and Brier before/after (`reliability.py`).
- **Faithfulness:** deletion test — removing the top-relevance patch vs a random patch on true positives.
- **Speed/repro:** mixed precision on CUDA, fixed seeds, deterministic cuDNN.

## 🌐 MVP Web Demo & Deployment

```bash
python app.py                      # local: http://127.0.0.1:7860 (uses checkpoints/medai_final.pth)
hf auth login                      # once, with a write token
python deploy/publish.py --user <hf-username>
```

`deploy/publish.py` uploads `medai_final.pth` to the model repo `<user>/medai-thyroid` and deploys the Gradio
Space `<user>/medai-thyroid-demo` (CPU). The app shows the calibrated P(PTC), the decision at the deployed
threshold and the cytology patches ranked by relevance. **Research use only.**

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

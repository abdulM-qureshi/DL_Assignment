# 🏥 Clinical Early Warning System Patient Survival Prediction

> Predicting in-hospital patient mortality using ICU data through three generations of deep learning models.

**Student:** Wajih ur Rehman
**Roll No:** 22F-BSAI-50
**Dataset:** [Kaggle  Patient Survival Prediction](https://www.kaggle.com/datasets/mitishaagarwal/patient)
**Target Variable:** `hospital_death`  binary classification (0 = Survived, 1 = Died)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Feature Selection](#feature-selection)
- [Model Generations](#model-generations)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Overview

This project builds a Clinical Early Warning System to predict whether an ICU patient will die during hospitalization. Three generations of deep learning models are explored  from standard DNNs to ClinicalBERT  with **recall** as the primary metric, since missing a death prediction carries a higher clinical cost than a false alarm.

**Class Distribution:**
- Survived (0): ~91–92%
- Died (1): ~8–9%

All models use `class_weight='balanced'` to handle this imbalance.

---

## Dataset

The dataset contains ~91,000 ICU patient records with 83 feature columns covering:

- APACHE IV mortality probability scores
- Glasgow Coma Scale (GCS) components
- Day-1 and Hour-1 vital signs (heart rate, blood pressure, SpO2, respiratory rate, temperature)
- Demographics (age, BMI)
- Comorbidities (diabetes, cirrhosis, immunosuppression, solid tumor with metastasis)
- Admission context (elective surgery, pre-ICU LOS)

---

## Feature Selection

Features were selected using a **three-stage pipeline:**

| Stage | Method | Threshold |
|-------|--------|-----------|
| 1 | Missing Value Filtering | Drop columns with > 60% missing |
| 2 | Random Forest Importance | Top features by mean impurity decrease |
| 3 | Point-Biserial Correlation | Absolute correlation with `hospital_death` |

**37 final features** were retained, combining statistical ranking with clinical domain knowledge (APACHE scoring, GCS standards, ICU monitoring protocols).

---

## Model Generations

### Generation 1  Deep Neural Network (DNN)

Architecture: `256 → 128 → 64` fully connected layers with BatchNorm + Dropout(0.3)

| Model | Optimizer | LR |
|-------|-----------|----|
| Model 1 | SGD (momentum=0.9) | 0.01 |
| Model 2 | Adam | 0.001 |

---

### Generation 2  Sequence Models (LSTM / Bi-LSTM / GRU)

Hour-1 vitals → timestep 1, Day-1 vitals → timestep 2
Input shape: `(patients, 2, features)`

| Model | Architecture |
|-------|-------------|
| Model 3 | LSTM (64 → 32) + Dropout |
| Model 4 | Bidirectional LSTM (64 → 32) + Dropout |
| Model 5 | GRU (64 → 32) + Dropout |

---

### Generation 3  ClinicalBERT (NLP)

Synthetic clinical notes are generated from numeric vitals using medically validated thresholds (Sepsis-3, GCS, APACHE). Notes are tokenized and fed into `emilyalsentzer/Bio_ClinicalBERT` pre-trained on MIMIC-III.

| Model | Strategy |
|-------|----------|
| Model 6a | ClinicalBERT  Frozen (classifier head only) |
| Model 6b | ClinicalBERT  Full Fine-Tune (all 108M params) |

> ⚠️ The note generation function never writes "death" or "survived"  preventing direct label leakage.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DNN + SGD | ~0.82 | ~0.35 | ~0.68 | ~0.46 |
| DNN + Adam | ~0.85 | ~0.38 | ~0.72 | ~0.50 |
| LSTM | ~0.83 | ~0.36 | ~0.70 | ~0.48 |
| Bidirectional LSTM | ~0.84 | ~0.37 | ~0.71 | ~0.49 |
| GRU | ~0.83 | ~0.36 | ~0.70 | ~0.48 |
| ClinicalBERT Frozen | ~0.80 | ~0.32 | ~0.65 | ~0.43 |
| **ClinicalBERT Fine-Tune** | **~0.86** | **~0.40** | **~0.74** | **~0.52** |

> **Best Model:** ClinicalBERT Fine-Tune  highest Recall and F1 across all generations.

---

## Project Structure

```
├── Patient_Survival_Notebook.ipynb   # Main notebook (all 5 parts, 39 steps)
├── train.csv                         # Raw dataset (download from Kaggle)
├── Dataset_final.csv                 # Generated: 37-feature cleaned dataset
├── gen1_results.csv                  # DNN model metrics
├── gen2_results.csv                  # LSTM/Bi-LSTM/GRU metrics
├── gen3_results.csv                  # ClinicalBERT metrics
├── final_comparison.csv              # All 6 models combined
├── plots/
│   ├── step0_class_distribution.png
│   ├── step1_missing_values.png
│   ├── step2_rf_importance.png
│   ├── step3_correlation.png
│   ├── gen1_training_history.png
│   ├── gen1_confusion_matrices.png
│   ├── gen2_confusion_matrices.png
│   ├── gen3_confusion_matrices.png
│   ├── gen3_attention_heatmap.png
│   └── final_comparison_chart.png
└── README.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/patient-survival-prediction.git
   cd patient-survival-prediction
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/mitishaagarwal/patient) and place `train.csv` in the root directory.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook Patient_Survival_Notebook.ipynb
   ```

   Run cells in order  Parts 0 through 5. Each part saves its outputs automatically.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow>=2.10
torch
transformers==4.36.0
tokenizers==0.15.0
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch transformers==4.36.0 tokenizers==0.15.0 jupyter
```

> **Note:** ClinicalBERT (Part 4) requires a GPU for reasonable training time. On CPU, expect significantly longer runtimes for Models 6a and 6b.

---

## Key Design Decisions

- **60% missing threshold**  slightly stricter than PhysioNet's 70% to reduce imputation noise on this smaller dataset
- **Balanced class weights**  preferred over SMOTE to avoid synthetic sample overfitting
- **StandardScaler fit on train only**  prevents data leakage from test/validation sets
- **Recall as primary metric**  false negatives (missed deaths) carry higher clinical cost than false positives
- **Synthetic notes avoid leakage**  note generator uses clinical thresholds, never writes outcome labels directly

---

*Bachelor of Science in Artificial Intelligence | DUET KARACHI*- Admission context (elective surgery, pre-ICU LOS)

---

## Feature Selection

Features were selected using a **three-stage pipeline:**

| Stage | Method | Threshold |
|-------|--------|-----------|
| 1 | Missing Value Filtering | Drop columns with > 60% missing |
| 2 | Random Forest Importance | Top features by mean impurity decrease |
| 3 | Point-Biserial Correlation | Absolute correlation with `hospital_death` |

**37 final features** were retained, combining statistical ranking with clinical domain knowledge (APACHE scoring, GCS standards, ICU monitoring protocols).

---

## Model Generations

### Generation 1 — Deep Neural Network (DNN)

Architecture: `256 → 128 → 64` fully connected layers with BatchNorm + Dropout(0.3)

| Model | Optimizer | LR |
|-------|-----------|----|
| Model 1 | SGD (momentum=0.9) | 0.01 |
| Model 2 | Adam | 0.001 |

---

### Generation 2 — Sequence Models (LSTM / Bi-LSTM / GRU)

Hour-1 vitals → timestep 1, Day-1 vitals → timestep 2
Input shape: `(patients, 2, features)`

| Model | Architecture |
|-------|-------------|
| Model 3 | LSTM (64 → 32) + Dropout |
| Model 4 | Bidirectional LSTM (64 → 32) + Dropout |
| Model 5 | GRU (64 → 32) + Dropout |

---

### Generation 3 — ClinicalBERT (NLP)

Synthetic clinical notes are generated from numeric vitals using medically validated thresholds (Sepsis-3, GCS, APACHE). Notes are tokenized and fed into `emilyalsentzer/Bio_ClinicalBERT` pre-trained on MIMIC-III.

| Model | Strategy |
|-------|----------|
| Model 6a | ClinicalBERT — Frozen (classifier head only) |
| Model 6b | ClinicalBERT — Full Fine-Tune (all 108M params) |

> ⚠️ The note generation function never writes "death" or "survived" — preventing direct label leakage.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| DNN + SGD | ~0.82 | ~0.35 | ~0.68 | ~0.46 |
| DNN + Adam | ~0.85 | ~0.38 | ~0.72 | ~0.50 |
| LSTM | ~0.83 | ~0.36 | ~0.70 | ~0.48 |
| Bidirectional LSTM | ~0.84 | ~0.37 | ~0.71 | ~0.49 |
| GRU | ~0.83 | ~0.36 | ~0.70 | ~0.48 |
| ClinicalBERT Frozen | ~0.80 | ~0.32 | ~0.65 | ~0.43 |
| **ClinicalBERT Fine-Tune** | **~0.86** | **~0.40** | **~0.74** | **~0.52** |

> **Best Model:** ClinicalBERT Fine-Tune — highest Recall and F1 across all generations.

---

## Project Structure

```
├── Patient_Survival_Notebook.ipynb   # Main notebook (all 5 parts, 39 steps)
├── train.csv                         # Raw dataset (download from Kaggle)
├── Dataset_final.csv                 # Generated: 37-feature cleaned dataset
├── gen1_results.csv                  # DNN model metrics
├── gen2_results.csv                  # LSTM/Bi-LSTM/GRU metrics
├── gen3_results.csv                  # ClinicalBERT metrics
├── final_comparison.csv              # All 6 models combined
├── plots/
│   ├── step0_class_distribution.png
│   ├── step1_missing_values.png
│   ├── step2_rf_importance.png
│   ├── step3_correlation.png
│   ├── gen1_training_history.png
│   ├── gen1_confusion_matrices.png
│   ├── gen2_confusion_matrices.png
│   ├── gen3_confusion_matrices.png
│   ├── gen3_attention_heatmap.png
│   └── final_comparison_chart.png
└── README.md
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/patient-survival-prediction.git
   cd patient-survival-prediction
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/mitishaagarwal/patient) and place `train.csv` in the root directory.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**
   ```bash
   jupyter notebook Patient_Survival_Notebook.ipynb
   ```

   Run cells in order — Parts 0 through 5. Each part saves its outputs automatically.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow>=2.10
torch
transformers==4.36.0
tokenizers==0.15.0
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow torch transformers==4.36.0 tokenizers==0.15.0 jupyter
```

> **Note:** ClinicalBERT (Part 4) requires a GPU for reasonable training time. On CPU, expect significantly longer runtimes for Models 6a and 6b.

---

## Key Design Decisions

- **60% missing threshold** — slightly stricter than PhysioNet's 70% to reduce imputation noise on this smaller dataset
- **Balanced class weights** — preferred over SMOTE to avoid synthetic sample overfitting
- **StandardScaler fit on train only** — prevents data leakage from test/validation sets
- **Recall as primary metric** — false negatives (missed deaths) carry higher clinical cost than false positives
- **Synthetic notes avoid leakage** — note generator uses clinical thresholds, never writes outcome labels directly

---

*Bachelor of Science in Artificial Intelligence | DUET KARACHI*

# Understanding Linear and Logistic Regression

### Biomedical Machine Learning Lab 1 — A Reproducible Two-Dataset Study

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![Status](https://img.shields.io/badge/Status-Submitted-success)
![Purpose](https://img.shields.io/badge/Purpose-Educational-lightgrey)

| | |
|---|---|
| **Author** | Burak Bekir Çakırer (Student ID: 2211011008) |
| **Course** | [Biomedical System Design Capsule-S1](https://canvas.agu.edu.tr/courses/374), Abdullah Gül University |
| **Submission date** | 21 October 2026 |
| **Language** | Python 3 · English report |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Datasets](#2-datasets)
3. [Methodology](#3-methodology)
4. [Key Results](#4-key-results)
5. [Figure Previews](#5-figure-previews)
6. [Repository Structure](#6-repository-structure)
7. [Getting Started](#7-getting-started)
8. [Reports and Overleaf](#8-reports-and-overleaf)
9. [Limitations and Research Integrity](#9-limitations-and-research-integrity)
10. [References](#10-references)

---

## 1. Abstract

This repository contains a reproducible investigation of **linear regression** (continuous outcome) and **logistic regression** (binary classification) on two real biomedical datasets. It covers experimental sections **A–D** of the *Understanding Linear and Logistic Regression* lab brief.

Only two input features per dataset are modeled, so that coefficients and decision boundaries remain visually interpretable. Analysis scripts, exported results, figures, and reports are kept in separate directories. The submitted PDF, Word, and LaTeX reports contain **no source code**.

## 2. Datasets

Both datasets are distributed with scikit-learn and load offline.

| Dataset | Task | Outcome | Samples | Selected features |
|---|---|---|---:|---|
| [**Diabetes**](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) | Linear regression | One-year disease progression score | 442 | BMI, average blood pressure |
| [**Breast Cancer Wisconsin (Diagnostic)**](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic) | Logistic regression | Malignant (1) vs. benign (0) | 569 | Worst radius, worst concave points |

> **Label encoding.** The original scikit-learn loader uses `0 = malignant, 1 = benign`. This project recodes labels to **`1 = malignant, 0 = benign`**, so that malignant is the positive class.

The files in `results/data/` are two-feature exports with recorded train/test membership. They are not additional datasets.

## 3. Methodology

| Component | Configuration |
|---|---|
| **Data split** | 80% training / 20% held-out test, `random_state=42`; stratified by class for the classification task |
| **Scaling** | Fitted on training data only; applied to the test set without refitting (no data leakage) |
| **Linear model** | Ordinary least squares (unscaled vs. standardized comparison) |
| **Convergence study** | Batch gradient descent, 150 iterations, learning rates 0.01, 0.10, 0.80, 1.50; training loss monitored |
| **Logistic model** | Standardized inputs, L2 regularization (`C=1.0`), `lbfgs` solver, `max_iter=2000` |
| **Threshold analysis** | Thresholds 0.3, 0.5, 0.7 applied to the same test probabilities, without retraining |
| **Feature comparison** | One vs. two features, identical partition, threshold 0.5 |

**Environment:** Python 3.13.5 · NumPy 2.3.5 · pandas 2.2.3 · scikit-learn 1.8.0 · Matplotlib 3.10.8 (see [`requirements.txt`](requirements.txt)).

## 4. Key Results

All metrics are computed on the held-out test partition unless stated otherwise.

| ID | Experiment | Outcome |
|---|---|---|
| **B1** | Baseline linear model | MAE **50.044** · MSE **3740.507** · R² **0.294** |
| **B2** | Feature standardization | OLS predictions and test error unchanged to floating-point precision; only coefficient units change |
| **B3** | Learning-rate study | On standardized training data, η = **0.80** converged fastest of the four settings; η = **1.50** diverged |
| **C1** | Logistic model, threshold 0.5 | Accuracy **95.61%** · Precision **0.974** · Recall **0.905** · F1 **0.938** |
| **C2** | Threshold sensitivity (0.3 / 0.5 / 0.7) | Malignant recall **1.000 / 0.905 / 0.810** (same model, same probabilities) |
| **C3** | One vs. two features | Accuracy improved from **90.35%** to **95.61%** with the second feature |

Coefficient tables, the confusion matrix, interpretations, and all nine figures are in the [PDF report](report/Biomedical_ML_Lab1_Report.pdf).

## 5. Figure Previews

**Linear regression: observed vs. predicted values (test data)**

![Linear regression predictions on holdout data](results/figures/fig02_regression_actual_predicted.png)

**Logistic regression: two-feature decision boundary**

![Logistic regression decision boundary](results/figures/fig08_decision_boundary.png)

## 6. Repository Structure

```text
Biomedical_ML_Lab1/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── run_lab.py                  # Reproducible experiment pipeline
├── results/
│   ├── data/
│   │   ├── diabetes_two_feature_split.csv
│   │   ├── breast_cancer_two_feature_split.csv
│   │   └── metrics.json            # Detailed numerical results
│   └── figures/                    # Nine generated experiment figures
├── report/
│   ├── Biomedical_ML_Lab1_Report.pdf
│   ├── Biomedical_ML_Lab1_Report.docx
│   ├── main.tex                    # Overleaf-compatible LaTeX source
│   └── figures/                    # Figures referenced by main.tex
└── docs/
    └── GITHUB_DESKTOP_TR.md        # Windows GitHub Desktop guide (Turkish)
```

## 7. Getting Started

### Prerequisites

- Python **3.10 or newer** (3.13.5 was used for the supplied results)

### Installation and execution (Windows / PowerShell)

```powershell
cd D:\GitHub\Biomedical_ML_Lab1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\run_lab.py
```

The script writes `results/data/metrics.json`, both annotated data exports, and all nine plots to `results/figures/`. No network connection is needed after package installation.

> **Note:** If PowerShell script execution is restricted by system policy, skip activation and run
> `.\.venv\Scripts\python.exe src\run_lab.py` directly.

For a quick run without a virtual environment: `py -m pip install -r requirements.txt` followed by `py src\run_lab.py`.

## 8. Reports and Overleaf

| Format | File |
|---|---|
| PDF (submission copy) | [`Biomedical_ML_Lab1_Report.pdf`](report/Biomedical_ML_Lab1_Report.pdf) |
| Word (editable) | [`Biomedical_ML_Lab1_Report.docx`](report/Biomedical_ML_Lab1_Report.docx) |
| LaTeX source | [`main.tex`](report/main.tex) |

**To compile on Overleaf:** create a project, upload `report/main.tex` together with the entire `report/figures/` directory (preserving relative paths), set `main.tex` as the main document, and compile with **pdfLaTeX**.

## 9. Limitations and Research Integrity

- This work is an **educational demonstration** and is **not a clinical decision system**.
- Only two of the 10 (Diabetes) and 30 (Breast Cancer) available predictors are used.
- Reported scores come from a **single held-out test partition**, so they carry sampling variability.
- The threshold comparison in C2 is illustrative. Choosing a classification cutoff from test-set results is **not a valid deployment or clinical threshold-selection procedure**; that would require independent validation, calibration, and explicit assessment of error costs.

## 10. References

1. Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least Angle Regression. *The Annals of Statistics*, 32(2), 407–499. https://doi.org/10.1214/009053604000000067 *(Diabetes dataset background)*
2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
3. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *SPIE 1905*, 861–870. https://doi.org/10.1117/12.148698
4. scikit-learn documentation: [`load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) · [`load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

The full bibliography, dataset provenance, and experimental discussion are in the [report](report/Biomedical_ML_Lab1_Report.pdf).

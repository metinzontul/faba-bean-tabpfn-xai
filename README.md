# Explainable TabPFN-Based Machine Learning for Single-Plant Yield Estimation and Trait Prioritization in Faba Bean (Vicia faba L.)
This repository contains the official data preprocessing, hyperparameter
optimization, model training, and explainability (XAI) scripts for the research
article:

**"Explainable TabPFN-Based Machine Learning for Single-Plant Yield Estimation
and Trait Prioritization in Faba Bean (*Vicia faba* L.)"**

> **Status:** the associated article is currently published.

**Code author:** İlkay Yelmen

The repository is hosted on the GitHub account of Metin Zontul on behalf of the
author team; authorship of the code and of this documentation belongs to İlkay
Yelmen.

## Overview

This project evaluates the predictive performance and biological
interpretability of a pre-trained tabular foundation model (**TabPFN**) against
conventional machine learning baselines (Random Forest, Extra Trees, XGBoost,
HistGradientBoosting, SVR). The framework uses directly measured and
domain-engineered agronomic traits to estimate plot-mean single-plant yield
(g plant⁻¹) from 19 predictors — 13 measured traits and 6 derived features.
It further incorporates complementary Explainable AI techniques (**SHAP,
Permutation Importance, and LOCO**) to prioritize yield-determining traits and
support breeding decisions.

Every script is seeded, so running them on the same data reproduces the reported
numbers exactly.

## Contents

* **Data preprocessing** — quantile transformation and train/test splitting,
  applied inside a scikit-learn `Pipeline` within every cross-validation fold so
  that no information leaks from validation or test data.
* **Hyperparameter tuning** — fully seeded, exactly reproducible `Optuna`
  workflows for the five baseline models.
* **Model training and evaluation** — performance metrics (R², RMSE, MAE, MAPE)
  and statistical comparisons (Holm-corrected Wilcoxon, Friedman/Nemenyi),
  together with an ablation over predictor sets and a repeated-split robustness
  analysis.
* **Explainability (XAI)** — SHAP value calculations, permutation importance,
  and Leave-One-Covariate-Out (LOCO) analyses, plus feature-dependence
  diagnostics (correlations, VIF, trait clusters) needed to interpret them.
* **Requirements** — pinned software dependencies (`requirements.txt`) for exact
  reproducibility.

## Dataset

`data/faba_bean_dataset.csv` — 398 rows, semicolon-separated, decimal point.

Each row is one **experimental plot** (one genotype in one block of the
augmented design), not an individual plant. Plant-level traits, including the
target, are arithmetic means of the plants sampled within the plot.

| Column | Type | Description |
|---|---|---|
| `Days_to_Flowering` | measured | Days from sowing to 50% flowering |
| `Days_to_Maturity` | measured | Days from sowing to physiological maturity |
| `First_Pod_Height_cm` | measured | Height of the first pod from the ground (cm) |
| `Branch_Number` | measured | Branches per plant |
| `Plant_Height_cm` | measured | Plant height (cm) |
| `Biological_Yield_g` | measured | Aboveground dry matter per plant (g) |
| `Pod_Number` | measured | Pods per plant |
| `Pod_Weight_g` | measured | Total pod weight per plant (g) |
| `Pod_Length_cm` | measured | Mean pod length (cm) |
| `Seeds_per_Pod` | measured | Mean seeds per pod |
| `Seeds_per_Plant` | measured | Seeds per plant |
| `Plot_Yield_kg_da` | measured | Plot yield (kg da⁻¹) |
| `Seed_Weight_100_g` | measured | Weight of 100 seeds (g) |
| `Pod_Filling_Rate` | derived | Pod weight / pod number |
| `Seed_to_Pod_Ratio` | derived | Seeds per plant / pod number |
| `Seed_Unit_Weight` | derived | Pod weight / seeds per plant |
| `Pod_Length_Yield_Index` | derived | Pod weight / pod length |
| `Plant_Yield_Ratio` | derived | Biological yield / pod number |
| `Vegetation_Period_days` | derived | Days to maturity − days to flowering |
| **`Single_Plant_Yield_g`** | **target** | **Plot-mean single-plant yield (g plant⁻¹)** |

## Protocol

| Setting | Value |
|---|---|
| Train / test split | 80% / 20%, `random_state=42` → 318 / 80 |
| Preprocessing | `QuantileTransformer(output_distribution='normal', n_quantiles=200)`, fitted inside each fold |
| Cross-validation | `KFold(n_splits=5, shuffle=True, random_state=42)` |
| Baseline tuning | Optuna 4.8.0, TPE sampler, **seed = 42**, 100 trials per model, objective = mean 5-fold CV R² |
| TabPFN | Not tuned; `n_estimators` fixed a priori at 4 |
| Repeated-split analysis | 10 independent 80/20 splits, seeds 0–9 |

Two details matter for reproducibility. First, every Optuna study uses a seeded
`TPESampler`, so repeated runs give identical configurations. Second, the
transformer is placed inside a `Pipeline`, so it is never fitted on validation
or test data.

## Repository structure

```
.
├── data/
│   └── faba_bean_dataset.csv               398 plots, 19 predictors + target
├── analysis/
│   ├── feature_dependence.py               correlations, VIF, trait clusters
│   ├── model_comparison_seeded.py          reference-split model comparison,
│   │                                       Wilcoxon, Friedman, Nemenyi, figures
│   ├── SHAP_analysis.py                    SHAP explainability
│   ├── permutation_importance.py           permutation importance, 50 repeats
│   ├── loco_analysis.py                    leave-one-covariate-out
│   ├── repeated_split_evaluation.py        ablation and robustness over 10 splits
│   └── wilcoxon_test.py                    earlier standalone Wilcoxon script
├── TabPFN.py                               single-model scripts, one per algorithm
├── RandomForest.py
├── ExtraTrees.py
├── SVR.py
├── HGB.py
├── XGBoost.py
├── MLP.py                                  not used in the manuscript
├── Ensemble.py                             not used in the manuscript
├── run_all.py
└── requirements.txt
```

`analysis/model_comparison_seeded.py` is the reference implementation of the
model comparison. The single-model scripts in the repository root are kept for
transparency but use an unseeded Optuna sampler, so their tuned configurations
vary slightly between runs.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the reported
results. TabPFN requires PyTorch; the reported runs used the CUDA 11.8 build.
On a machine without a GPU the CPU build is sufficient and TabPFN falls back to
the CPU automatically.

> TabPFN requires a free account at [tabpfn.com](https://tabpfn.com). On first
> run you will be prompted to log in through your browser. Credentials are
> stored locally and are not part of this repository.

## Usage

Run the analyses individually:

```bash
python analysis/feature_dependence.py            # correlations, VIF, trait clusters
python analysis/model_comparison_seeded.py       # model comparison and statistical tests
python analysis/SHAP_analysis.py                 # SHAP values and plots
python analysis/permutation_importance.py        # permutation importance
python analysis/loco_analysis.py                 # LOCO feature contribution
python analysis/repeated_split_evaluation.py     # ablation and repeated-split robustness
```

Or run everything, including the single-model scripts:

```bash
python run_all.py
```

All outputs are written to `results/`, which is git-ignored.

Approximate runtimes on a 20-thread CPU with a single GPU: the reference-split
comparison takes about 30 min, and the repeated-split evaluation about 1.5 h,
both dominated by the HistGradientBoosting searches.

## Notes on reproducibility

- The reported TabPFN results are deterministic and reproduce exactly.
- The tuned baselines reproduce exactly when run through
  `analysis/model_comparison_seeded.py` or
  `analysis/repeated_split_evaluation.py`, which seed the Optuna sampler. The
  single-model scripts in the repository root do not seed it.
- In the repeated-split analysis the baseline hyperparameters are tuned once on
  the reference split and then held fixed across the 10 evaluation splits.
  Retuning within each split would require roughly 15 h. Because TabPFN is the
  only model that needs no tuning, this design may mildly favour it; the
  manuscript states this as a limitation.

## Citation

> *Citation will be added upon publication.*

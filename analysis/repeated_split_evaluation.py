"""
repeated_split_evaluation.py — repeated-split ablation and robustness analysis.

Reproduces the ablation and repeated-split robustness analyses of the study.

The single reference split (random_state = 42) describes one particular
partition of the 398 plot-level observations. Because the standard deviation of
the test R² across partitions is larger than the differences between models,
the whole evaluation is repeated over 10 independent 80/20 splits (seeds 0-9)
for three predictor configurations:

  S1  13 directly measured agronomic traits
  S2  19 predictors (measured + derived) — the configuration used in the paper
  S3  12 predictors, after removing the target-proximal yield variables
      (pod weight, biological yield, plot yield) together with every derived
      feature computed from them (pod filling rate, seed unit weight,
      pod length-yield index, plant yield ratio)

Protocol and its limitation
  Baseline hyperparameters are tuned ONCE per predictor set on the reference
  split (Optuna, 100 trials, TPESampler(seed=42)) and then held fixed across the
  10 evaluation splits; retuning within every split would require roughly 15 h
  of computation. The same rule is applied to all five baselines, but TabPFN is
  the only model that needs no tuning at all, so the design may mildly favour
  TabPFN. This is stated as a limitation in the manuscript.

  TabPFN is refitted on every split with n_estimators = 4 and no tuning.
  In every split the QuantileTransformer is fitted on the training part only.

Outputs (../results/)
  ablation_repeated_splits.csv     mean ± SD per predictor set and model
  ablation_paired_tests.csv        S1↔S2, S2↔S3 and S1↔S3 paired tests
  robustness_repeated_splits.csv   S2 robustness summary with win counts
  tabpfn_vs_baselines.csv          TabPFN against each baseline over the splits
  repeated_split_raw.csv           every split × predictor set × model metric
  repeated_split_best_params.csv   the tuned configurations that were used

Runtime: roughly 1.5 h on a 20-thread CPU with a single GPU, dominated by the
HistGradientBoosting searches.
"""

import warnings; warnings.filterwarnings("ignore")
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "faba_bean_dataset.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TARGET = "Single_Plant_Yield_g"
N_TRIALS = 100
TABPFN_NE = 4
ALPHA = 0.05
TUNE_SEED = 42
EVAL_SEEDS = list(range(10))
KF = KFold(n_splits=5, shuffle=True, random_state=TUNE_SEED)

ORIGINAL = ["Days_to_Flowering", "Days_to_Maturity", "First_Pod_Height_cm",
            "Branch_Number", "Plant_Height_cm", "Biological_Yield_g",
            "Pod_Number", "Pod_Weight_g", "Pod_Length_cm", "Seeds_per_Pod",
            "Seeds_per_Plant", "Plot_Yield_kg_da", "Seed_Weight_100_g"]
DERIVED = ["Pod_Filling_Rate", "Seed_to_Pod_Ratio", "Seed_Unit_Weight",
           "Pod_Length_Yield_Index", "Plant_Yield_Ratio",
           "Vegetation_Period_days"]
TARGET_PROXIMAL = ["Pod_Weight_g", "Biological_Yield_g", "Plot_Yield_kg_da",
                   "Pod_Filling_Rate", "Seed_Unit_Weight",
                   "Pod_Length_Yield_Index", "Plant_Yield_Ratio"]

SETS = {
    "S1": ORIGINAL,
    "S2": ORIGINAL + DERIVED,
    "S3": [c for c in ORIGINAL + DERIVED if c not in TARGET_PROXIMAL],
}
MODELS = ["TabPFN", "RandomForest", "SVR", "ExtraTrees", "HGB", "XGBoost"]
CTOR = {"RandomForest": RandomForestRegressor, "SVR": NuSVR,
        "ExtraTrees": ExtraTreesRegressor,
        "HGB": HistGradientBoostingRegressor, "XGBoost": XGBRegressor}

df = pd.read_csv(DATA, sep=";")
y_all = df[TARGET].values
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  observations={len(y_all)}")


def make_qt(seed):
    return QuantileTransformer(output_distribution="normal", n_quantiles=200,
                               random_state=seed)


def make_study():
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))


def metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mean_absolute_error(y_true, y_pred),
            float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100))


TREE = {"n": "n_estimators", "d": "max_depth", "mss": "min_samples_split",
        "msl": "min_samples_leaf", "mf": "max_features"}


def search_spaces(X_tr, y_tr):
    """Objective, name mapping and fixed arguments for each baseline model."""

    def cv_r2(model):
        return cross_val_score(Pipeline([("qt", make_qt(TUNE_SEED)), ("m", model)]),
                               X_tr, y_tr, cv=KF, scoring="r2").mean()

    return {
        "RandomForest": (
            lambda t: cv_r2(RandomForestRegressor(
                n_estimators=t.suggest_int("n", 100, 600, step=50),
                max_depth=t.suggest_categorical("d", [None, 5, 10, 15, 20]),
                min_samples_split=t.suggest_int("mss", 2, 15),
                min_samples_leaf=t.suggest_int("msl", 1, 8),
                max_features=t.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
                random_state=TUNE_SEED, n_jobs=-1)),
            TREE, {"random_state": TUNE_SEED, "n_jobs": -1}),
        "SVR": (
            lambda t: cv_r2(NuSVR(
                kernel="rbf",
                nu=t.suggest_float("nu", 0.01, 0.99),
                C=t.suggest_float("C", 0.1, 300.0, log=True),
                gamma=t.suggest_float("gamma", 1e-4, 10.0, log=True))),
            {}, {"kernel": "rbf"}),
        "ExtraTrees": (
            lambda t: cv_r2(ExtraTreesRegressor(
                n_estimators=t.suggest_int("n", 100, 600, step=50),
                max_depth=t.suggest_categorical("d", [None, 5, 10, 15, 20]),
                min_samples_split=t.suggest_int("mss", 2, 15),
                min_samples_leaf=t.suggest_int("msl", 1, 8),
                max_features=t.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
                random_state=TUNE_SEED, n_jobs=-1)),
            TREE, {"random_state": TUNE_SEED, "n_jobs": -1}),
        "HGB": (
            lambda t: cv_r2(HistGradientBoostingRegressor(
                max_iter=t.suggest_int("mi", 100, 800, step=50),
                max_depth=t.suggest_int("md", 2, 12),
                learning_rate=t.suggest_float("lr", 0.005, 0.3, log=True),
                min_samples_leaf=t.suggest_int("msl", 5, 40),
                l2_regularization=t.suggest_float("l2", 1e-4, 10.0, log=True),
                max_leaf_nodes=t.suggest_int("mln", 10, 80),
                random_state=TUNE_SEED)),
            {"mi": "max_iter", "md": "max_depth", "lr": "learning_rate",
             "msl": "min_samples_leaf", "l2": "l2_regularization",
             "mln": "max_leaf_nodes"},
            {"random_state": TUNE_SEED}),
        "XGBoost": (
            lambda t: cv_r2(XGBRegressor(
                n_estimators=t.suggest_int("n", 100, 600, step=50),
                max_depth=t.suggest_int("md", 2, 10),
                learning_rate=t.suggest_float("lr", 0.005, 0.3, log=True),
                subsample=t.suggest_float("ss", 0.5, 1.0),
                colsample_bytree=t.suggest_float("cbt", 0.5, 1.0),
                reg_alpha=t.suggest_float("ra", 1e-4, 10.0, log=True),
                reg_lambda=t.suggest_float("rl", 1e-4, 10.0, log=True),
                min_child_weight=t.suggest_int("mcw", 1, 8),
                random_state=TUNE_SEED, n_jobs=-1, verbosity=0)),
            {"n": "n_estimators", "md": "max_depth", "lr": "learning_rate",
             "ss": "subsample", "cbt": "colsample_bytree", "ra": "reg_alpha",
             "rl": "reg_lambda", "mcw": "min_child_weight"},
            {"random_state": TUNE_SEED, "n_jobs": -1, "verbosity": 0}),
    }


# ---- step 1: tune each baseline once per predictor set, on the reference split
PARAMS = {}
for set_name, feats in SETS.items():
    X_all = df[feats].values
    X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=0.20,
                                        random_state=TUNE_SEED)
    spaces = search_spaces(X_tr, y_tr)
    print(f"\n[tuning] {set_name} ({len(feats)} predictors)")
    for name in MODELS:
        if name == "TabPFN":
            continue
        t0 = time.time()
        objective, rename, fixed = spaces[name]
        study = make_study()
        study.optimize(objective, n_trials=N_TRIALS)
        params = {rename.get(k, k): v for k, v in study.best_params.items()}
        params.update(fixed)
        PARAMS[(set_name, name)] = params
        print(f"  {name:<13} CV R2={study.best_value:.4f}  ({time.time() - t0:.0f}s)")

pd.DataFrame([{"Feature_set": s, "Model": m,
               "Best_parameters": json.dumps(p, default=str)}
              for (s, m), p in PARAMS.items()]).to_csv(
    OUT / "repeated_split_best_params.csv", index=False, sep=";")

# ---- step 2: evaluate over the 10 splits with the tuned configurations -------
rows = []
for set_name, feats in SETS.items():
    X_all = df[feats].values
    for seed in EVAL_SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.20,
                                                  random_state=seed)
        for name in MODELS:
            if name == "TabPFN":
                qt = make_qt(seed)
                model = TabPFNRegressor(n_estimators=TABPFN_NE, device=device,
                                        random_state=TUNE_SEED)
                model.fit(qt.fit_transform(X_tr), y_tr)
                pred = model.predict(qt.transform(X_te))
            else:
                pipe = Pipeline([("qt", make_qt(seed)),
                                 ("m", CTOR[name](**PARAMS[(set_name, name)]))])
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_te)
            r2, rmse, mae, mape = metrics(y_te, pred)
            rows.append({"Feature_set": set_name, "Seed": seed, "Model": name,
                         "Test_R2": r2, "Test_RMSE": rmse, "Test_MAE": mae,
                         "Test_MAPE": mape})
    print(f"[evaluated] {set_name}")

raw = pd.DataFrame(rows)
raw.to_csv(OUT / "repeated_split_raw.csv", index=False, sep=";")

summary = (raw.groupby(["Feature_set", "Model"])
              .agg(R2_mean=("Test_R2", "mean"), R2_sd=("Test_R2", "std"),
                   RMSE_mean=("Test_RMSE", "mean"), RMSE_sd=("Test_RMSE", "std"),
                   MAE_mean=("Test_MAE", "mean"), MAE_sd=("Test_MAE", "std"),
                   MAPE_mean=("Test_MAPE", "mean"), MAPE_sd=("Test_MAPE", "std"))
              .round(4).reset_index())

# ---- ablation over predictor sets ------------------------------------------
summary.to_csv(OUT / "ablation_repeated_splits.csv", index=False, sep=";")

paired = []
for name in MODELS:
    for a, b in (("S1", "S2"), ("S2", "S3"), ("S1", "S3")):
        ra = raw[(raw.Feature_set == a) & (raw.Model == name)].sort_values("Seed")["Test_R2"].values
        rb = raw[(raw.Feature_set == b) & (raw.Model == name)].sort_values("Seed")["Test_R2"].values
        _, p = wilcoxon(ra, rb, alternative="two-sided")
        paired.append({"Model": name, "Set_A": a, "Set_B": b,
                       "R2_A_mean": round(ra.mean(), 4),
                       "R2_B_mean": round(rb.mean(), 4),
                       "Delta_R2": round(rb.mean() - ra.mean(), 4),
                       "B_better_in_n_of_10": int((rb > ra).sum()),
                       "Wilcoxon_p": round(float(p), 4)})
pd.DataFrame(paired).to_csv(OUT / "ablation_paired_tests.csv",
                            index=False, sep=";")

# ---- robustness of the model comparison on S2 -------------------------------
s2 = raw[raw.Feature_set == "S2"]
wins = s2.loc[s2.groupby("Seed")["Test_R2"].idxmax(), "Model"].value_counts()

tab = s2[s2.Model == "TabPFN"].sort_values("Seed")["Test_R2"].values
comparison = []
for name in MODELS:
    if name == "TabPFN":
        comparison.append({"Model": name, "Times_best_of_10": int(wins.get(name, 0)),
                           "Delta_R2_vs_TabPFN": None,
                           "TabPFN_better_in_n_of_10": None, "Wilcoxon_p": None})
        continue
    base = s2[s2.Model == name].sort_values("Seed")["Test_R2"].values
    _, p = wilcoxon(tab, base, alternative="two-sided")
    comparison.append({"Model": name, "Times_best_of_10": int(wins.get(name, 0)),
                       "Delta_R2_vs_TabPFN": round(tab.mean() - base.mean(), 4),
                       "TabPFN_better_in_n_of_10": int((tab > base).sum()),
                       "Wilcoxon_p": round(float(p), 4)})

comparison = pd.DataFrame(comparison)
# Holm correction across the five TabPFN-versus-baseline comparisons
mask = comparison["Wilcoxon_p"].notna()
sub = comparison[mask].sort_values("Wilcoxon_p")
holm, running = [], 0.0
for i, p in enumerate(sub["Wilcoxon_p"].values):
    running = max(running, (len(sub) - i) * p)
    holm.append(min(running, 1.0))
sub["Wilcoxon_p_holm"] = np.round(holm, 4)
comparison = comparison.merge(sub[["Model", "Wilcoxon_p_holm"]], on="Model",
                              how="left")

robustness = (summary[summary.Feature_set == "S2"]
           .drop(columns="Feature_set")
           .merge(comparison, on="Model")
           .sort_values("R2_mean", ascending=False)
           .reset_index(drop=True))
robustness.to_csv(OUT / "robustness_repeated_splits.csv", index=False, sep=";")
comparison.to_csv(OUT / "tabpfn_vs_baselines.csv", index=False, sep=";")

pd.set_option("display.width", 200)
print("\n=== Robustness on S2 over 10 repeated splits ===")
print(robustness.to_string(index=False))
print("\n=== Ablation, paired tests across splits ===")
print(pd.DataFrame(paired).to_string(index=False))
print(f"\nOutputs written to {OUT}")

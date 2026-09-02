import warnings; warnings.filterwarnings("ignore")
import itertools, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import NuSVR
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = Path(__file__).parent.parent / "data" / "faba_bean_dataset.csv"
OUT_DIR   = Path(__file__).parent
TARGET    = "Single_Plant_Yield_g"
N_TRIALS  = 20
ALPHA     = 0.05
TUNE_KF   = KFold(n_splits=5, shuffle=True, random_state=42)
SEP       = "=" * 70

def make_qt():
    return QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=42)

print(SEP)
print("  Wilcoxon Signed-Rank Test — 6 Models / MLP Excluded (Test-Set Absolute Errors)")
print(SEP)

df    = pd.read_csv(DATA_PATH, sep=";")
FEATS = [c for c in df.columns if c != TARGET]
X_raw = df[FEATS].values
y     = df[TARGET].values

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_raw, y, test_size=0.20, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}  |  Features: {len(FEATS)}")
print(f"  Wilcoxon N = {len(y_te)} paired absolute errors per comparison")
print(f"  Optuna trials: {N_TRIALS}  |  device: {device}")
print(SEP)

test_preds = {}

# TabPFN (manual scaling, fits scaler on X_tr only)
print("\n[1/6] TabPFN ...")
t0 = time.time()
qt_tab = make_qt()
Xtr_sc = qt_tab.fit_transform(X_tr_raw)
Xte_sc = qt_tab.transform(X_te_raw)
tab_m  = TabPFNRegressor(n_estimators=4, device=device, random_state=42)
tab_m.fit(Xtr_sc, y_tr)
test_preds["TabPFN"] = tab_m.predict(Xte_sc)
print(f"  MAE={mean_absolute_error(y_te, test_preds['TabPFN']):.4f}  R2={r2_score(y_te, test_preds['TabPFN']):.4f}  ({time.time()-t0:.0f}s)")

# RandomForest
print("\n[2/6] RandomForest ...")
t0 = time.time()
def rf_obj(trial):
    p = dict(n_estimators=trial.suggest_int("n", 100, 600, step=50),
             max_depth=trial.suggest_categorical("d", [None, 5, 10, 15, 20]),
             min_samples_split=trial.suggest_int("mss", 2, 15),
             min_samples_leaf=trial.suggest_int("msl", 1, 8),
             max_features=trial.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
             random_state=42, n_jobs=-1)
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", RandomForestRegressor(**p))]),
                           X_tr_raw, y_tr, cv=TUNE_KF, scoring="r2").mean()
st = optuna.create_study(direction="maximize"); st.optimize(rf_obj, n_trials=N_TRIALS)
_RF = {"n": "n_estimators", "d": "max_depth", "mss": "min_samples_split", "msl": "min_samples_leaf", "mf": "max_features"}
bp = {_RF.get(k, k): v for k, v in st.best_params.items()}
bp.update({"random_state": 42, "n_jobs": -1})
pipe = Pipeline([("qt", make_qt()), ("m", RandomForestRegressor(**bp))]); pipe.fit(X_tr_raw, y_tr)
test_preds["RandomForest"] = pipe.predict(X_te_raw)
print(f"  MAE={mean_absolute_error(y_te, test_preds['RandomForest']):.4f}  R2={r2_score(y_te, test_preds['RandomForest']):.4f}  ({time.time()-t0:.0f}s)")

# SVR
print("\n[3/6] SVR ...")
t0 = time.time()
def svr_obj(trial):
    p = dict(nu=trial.suggest_float("nu", 0.01, 0.99),
             C=trial.suggest_float("C", 0.1, 300.0, log=True),
             gamma=trial.suggest_float("gamma", 1e-4, 10.0, log=True))
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", NuSVR(kernel="rbf", **p))]),
                           X_tr_raw, y_tr, cv=TUNE_KF, scoring="r2").mean()
st = optuna.create_study(direction="maximize"); st.optimize(svr_obj, n_trials=N_TRIALS)
bp = {**st.best_params, "kernel": "rbf"}
pipe = Pipeline([("qt", make_qt()), ("m", NuSVR(**bp))]); pipe.fit(X_tr_raw, y_tr)
test_preds["SVR"] = pipe.predict(X_te_raw)
print(f"  MAE={mean_absolute_error(y_te, test_preds['SVR']):.4f}  R2={r2_score(y_te, test_preds['SVR']):.4f}  ({time.time()-t0:.0f}s)")

# ExtraTrees
print("\n[4/6] ExtraTrees ...")
t0 = time.time()
def et_obj(trial):
    p = dict(n_estimators=trial.suggest_int("n", 100, 600, step=50),
             max_depth=trial.suggest_categorical("d", [None, 5, 10, 15, 20]),
             min_samples_split=trial.suggest_int("mss", 2, 15),
             min_samples_leaf=trial.suggest_int("msl", 1, 8),
             max_features=trial.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
             random_state=42, n_jobs=-1)
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", ExtraTreesRegressor(**p))]),
                           X_tr_raw, y_tr, cv=TUNE_KF, scoring="r2").mean()
st = optuna.create_study(direction="maximize"); st.optimize(et_obj, n_trials=N_TRIALS)
_ET = {"n": "n_estimators", "d": "max_depth", "mss": "min_samples_split", "msl": "min_samples_leaf", "mf": "max_features"}
bp = {_ET.get(k, k): v for k, v in st.best_params.items()}
bp.update({"random_state": 42, "n_jobs": -1})
pipe = Pipeline([("qt", make_qt()), ("m", ExtraTreesRegressor(**bp))]); pipe.fit(X_tr_raw, y_tr)
test_preds["ExtraTrees"] = pipe.predict(X_te_raw)
print(f"  MAE={mean_absolute_error(y_te, test_preds['ExtraTrees']):.4f}  R2={r2_score(y_te, test_preds['ExtraTrees']):.4f}  ({time.time()-t0:.0f}s)")

# HGB
print("\n[5/6] HGB ...")
t0 = time.time()
def hgb_obj(trial):
    p = dict(max_iter=trial.suggest_int("mi", 100, 800, step=50),
             max_depth=trial.suggest_int("md", 2, 12),
             learning_rate=trial.suggest_float("lr", 0.005, 0.3, log=True),
             min_samples_leaf=trial.suggest_int("msl", 5, 40),
             l2_regularization=trial.suggest_float("l2", 1e-4, 10.0, log=True),
             max_leaf_nodes=trial.suggest_int("mln", 10, 80),
             random_state=42)
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", HistGradientBoostingRegressor(**p))]),
                           X_tr_raw, y_tr, cv=TUNE_KF, scoring="r2").mean()
st = optuna.create_study(direction="maximize"); st.optimize(hgb_obj, n_trials=N_TRIALS)
_HGB = {"mi": "max_iter", "md": "max_depth", "lr": "learning_rate", "msl": "min_samples_leaf",
        "l2": "l2_regularization", "mln": "max_leaf_nodes"}
bp = {_HGB.get(k, k): v for k, v in st.best_params.items()}
bp["random_state"] = 42
pipe = Pipeline([("qt", make_qt()), ("m", HistGradientBoostingRegressor(**bp))]); pipe.fit(X_tr_raw, y_tr)
test_preds["HGB"] = pipe.predict(X_te_raw)
print(f"  MAE={mean_absolute_error(y_te, test_preds['HGB']):.4f}  R2={r2_score(y_te, test_preds['HGB']):.4f}  ({time.time()-t0:.0f}s)")

# XGBoost
print("\n[6/6] XGBoost ...")
t0 = time.time()
def xgb_obj(trial):
    p = dict(n_estimators=trial.suggest_int("n", 100, 600, step=50),
             max_depth=trial.suggest_int("md", 2, 10),
             learning_rate=trial.suggest_float("lr", 0.005, 0.3, log=True),
             subsample=trial.suggest_float("ss", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("cbt", 0.5, 1.0),
             reg_alpha=trial.suggest_float("ra", 1e-4, 10.0, log=True),
             reg_lambda=trial.suggest_float("rl", 1e-4, 10.0, log=True),
             min_child_weight=trial.suggest_int("mcw", 1, 8),
             random_state=42, n_jobs=-1, verbosity=0)
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", XGBRegressor(**p))]),
                           X_tr_raw, y_tr, cv=TUNE_KF, scoring="r2").mean()
st = optuna.create_study(direction="maximize"); st.optimize(xgb_obj, n_trials=N_TRIALS)
_XGB = {"n": "n_estimators", "md": "max_depth", "lr": "learning_rate", "ss": "subsample",
        "cbt": "colsample_bytree", "ra": "reg_alpha", "rl": "reg_lambda", "mcw": "min_child_weight"}
bp = {_XGB.get(k, k): v for k, v in st.best_params.items()}
bp.update({"random_state": 42, "n_jobs": -1, "verbosity": 0})
pipe = Pipeline([("qt", make_qt()), ("m", XGBRegressor(**bp))]); pipe.fit(X_tr_raw, y_tr)
test_preds["XGBoost"] = pipe.predict(X_te_raw)
print(f"  MAE={mean_absolute_error(y_te, test_preds['XGBoost']):.4f}  R2={r2_score(y_te, test_preds['XGBoost']):.4f}  ({time.time()-t0:.0f}s)")

# Absolute errors
models = list(test_preds.keys())
ae     = {m: np.abs(y_te - test_preds[m]) for m in models}

print("\n" + SEP)
print("  Test-Set Performance Summary  (N=80)")
print(SEP)
perf_rows = []
for m in models:
    r2  = r2_score(y_te, test_preds[m])
    mae = ae[m].mean()
    perf_rows.append({"Model": m, "Test_R2": round(r2, 4), "Test_MAE": round(mae, 4),
                      "Median_AE": round(np.median(ae[m]), 4)})
    print(f"  {m:<15}  R2={r2:.4f}  MAE={mae:.4f}")
perf_df = pd.DataFrame(perf_rows).sort_values("Test_R2", ascending=False).reset_index(drop=True)

# Pairwise Wilcoxon + Holm-Bonferroni
print("\n" + SEP)
print("  Pairwise Wilcoxon (two-tailed) on |AE|  (N=80)  +  Holm correction  [6 models, 15 pairs]")
print(SEP)

pairs_raw = []
for m1, m2 in itertools.combinations(models, 2):
    d = ae[m1] - ae[m2]
    if np.all(d == 0):
        W, pval = np.nan, 1.0
    else:
        W, pval = wilcoxon(ae[m1], ae[m2], alternative="two-sided")
    n   = len(d)
    mu  = n * (n + 1) / 4
    sig = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z   = (W - mu) / sig if (sig > 0 and not np.isnan(W)) else 0.0
    r   = abs(z) / np.sqrt(n)
    winner = m1 if ae[m1].mean() < ae[m2].mean() else m2
    pairs_raw.append({"m1": m1, "m2": m2, "W": W, "p_raw": pval, "r": r, "winner": winner,
                      "mae1": ae[m1].mean(), "mae2": ae[m2].mean()})

k = len(pairs_raw)
pairs_raw.sort(key=lambda x: x["p_raw"])
p_adj = [min(1.0, row["p_raw"] * (k - i)) for i, row in enumerate(pairs_raw)]
for i in range(1, len(p_adj)):
    p_adj[i] = max(p_adj[i], p_adj[i - 1])

pairs_results = []
print(f"\n  {'Model A':<15} {'Model B':<15} {'W':>8} {'p_raw':>8} {'p_adj':>8} {'Sig':>4}  {'r':>6}  Winner")
print("  " + "-" * 82)
for row, pa in zip(pairs_raw, p_adj):
    row["p_adj"] = round(pa, 4)
    sig = "***" if pa < 0.001 else ("**" if pa < 0.01 else ("*" if pa < ALPHA else "ns"))
    row["sig"] = sig
    w_str = str(round(row["W"], 1)) if not np.isnan(row["W"]) else "n/a"
    print(f"  {row['m1']:<15} {row['m2']:<15} {w_str:>8} {row['p_raw']:>8.4f} {pa:>8.4f} {sig:>4}  {row['r']:>6.3f}  {row['winner']}")
    pairs_results.append(row)

pairs_df = pd.DataFrame(pairs_results)

# Heatmap
PLOT_HM = OUT_DIR / "wilcoxon_heatmap.png"
model_order = list(perf_df["Model"])
pm = pd.DataFrame(np.nan, index=model_order, columns=model_order)
for row in pairs_results:
    pm.loc[row["m1"], row["m2"]] = row["p_adj"]
    pm.loc[row["m2"], row["m1"]] = row["p_adj"]
pm_arr = pm.values.astype(float)

fig, ax = plt.subplots(figsize=(9, 7))
cax = ax.imshow(pm_arr, cmap="RdYlGn", vmin=0, vmax=0.2, aspect="auto")
plt.colorbar(cax, ax=ax, label="Holm-adjusted p-value", shrink=0.8)
for i in range(len(model_order)):
    for j in range(len(model_order)):
        if i == j:
            ax.text(j, i, "--", ha="center", va="center", fontsize=11, color="gray")
        else:
            v = pm_arr[i, j]
            if not np.isnan(v):
                s = "***" if v < 0.001 else ("**" if v < 0.01 else ("*" if v < 0.05 else ""))
                ax.text(j, i, f"{v:.3f}\n{s}", ha="center", va="center", fontsize=8,
                        color="black" if v > 0.05 else "darkblue",
                        fontweight="bold" if s else "normal")
ax.set_xticks(range(len(model_order))); ax.set_yticks(range(len(model_order)))
ax.set_xticklabels(model_order, rotation=35, ha="right", fontsize=10)
ax.set_yticklabels(model_order, fontsize=10)
ax.set_title("Pairwise Wilcoxon Signed-Rank Test — Holm-Adjusted p-value Matrix\n"
             "Criterion: Absolute Error per Test Sample (N=80)",
             fontsize=12, fontweight="bold", pad=12)
handles = [mpatches.Patch(color="darkgreen", label="p<0.001 (***)"),
           mpatches.Patch(color="yellowgreen", label="p<0.01 (**)"),
           mpatches.Patch(color="gold", label="p<0.05 (*)"),
           mpatches.Patch(color="red", label="p>=0.05 (ns)")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1), fontsize=9)
plt.tight_layout(); plt.savefig(PLOT_HM, dpi=180, bbox_inches="tight"); plt.close()
print(f"\n  Heatmap saved -> {PLOT_HM}")

# MAE bar chart
PLOT_MAE = OUT_DIR / "wilcoxon_mae_comparison.png"
colors_bar = ["#2ecc71" if m == "TabPFN" else "#3498db" for m in model_order[::-1]]
fig, ax = plt.subplots(figsize=(9, 6))
y_pos    = np.arange(len(model_order))
vals_bar = [ae[m].mean() for m in model_order[::-1]]
ax.barh(y_pos, vals_bar, color=colors_bar, edgecolor="white", height=0.6)
for i, v in enumerate(vals_bar):
    ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
ax.set_yticks(y_pos); ax.set_yticklabels(model_order[::-1], fontsize=10)
ax.set_xlabel("Mean Absolute Error (g)", fontsize=11)
ax.set_title("Test-Set Mean Absolute Error by Model\nFaba Bean: Single Plant Yield Prediction",
             fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.4)
handles2 = [mpatches.Patch(color="#2ecc71", label="TabPFN (reference)"),
            mpatches.Patch(color="#3498db", label="Baseline models")]
ax.legend(handles=handles2, fontsize=9)
plt.tight_layout(); plt.savefig(PLOT_MAE, dpi=180, bbox_inches="tight"); plt.close()
print(f"  MAE plot saved -> {PLOT_MAE}")

# Save CSV results
pairs_df.to_csv(OUT_DIR / "wilcoxon_results.csv", index=False)
perf_df.to_csv(OUT_DIR / "model_performance_summary.csv", index=False)
print(f"\n  Results saved: wilcoxon_results.csv, model_performance_summary.csv")
print(SEP)

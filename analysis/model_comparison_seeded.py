import warnings; warnings.filterwarnings("ignore")
import itertools
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle, Patch

from scipy.stats import (wilcoxon, friedmanchisquare, rankdata,
                         studentized_range, f as fdist)
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
# TabPFN is not tuned by Optuna. The grid {4, 8, 16, 32} was scanned (see
# TabPFN.py); n_estimators = 4 is the configuration reported in the manuscript
# and is fixed here so that this script reproduces the published TabPFN results
# exactly. Larger values change the results only marginally and cost more time.
TABPFN_NE = 4
ALPHA = 0.05
SEED = 42
KF = KFold(n_splits=5, shuffle=True, random_state=SEED)


def make_qt():
    return QuantileTransformer(output_distribution="normal", n_quantiles=200,
                               random_state=SEED)


def make_study():
    return optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))


def metrics(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)}


df = pd.read_csv(DATA, sep=";")
FEATS = [c for c in df.columns if c != TARGET]
X, y = df[FEATS].values, df[TARGET].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"train={len(y_tr)}  test={len(y_te)}  features={len(FEATS)}  device={device}")

test_pred, oof_pred, best_params, timing = {}, {}, {}, {}


# --------------------------------------------------------------- TabPFN ------
print(f"\n[1/6] TabPFN (n_estimators={TABPFN_NE}, not Optuna-tuned)")
oof = np.zeros(len(y_tr))
for tr_i, va_i in KF.split(X_tr):
    qt = make_qt()
    model = TabPFNRegressor(n_estimators=TABPFN_NE, device=device, random_state=SEED)
    model.fit(qt.fit_transform(X_tr[tr_i]), y_tr[tr_i])
    oof[va_i] = model.predict(qt.transform(X_tr[va_i]))
oof_pred["TabPFN"] = oof

qt = make_qt()
Xtr_s, Xte_s = qt.fit_transform(X_tr), qt.transform(X_te)
model = TabPFNRegressor(n_estimators=TABPFN_NE, device=device, random_state=SEED)
t0 = time.time(); model.fit(Xtr_s, y_tr); t_fit = time.time() - t0
t0 = time.time(); test_pred["TabPFN"] = model.predict(Xte_s); t_pred = time.time() - t0
best_params["TabPFN"] = {"n_estimators": TABPFN_NE}
timing["TabPFN"] = {"search_s": 0.0, "fit_s": round(t_fit, 3),
                    "predict_s": round(t_pred, 4)}
print(f"      test R2={r2_score(y_te, test_pred['TabPFN']):.4f}")


def cv_r2(model):
    """Mean 5-fold CV R2; the transformer is fitted inside each fold."""
    return cross_val_score(Pipeline([("qt", make_qt()), ("m", model)]),
                           X_tr, y_tr, cv=KF, scoring="r2").mean()


def tune_and_fit(name, objective, rename, ctor, fixed):
    print(f"\n[{name}]")
    t0 = time.time()
    study = make_study()
    study.optimize(objective, n_trials=N_TRIALS)
    t_search = time.time() - t0

    params = {rename.get(k, k): v for k, v in study.best_params.items()}
    params.update(fixed)
    best_params[name] = {k: v for k, v in params.items()
                         if k not in ("n_jobs", "verbosity", "random_state")}

    oof = np.zeros(len(y_tr))
    for tr_i, va_i in KF.split(X_tr):
        pipe = Pipeline([("qt", make_qt()), ("m", ctor(**params))])
        pipe.fit(X_tr[tr_i], y_tr[tr_i])
        oof[va_i] = pipe.predict(X_tr[va_i])
    oof_pred[name] = oof

    pipe = Pipeline([("qt", make_qt()), ("m", ctor(**params))])
    t0 = time.time(); pipe.fit(X_tr, y_tr); t_fit = time.time() - t0
    t0 = time.time(); test_pred[name] = pipe.predict(X_te); t_pred = time.time() - t0
    timing[name] = {"search_s": round(t_search, 2), "fit_s": round(t_fit, 3),
                    "predict_s": round(t_pred, 4)}
    print(f"      test R2={r2_score(y_te, test_pred[name]):.4f}  "
          f"(search {t_search:.0f}s)")


TREE = {"n": "n_estimators", "d": "max_depth", "mss": "min_samples_split",
        "msl": "min_samples_leaf", "mf": "max_features"}

tune_and_fit(
    "RandomForest",
    lambda t: cv_r2(RandomForestRegressor(
        n_estimators=t.suggest_int("n", 100, 600, step=50),
        max_depth=t.suggest_categorical("d", [None, 5, 10, 15, 20]),
        min_samples_split=t.suggest_int("mss", 2, 15),
        min_samples_leaf=t.suggest_int("msl", 1, 8),
        max_features=t.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
        random_state=SEED, n_jobs=-1)),
    TREE, RandomForestRegressor, {"random_state": SEED, "n_jobs": -1})

tune_and_fit(
    "SVR",
    lambda t: cv_r2(NuSVR(
        kernel="rbf",
        nu=t.suggest_float("nu", 0.01, 0.99),
        C=t.suggest_float("C", 0.1, 300.0, log=True),
        gamma=t.suggest_float("gamma", 1e-4, 10.0, log=True))),
    {}, NuSVR, {"kernel": "rbf"})

tune_and_fit(
    "ExtraTrees",
    lambda t: cv_r2(ExtraTreesRegressor(
        n_estimators=t.suggest_int("n", 100, 600, step=50),
        max_depth=t.suggest_categorical("d", [None, 5, 10, 15, 20]),
        min_samples_split=t.suggest_int("mss", 2, 15),
        min_samples_leaf=t.suggest_int("msl", 1, 8),
        max_features=t.suggest_categorical("mf", ["sqrt", "log2", 0.5, 0.8]),
        random_state=SEED, n_jobs=-1)),
    TREE, ExtraTreesRegressor, {"random_state": SEED, "n_jobs": -1})

tune_and_fit(
    "HGB",
    lambda t: cv_r2(HistGradientBoostingRegressor(
        max_iter=t.suggest_int("mi", 100, 800, step=50),
        max_depth=t.suggest_int("md", 2, 12),
        learning_rate=t.suggest_float("lr", 0.005, 0.3, log=True),
        min_samples_leaf=t.suggest_int("msl", 5, 40),
        l2_regularization=t.suggest_float("l2", 1e-4, 10.0, log=True),
        max_leaf_nodes=t.suggest_int("mln", 10, 80),
        random_state=SEED)),
    {"mi": "max_iter", "md": "max_depth", "lr": "learning_rate",
     "msl": "min_samples_leaf", "l2": "l2_regularization",
     "mln": "max_leaf_nodes"},
    HistGradientBoostingRegressor, {"random_state": SEED})

tune_and_fit(
    "XGBoost",
    lambda t: cv_r2(XGBRegressor(
        n_estimators=t.suggest_int("n", 100, 600, step=50),
        max_depth=t.suggest_int("md", 2, 10),
        learning_rate=t.suggest_float("lr", 0.005, 0.3, log=True),
        subsample=t.suggest_float("ss", 0.5, 1.0),
        colsample_bytree=t.suggest_float("cbt", 0.5, 1.0),
        reg_alpha=t.suggest_float("ra", 1e-4, 10.0, log=True),
        reg_lambda=t.suggest_float("rl", 1e-4, 10.0, log=True),
        min_child_weight=t.suggest_int("mcw", 1, 8),
        random_state=SEED, n_jobs=-1, verbosity=0)),
    {"n": "n_estimators", "md": "max_depth", "lr": "learning_rate",
     "ss": "subsample", "cbt": "colsample_bytree", "ra": "reg_alpha",
     "rl": "reg_lambda", "mcw": "min_child_weight"},
    XGBRegressor, {"random_state": SEED, "n_jobs": -1, "verbosity": 0})

# ------------------------------------------------- model comparison table ----
MODELS = ["TabPFN", "RandomForest", "SVR", "ExtraTrees", "HGB", "XGBoost"]
rows = []
for name in MODELS:
    per_fold = {"R2": [], "RMSE": [], "MAE": [], "MAPE": []}
    for _, va_i in KF.split(X_tr):
        fold = metrics(y_tr[va_i], oof_pred[name][va_i])
        for key in per_fold:
            per_fold[key].append(fold[key])
    te = metrics(y_te, test_pred[name])
    row = {"Model": name}
    for key, dec in (("R2", 4), ("RMSE", 4), ("MAE", 4), ("MAPE", 2)):
        row[f"CV_{key}"] = round(float(np.mean(per_fold[key])), dec)
        row[f"CV_{key}_std"] = round(float(np.std(per_fold[key])), dec)
    row.update({f"Test_{k}": round(v, 4 if k != "MAPE" else 2)
                for k, v in te.items()})
    rows.append(row)

comparison = pd.DataFrame(rows).sort_values("Test_R2", ascending=False)
comparison = comparison.reset_index(drop=True)
comparison.to_csv(OUT / "model_comparison.csv", index=False, sep=";")
print("\n=== Model comparison ===")
print(comparison.to_string(index=False))

pd.DataFrame([{"Model": k, "Best_parameters": str(v)}
              for k, v in best_params.items()]).to_csv(
    OUT / "best_hyperparameters.csv", index=False, sep=";")

ORDER = list(comparison["Model"])
AE = np.column_stack([np.abs(y_te - test_pred[m]) for m in ORDER])
N, k = AE.shape

predictions = pd.DataFrame({"y_true": y_te})
for i, name in enumerate(ORDER):
    predictions[f"pred_{name}"] = test_pred[name]
    predictions[f"AE_{name}"] = AE[:, i]
predictions.to_csv(OUT / "test_predictions_and_errors.csv", index=False, sep=";")

# ------------------------------------- pairwise Wilcoxon + Holm correction ---
pairs = []
for i, j in itertools.combinations(range(k), 2):
    W, p_raw = wilcoxon(AE[:, i], AE[:, j], alternative="two-sided")
    mu = N * (N + 1) / 4
    sd = np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    z = (W - mu) / sd
    pairs.append({"Model_1": ORDER[i], "Model_2": ORDER[j], "Wilcoxon_W": W,
                  "p_raw": p_raw, "effect_r": round(abs(z) / np.sqrt(N), 3),
                  "Lower_error": ORDER[i] if AE[:, i].mean() < AE[:, j].mean()
                                 else ORDER[j]})

wilcox = pd.DataFrame(pairs).sort_values("p_raw").reset_index(drop=True)
holm, running = np.empty(len(wilcox)), 0.0
for idx, p in enumerate(wilcox["p_raw"].values):
    running = max(running, (len(wilcox) - idx) * p)
    holm[idx] = min(running, 1.0)
wilcox["p_holm"] = np.round(holm, 4)
wilcox["Result"] = np.where(wilcox["p_holm"] < ALPHA, "Significant",
                            "Not significant")
wilcox["p_raw"] = wilcox["p_raw"].round(4)
wilcox.to_csv(OUT / "wilcoxon_holm.csv", index=False, sep=";")
print("\n=== Pairwise Wilcoxon (Holm-corrected) ===")
print(wilcox.to_string(index=False))

# --------------------------------------------- Friedman + Nemenyi post-hoc ---
ranks = np.vstack([rankdata(AE[i, :]) for i in range(N)])
mean_rank = ranks.mean(axis=0)
chi2, p_friedman = friedmanchisquare(*[AE[:, j] for j in range(k)])
F_id = ((N - 1) * chi2) / (N * (k - 1) - chi2)
p_id = fdist.sf(F_id, k - 1, (k - 1) * (N - 1))
kendall_w = chi2 / (N * (k - 1))

se = np.sqrt(k * (k + 1) / (6.0 * N))
CD = (studentized_range.ppf(1 - ALPHA, k, np.inf) / np.sqrt(2)) * se

print(f"\nFriedman chi2({k - 1}) = {chi2:.4f}, p = {p_friedman:.3e}")
print(f"Iman-Davenport F({k - 1}, {(k - 1) * (N - 1)}) = {F_id:.4f}, p = {p_id:.3e}")
print(f"Kendall's W = {kendall_w:.4f}   Nemenyi CD = {CD:.4f}")

nemenyi_rows = []
for i, j in itertools.combinations(range(k), 2):
    diff = abs(mean_rank[i] - mean_rank[j])
    p = min(float(studentized_range.sf((diff / se) * np.sqrt(2), k, np.inf)), 1.0)
    nemenyi_rows.append({"Model_1": ORDER[i], "Model_2": ORDER[j],
                         "Mean_rank_1": round(mean_rank[i], 3),
                         "Mean_rank_2": round(mean_rank[j], 3),
                         "Rank_difference": round(diff, 3),
                         "Exceeds_CD": "Yes" if diff > CD else "No",
                         "Nemenyi_p": round(p, 4),
                         "Result": "Significant" if p < ALPHA
                                   else "Not significant"})

nemenyi = pd.DataFrame(nemenyi_rows).sort_values("Nemenyi_p").reset_index(drop=True)
nemenyi.to_csv(OUT / "nemenyi_posthoc.csv", index=False, sep=";")
pd.DataFrame({"Model": ORDER, "Mean_rank": np.round(mean_rank, 3)}).to_csv(
    OUT / "nemenyi_mean_ranks.csv", index=False, sep=";")
pd.DataFrame([
    {"Statistic": "Friedman chi2", "Value": round(chi2, 4)},
    {"Statistic": "df", "Value": k - 1},
    {"Statistic": "Friedman p", "Value": f"{p_friedman:.3e}"},
    {"Statistic": "Iman-Davenport F", "Value": round(F_id, 4)},
    {"Statistic": "Iman-Davenport p", "Value": f"{p_id:.3e}"},
    {"Statistic": "Kendall W", "Value": round(kendall_w, 4)},
    {"Statistic": "Nemenyi CD (alpha=0.05)", "Value": round(CD, 4)},
    {"Statistic": "N (test samples)", "Value": N},
    {"Statistic": "k (models)", "Value": k},
]).to_csv(OUT / "friedman_summary.csv", index=False, sep=";")
print("\n=== Nemenyi post-hoc ===")
print(nemenyi.to_string(index=False))

# ---------------------------------- figure: Holm-adjusted p-value matrix -----
# The colour map is cividis rather than a red-green ramp, so the figure stays
# readable for readers with red-green colour-vision deficiency; significance is
# additionally encoded by a black cell border and by significance stars, so the
# figure can also be read in greyscale.
P = np.full((k, k), np.nan)
lookup = {}
for _, r in wilcox.iterrows():
    lookup[(r.Model_1, r.Model_2)] = r.p_holm
    lookup[(r.Model_2, r.Model_1)] = r.p_holm
for i in range(k):
    for j in range(k):
        if i != j:
            P[i, j] = lookup[(ORDER[i], ORDER[j])]

BOUNDS = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
cmap = plt.get_cmap("cividis", len(BOUNDS) - 1)
norm = BoundaryNorm(BOUNDS, cmap.N)


def text_colour(cls):
    """White on dark cells, black on light cells, from WCAG relative luminance."""
    r, g, b, _ = cmap(cls)

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "white" if lum < 0.4 else "black"


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


fig, ax = plt.subplots(figsize=(11.5, 9.0))
ax.imshow(np.ma.masked_invalid(P), cmap=cmap, norm=norm, aspect="equal")
for i in range(k):
    for j in range(k):
        if i == j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="0.90",
                                   edgecolor="white", lw=2))
            ax.text(j, i, "—", ha="center", va="center", fontsize=18,
                    color="0.35", fontweight="bold")
            continue
        p = P[i, j]
        cls = min(max(int(np.digitize(p, BOUNDS) - 1), 0), len(BOUNDS) - 2)
        colour = text_colour(cls)
        ax.text(j, i - .16, "<0.001" if p < 0.001 else f"{p:.3f}", ha="center",
                va="center", fontsize=15, fontweight="bold", color=colour)
        ax.text(j, i + .22, stars(p), ha="center", va="center", fontsize=14,
                fontweight="bold", color=colour)
        if p < ALPHA:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor="black", lw=3.5, zorder=5))

ax.set_xticks(range(k)); ax.set_yticks(range(k))
ax.set_xticklabels(ORDER, rotation=30, ha="right", fontsize=15)
ax.set_yticklabels(ORDER, fontsize=15)
ax.set_xticks(np.arange(-.5, k, 1), minor=True)
ax.set_yticks(np.arange(-.5, k, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0, pad=6)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Pairwise Wilcoxon signed-rank test – Holm-adjusted p-value matrix\n"
             f"Criterion: absolute error per test sample (N = {N})",
             fontsize=17, fontweight="bold", pad=16)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    boundaries=BOUNDS, ticks=BOUNDS, spacing="uniform",
                    fraction=0.046, pad=0.03)
cbar.set_label("Holm-adjusted p-value", fontsize=15, labelpad=10)
cbar.ax.set_yticklabels(["0", "0.001", "0.01", "0.05", "0.10", "0.25", "0.50",
                         "1.00"])
cbar.ax.tick_params(labelsize=13)
ax.legend(handles=[
    Patch(facecolor="white", edgecolor="black", lw=3.5,
          label="Significant after Holm correction (p < 0.05)"),
    Patch(facecolor="white", edgecolor="0.75",
          label="*** p < 0.001   ** p < 0.01   * p < 0.05   ns = not significant")],
    loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=13,
    frameon=False, handlelength=1.6)
fig.savefig(OUT / "wilcoxon_holm_matrix.png", dpi=600, bbox_inches="tight",
            facecolor="white")
fig.savefig(OUT / "wilcoxon_holm_matrix.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ------------------------------- figure: Nemenyi critical-difference plot ----
order_idx = np.argsort(mean_rank)
names_sorted = [ORDER[i] for i in order_idx]
ranks_sorted = mean_rank[order_idx]

lo, hi = 1.0, float(k)
fig, ax = plt.subplots(figsize=(11.5, 4.2))
ax.set_xlim(lo - 1.15, hi + 1.15)
ax.set_ylim(0.85, 3.20)
ax.axis("off")
ax.hlines(2.60, lo, hi, color="black", lw=2.2)
for t in np.arange(lo, hi + 1e-9, 0.5):
    ax.vlines(t, 2.60, 2.74, color="black", lw=2.2)
    ax.text(t, 2.82, f"{t:g}", ha="center", va="bottom", fontsize=14)
ax.text((lo + hi) / 2, 3.02, "Mean rank (lower = lower prediction error)",
        ha="center", va="bottom", fontsize=15, fontweight="bold")

half = (k + 1) // 2
for idx, (name, rank) in enumerate(zip(names_sorted, ranks_sorted)):
    left = idx < half
    step = idx if left else (k - 1 - idx)
    level = 2.25 - 0.30 * step
    x_end = lo - 0.30 if left else hi + 0.30
    ax.vlines(rank, level, 2.60, color="black", lw=1.8)
    ax.hlines(level, min(rank, x_end), max(rank, x_end), color="black", lw=1.8)
    ax.text(x_end + (-0.08 if left else 0.08), level, f"{name} ({rank:.2f})",
            ha="right" if left else "left", va="center", fontsize=14,
            fontweight="bold" if name == "TabPFN" else "normal")

cliques = []
for i in range(k):
    j = i
    while j + 1 < k and (ranks_sorted[j + 1] - ranks_sorted[i]) <= CD:
        j += 1
    if j > i:
        cliques.append((i, j))
cliques = [(a, b) for a, b in cliques
           if not any((a2 <= a and b <= b2) and (a2, b2) != (a, b)
                      for a2, b2 in cliques)]
for n, (a, b) in enumerate(cliques):
    ax.hlines(2.46 - 0.10 * n, ranks_sorted[a] - 0.035, ranks_sorted[b] + 0.035,
              color="0.20", lw=5.5)

ax.hlines(1.18, lo, lo + CD, color="black", lw=3.2)
ax.vlines([lo, lo + CD], 1.10, 1.26, color="black", lw=3.2)
ax.text(lo + CD / 2, 1.02, f"CD = {CD:.3f}", ha="center", va="top", fontsize=14,
        fontweight="bold")
ax.text(lo + CD + 0.25, 1.18,
        "Models connected by a bar are not\nsignificantly different "
        "(Nemenyi, α = 0.05)", ha="left", va="center", fontsize=13)
ax.set_title("Nemenyi post-hoc comparison of the six regression models\n"
             f"Friedman χ²({k - 1}) = {chi2:.2f}, p = {p_friedman:.1e}"
             f"  |  {N} paired test observations",
             fontsize=16, fontweight="bold", pad=34)
fig.savefig(OUT / "nemenyi_critical_difference.png", dpi=600,
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "nemenyi_critical_difference.pdf", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ------------------------------------------------ runtime and environment ----
runtime = pd.DataFrame(timing).T.reset_index().rename(columns={"index": "Model"})
runtime.to_csv(OUT / "runtime_profile.csv", index=False, sep=";")
print("\n=== Runtime (s) ===")
print(runtime.to_string(index=False))

import sklearn, scipy, xgboost, tabpfn
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
env = "\n".join([
    f"OS                : {platform.system()} {platform.release()}",
    f"CPU logical cores : {__import__('os').cpu_count()}",
    f"GPU               : {gpu}",
    f"Python            : {platform.python_version()}",
    f"torch             : {torch.__version__}",
    f"tabpfn            : {getattr(tabpfn, '__version__', 'unknown')}",
    f"scikit-learn      : {sklearn.__version__}",
    f"scipy             : {scipy.__version__}",
    f"xgboost           : {xgboost.__version__}",
    f"optuna            : {optuna.__version__}",
    f"numpy             : {np.__version__}",
    f"pandas            : {pd.__version__}",
])
(OUT / "environment.txt").write_text(env + "\n", encoding="utf-8")
print("\n" + env)
print(f"\nAll outputs written to: {OUT}")

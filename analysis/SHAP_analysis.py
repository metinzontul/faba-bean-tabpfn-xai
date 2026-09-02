import warnings; warnings.filterwarnings('ignore')
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

DATA_PATH = Path(__file__).parent.parent / "data" / "faba_bean_dataset.csv"
OUT_DIR   = Path(__file__).parent
TARGET    = 'Single_Plant_Yield_g'

FEATURE_MAP = {
    'Days_to_Flowering'       : 'Days to Flowering',
    'Days_to_Maturity'        : 'Days to Maturity',
    'First_Pod_Height_cm'     : 'First Pod Height (cm)',
    'Branch_Number'           : 'Branch Number',
    'Plant_Height_cm'         : 'Plant Height (cm)',
    'Biological_Yield_g'      : 'Biological Yield (g)',
    'Pod_Number'              : 'Pod Number per Plant',
    'Pod_Weight_g'            : 'Pod Weight (g)',
    'Pod_Length_cm'           : 'Pod Length (cm)',
    'Seeds_per_Pod'           : 'Seeds per Pod',
    'Seeds_per_Plant'         : 'Seeds per Plant',
    'Plot_Yield_kg_da'        : 'Plot Yield (kg/da)',
    'Seed_Weight_100_g'       : '100-Seed Weight (g)',
    'Pod_Filling_Rate'        : 'Pod Filling Rate',
    'Seed_to_Pod_Ratio'       : 'Seed-to-Pod Ratio',
    'Seed_Unit_Weight'        : 'Seed Unit Weight',
    'Pod_Length_Yield_Index'  : 'Pod Length-Yield Index',
    'Plant_Yield_Ratio'       : 'Plant Yield Ratio',
    'Vegetation_Period_days'  : 'Vegetation Period (days)',
}

# TabPFN performance, printed in the figure caption only. These values are
# produced by analysis/model_comparison_seeded.py (results/model_comparison.csv).
# TabPFN is deterministic, so they are stable across runs.
MODEL_PERF = {
    'CV R²'    : '0.8660',
    'CV Std'   : '±0.0803',
    'CV RMSE'  : '1.9350',
    'CV MAE'   : '1.0528',
    'CV MAPE'  : '8.35%',
    'Test R²'  : '0.8746',
    'Test RMSE': '1.9132',
    'Test MAE' : '1.0819',
    'Test MAPE': '8.16%',
}

print("=" * 60)
print("  SHAP-Based Explainability Analysis — TabPFN")
print("=" * 60)

df     = pd.read_csv(DATA_PATH, sep=';')
FEATS  = [c for c in df.columns if c != TARGET]
X_raw  = df[FEATS].values
y      = df[TARGET].values
feat_en = [FEATURE_MAP.get(f, f) for f in FEATS]

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.20, random_state=42)

sc      = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
X_tr_sc = sc.fit_transform(X_tr)
X_te_sc = sc.transform(X_te)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = TabPFNRegressor(n_estimators=4, device=device, random_state=42)
model.fit(X_tr_sc, y_tr)
print(f"[1/4] TabPFN trained on {len(y_tr)} samples (device: {device})")

np.random.seed(42)
bg_idx    = np.random.choice(len(X_tr_sc), size=min(50, len(X_tr_sc)), replace=False)
X_bg      = X_tr_sc[bg_idx]
masker    = shap.maskers.Independent(X_bg, max_samples=50)
explainer = shap.PermutationExplainer(model.predict, masker)
shap_obj  = explainer(X_te_sc, max_evals=2 * 19 + 1)
shap_values = shap_obj.values
print(f"[2/4] SHAP values computed  (test set: {len(y_te)} samples)")

mean_abs = np.abs(shap_values).mean(axis=0)
shap_df  = pd.DataFrame({'Feature': feat_en, 'Mean_AbsSHAP': mean_abs})
shap_df  = shap_df.sort_values('Mean_AbsSHAP', ascending=False).reset_index(drop=True)
shap_df['Rank']                      = range(1, len(shap_df) + 1)
shap_df['Relative Contribution (%)'] = (shap_df['Mean_AbsSHAP'] / shap_df['Mean_AbsSHAP'].sum() * 100)
shap_df['Cumulative (%)']            = shap_df['Relative Contribution (%)'].cumsum()

feat_order  = [feat_en.index(f) for f in shap_df['Feature']]
shap_sorted = shap_values[:, feat_order]
Xte_sorted  = X_te_sc[:, feat_order]
feat_sorted = list(shap_df['Feature'])

print("\n  SHAP Importance Ranking:")
print(f"  {'Rank':>4}  {'Feature':<32}  {'Mean|SHAP|':>10}  {'Rel.Contr.':>10}  {'Cumul.':>8}")
print("  " + "-" * 72)
for _, row in shap_df.iterrows():
    print(f"  {int(row['Rank']):>4}  {row['Feature']:<32}  "
          f"{row['Mean_AbsSHAP']:>10.4f}  "
          f"{row['Relative Contribution (%)']:>9.2f}%  "
          f"{row['Cumulative (%)']:>7.2f}%")

# SHAP summary beeswarm plot
PLOT_SUMMARY = OUT_DIR / "shap_summary_plot.png"
fig, ax = plt.subplots(figsize=(11, 8))
shap.summary_plot(
    shap_sorted, Xte_sorted,
    feature_names=feat_sorted,
    show=False,
    plot_type='dot',
    color_bar_label='Feature Value',
    max_display=19,
    plot_size=None,
)
plt.title(
    "SHAP Summary Plot — TabPFN\nFaba Bean: Single Plant Yield Prediction",
    fontsize=13, fontweight='bold', pad=12
)
plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=11)
plt.tight_layout()
plt.savefig(PLOT_SUMMARY, dpi=180, bbox_inches='tight')
plt.close()
print(f"[3/4] Summary plot saved -> {PLOT_SUMMARY}")

# SHAP feature importance bar chart
PLOT_IMPORTANCE = OUT_DIR / "shap_importance_plot.png"
n_feat = len(shap_df)
cmap   = plt.cm.get_cmap('RdYlGn_r', n_feat)
colors = [cmap(i / (n_feat - 1)) for i in range(n_feat)]

fig, ax = plt.subplots(figsize=(11, 8))
y_pos   = np.arange(n_feat)
vals    = shap_df['Mean_AbsSHAP'].values[::-1]
labels  = shap_df['Feature'].values[::-1]
bar_col = colors[::-1]

bars = ax.barh(y_pos, vals, color=bar_col, edgecolor='white', height=0.7)
for bar, val in zip(bars, vals):
    ax.text(
        bar.get_width() + max(vals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.4f}', va='center', ha='left', fontsize=9, color='#333333'
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title(
    'SHAP Feature Importance — TabPFN\nFaba Bean: Single Plant Yield Prediction',
    fontsize=13, fontweight='bold'
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, max(vals) * 1.18)
ax.grid(axis='x', linestyle='--', alpha=0.4)

high_patch = mpatches.Patch(color=cmap(0.05), label='High importance')
low_patch  = mpatches.Patch(color=cmap(0.95), label='Low importance')
ax.legend(handles=[high_patch, low_patch], loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(PLOT_IMPORTANCE, dpi=180, bbox_inches='tight')
plt.close()
print(f"[4/4] Importance plot saved -> {PLOT_IMPORTANCE}")

shap_csv = OUT_DIR / "shap_importance_ranking.csv"
shap_df.to_csv(shap_csv, index=False)
print(f"\n  SHAP ranking saved -> {shap_csv}")
print("=" * 60)

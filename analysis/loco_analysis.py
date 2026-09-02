import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import QuantileTransformer

DATA_PATH = Path(__file__).parent.parent / "data" / "faba_bean_dataset.csv"
OUT_DIR   = Path(__file__).parent
TARGET    = 'Single_Plant_Yield_g'
SEED      = 42

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
    'Plot_Yield_kg_da'        : 'Plot Yield (kg da⁻¹)',
    'Seed_Weight_100_g'       : '100-Seed Weight (g)',
    'Pod_Filling_Rate'        : 'Pod Filling Rate',
    'Seed_to_Pod_Ratio'       : 'Seed-to-Pod Ratio',
    'Seed_Unit_Weight'        : 'Seed Unit Weight',
    'Pod_Length_Yield_Index'  : 'Pod Length-Yield Index',
    'Plant_Yield_Ratio'       : 'Plant Yield Ratio',
    'Vegetation_Period_days'  : 'Vegetation Period (days)',
}

def interpret(drop):
    if drop >= 0.005:
        return 'Important'
    if drop >= 0.0:
        return 'Minor'
    return 'Redundant / Compensable'

print("=" * 60)
print("  LOCO Feature Contribution Analysis — TabPFN")
print("=" * 60)

df    = pd.read_csv(DATA_PATH, sep=';')
FEATS = [c for c in df.columns if c != TARGET]
X     = df[FEATS].values
y     = df[TARGET].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)

sc   = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=SEED)
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Baseline model — all 19 features
baseline_model = TabPFNRegressor(n_estimators=4, device=device, random_state=SEED)
baseline_model.fit(X_tr, y_tr)
baseline_r2   = r2_score(y_te, baseline_model.predict(X_te))
baseline_rmse = np.sqrt(mean_squared_error(y_te, baseline_model.predict(X_te)))
print(f"[1/3] Baseline trained  |  Test R² = {baseline_r2:.4f}  RMSE = {baseline_rmse:.4f}  (device: {device})")
print(f"      Running LOCO over {len(FEATS)} features ...")

rows = []
for i, feat in enumerate(FEATS):
    cols   = [j for j in range(len(FEATS)) if j != i]
    m      = TabPFNRegressor(n_estimators=4, device=device, random_state=SEED)
    m.fit(X_tr[:, cols], y_tr)
    preds  = m.predict(X_te[:, cols])
    r2_wo  = r2_score(y_te, preds)
    rmse_wo = np.sqrt(mean_squared_error(y_te, preds))
    rows.append({
        'Feature_raw' : feat,
        'Feature'     : FEATURE_MAP.get(feat, feat),
        'R2_Without'  : round(r2_wo,   4),
        'RMSE_Without': round(rmse_wo,  4),
        'R2_Drop'     : round(baseline_r2 - r2_wo, 4),
    })
    print(f"  [{i+1:>2}/{len(FEATS)}] {FEATURE_MAP.get(feat, feat):<32}  "
          f"R²_without={r2_wo:.4f}  R²_drop={baseline_r2 - r2_wo:+.4f}")

loco_df = (pd.DataFrame(rows)
             .sort_values('R2_Drop', ascending=False)
             .reset_index(drop=True))
loco_df['Rank']           = range(1, len(loco_df) + 1)
loco_df['Interpretation'] = loco_df['R2_Drop'].apply(interpret)

print(f"\n[2/3] LOCO complete  |  Baseline R² = {baseline_r2:.4f}\n")
print(f"  {'Rank':>4}  {'Feature':<32}  {'R²_Without':>10}  {'RMSE_Without':>12}  {'R²_Drop':>8}  Interpretation")
print("  " + "-" * 84)
for _, row in loco_df.iterrows():
    print(f"  {int(row['Rank']):>4}  {row['Feature']:<32}  "
          f"{row['R2_Without']:>10.4f}  "
          f"{row['RMSE_Without']:>12.4f}  "
          f"{row['R2_Drop']:>+8.4f}  "
          f"{row['Interpretation']}")

# Bar chart
PLOT_PATH = OUT_DIR / "loco_importance_plot.png"
l_sorted  = loco_df.sort_values('R2_Drop')
colors    = ['#d73027' if v > 0 else '#4575b4' for v in l_sorted['R2_Drop']]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    l_sorted['Feature'], l_sorted['R2_Drop'],
    color=colors, edgecolor='white', height=0.7,
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('R² Change  (Baseline − R²_without)', fontsize=12)
ax.set_title(
    f'LOCO Feature Contribution — TabPFN\n'
    f'Faba Bean: Single Plant Yield Prediction  (Baseline R² = {baseline_r2:.4f})',
    fontsize=13, fontweight='bold',
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)

from matplotlib.patches import Patch
legend = [Patch(color='#d73027', label='Positive contribution (R² drops without)'),
          Patch(color='#4575b4', label='Redundant / noisy (R² improves without)')]
ax.legend(handles=legend, fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=180, bbox_inches='tight')
plt.close()

CSV_PATH = OUT_DIR / "loco_importance_ranking.csv"
loco_df.drop(columns='Feature_raw').to_csv(CSV_PATH, index=False)

print(f"[3/3] Plot  → {PLOT_PATH}")
print(f"      CSV   → {CSV_PATH}")
print("=" * 60)

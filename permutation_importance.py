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
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer

DATA_PATH  = Path(__file__).parent.parent / "data" / "faba_bean_dataset.csv"
OUT_DIR    = Path(__file__).parent
TARGET     = 'Single_Plant_Yield_g'
N_REPEATS  = 50
SEED       = 42

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

INTERP_RULES = [
    (0.05,  'Highly Important'),
    (0.01,  'Important'),
    (0.001, 'Minor Contribution'),
    (-999,  'Negligible / Noisy'),
]

def interpret(val):
    for threshold, label in INTERP_RULES:
        if val >= threshold:
            return label
    return 'Negligible / Noisy'

print("=" * 60)
print("  Permutation Importance Analysis — TabPFN")
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
model  = TabPFNRegressor(n_estimators=4, device=device, random_state=SEED)
model.fit(X_tr, y_tr)

baseline_r2 = r2_score(y_te, model.predict(X_te))
print(f"[1/3] TabPFN trained  |  Baseline Test R² = {baseline_r2:.4f}  (device: {device})")

rng        = np.random.default_rng(SEED)
imp_scores = np.zeros((len(FEATS), N_REPEATS))

for i, feat in enumerate(FEATS):
    for r in range(N_REPEATS):
        X_perm        = X_te.copy()
        X_perm[:, i]  = rng.permutation(X_perm[:, i])
        imp_scores[i, r] = baseline_r2 - r2_score(y_te, model.predict(X_perm))

feat_en = [FEATURE_MAP.get(f, f) for f in FEATS]
perm_df = pd.DataFrame({
    'Feature_raw'    : FEATS,
    'Feature'        : feat_en,
    'Mean_R2_Drop'   : imp_scores.mean(axis=1),
    'Std_R2_Drop'    : imp_scores.std(axis=1),
}).sort_values('Mean_R2_Drop', ascending=False).reset_index(drop=True)

perm_df['Rank']           = range(1, len(perm_df) + 1)
perm_df['Interpretation'] = perm_df['Mean_R2_Drop'].apply(interpret)

print(f"[2/3] Permutation done  ({N_REPEATS} repeats per feature)")

print(f"\n  {'Rank':>4}  {'Feature':<32}  {'Mean R² Drop':>12}  {'Std':>8}  Interpretation")
print("  " + "-" * 78)
for _, row in perm_df.iterrows():
    print(f"  {int(row['Rank']):>4}  {row['Feature']:<32}  "
          f"{row['Mean_R2_Drop']:>+12.4f}  "
          f"±{row['Std_R2_Drop']:>6.4f}  "
          f"{row['Interpretation']}")

# Bar chart
PLOT_PATH = OUT_DIR / "permutation_importance_plot.png"
p_sorted  = perm_df.sort_values('Mean_R2_Drop')
colors    = ['#d73027' if v > 0 else '#4575b4' for v in p_sorted['Mean_R2_Drop']]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    p_sorted['Feature'], p_sorted['Mean_R2_Drop'],
    xerr=p_sorted['Std_R2_Drop'], capsize=3,
    color=colors, edgecolor='white', height=0.7,
)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean R² Drop', fontsize=12)
ax.set_title(
    f'Permutation Importance — TabPFN\n'
    f'Faba Bean: Single Plant Yield Prediction  ({N_REPEATS} repeats)',
    fontsize=13, fontweight='bold',
)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=180, bbox_inches='tight')
plt.close()

CSV_PATH = OUT_DIR / "permutation_importance_ranking.csv"
perm_df.drop(columns='Feature_raw').to_csv(CSV_PATH, index=False)

print(f"[3/3] Plot  → {PLOT_PATH}")
print(f"      CSV   → {CSV_PATH}")
print("=" * 60)

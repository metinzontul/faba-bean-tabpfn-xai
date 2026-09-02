import warnings; warnings.filterwarnings("ignore")
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "faba_bean_dataset.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

TARGET = "Single_Plant_Yield_g"
SEED = 42
R_THRESHOLD = 0.70

df = pd.read_csv(DATA, sep=";")
FEATS = [c for c in df.columns if c != TARGET]
X_tr, _, _, _ = train_test_split(df[FEATS].values, df[TARGET].values,
                                 test_size=0.20, random_state=SEED)
train = pd.DataFrame(X_tr, columns=FEATS)
print(f"training rows used for the diagnostics: {train.shape}")

corr = train.corr(method="pearson")
corr.round(4).to_csv(OUT / "feature_correlation_matrix.csv", sep=";")

high = [{"Feature_1": a, "Feature_2": b, "Pearson_r": round(float(corr.loc[a, b]), 4)}
        for a, b in itertools.combinations(FEATS, 2)
        if abs(corr.loc[a, b]) >= R_THRESHOLD]
high = pd.DataFrame(high)
high = high.reindex(high.Pearson_r.abs().sort_values(ascending=False).index)
high = high.reset_index(drop=True)
high.to_csv(OUT / "feature_pairs_high_correlation.csv", index=False, sep=";")
print(f"\nPairs with |r| >= {R_THRESHOLD}:")
print(high.to_string(index=False))

# ---- variance inflation factors ---------------------------------------------
vif_rows = []
for col in FEATS:
    others = [c for c in FEATS if c != col]
    r2 = LinearRegression().fit(train[others], train[col]).score(train[others],
                                                                train[col])
    vif = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    vif_rows.append({"Feature": col, "R2_on_other_features": round(float(r2), 4),
                     "VIF": round(float(vif), 2)})

vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif = vif.reset_index(drop=True)
vif.to_csv(OUT / "feature_vif.csv", index=False, sep=";")
print("\nVariance inflation factors:")
print(vif.to_string(index=False))
print("\nNote: VIF = inf indicates an exact linear dependence. Vegetation period "
      "is defined as days to maturity minus days to flowering, so those three "
      "predictors are perfectly collinear and their individual contributions "
      "cannot be separated by any importance method.")

# ---- hierarchical clustering of 1 - |r| -------------------------------------
distance = 1.0 - corr.abs().values
np.fill_diagonal(distance, 0.0)
distance = (distance + distance.T) / 2.0
linkage_matrix = linkage(squareform(distance, checks=False), method="average")
labels = fcluster(linkage_matrix, t=1.0 - R_THRESHOLD, criterion="distance")

clusters = pd.DataFrame({"Feature": FEATS, "Cluster": labels})
clusters = clusters.sort_values(["Cluster", "Feature"]).reset_index(drop=True)
clusters.to_csv(OUT / "feature_clusters.csv", index=False, sep=";")
print(f"\nGroups of traits mutually correlated at |r| >= {R_THRESHOLD}:")
for cluster_id in sorted(set(labels)):
    members = clusters.loc[clusters.Cluster == cluster_id, "Feature"].tolist()
    if len(members) > 1:
        print(f"  cluster {cluster_id}: {', '.join(members)}")

print(f"\nAll outputs written to: {OUT}")

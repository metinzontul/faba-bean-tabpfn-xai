import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA   = Path(__file__).parent / "data" / "faba_bean_dataset.csv"
TARGET = 'Single_Plant_Yield_g'

df    = pd.read_csv(DATA, sep=';')
FEATS = [c for c in df.columns if c != TARGET]

X = df[FEATS].values
y = df[TARGET].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

sc   = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = dict(
        max_iter          = trial.suggest_int('max_iter', 100, 1000, step=50),
        max_depth         = trial.suggest_int('max_depth', 2, 15),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 5, 50),
        l2_regularization = trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        max_leaf_nodes    = trial.suggest_int('max_leaf_nodes', 10, 100),
        random_state=42,
    )
    oof = cross_val_predict(HistGradientBoostingRegressor(**params), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

print("Optuna tuning (100 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_p = study.best_params
best_p.update({'random_state': 42})
print(f"Best Optuna OOF R² = {study.best_value:.4f}")

model = HistGradientBoostingRegressor(**best_p)
oof = cross_val_predict(model, X_tr, y_tr, cv=kf)
cv_r2s, cv_rmses, cv_maes, cv_mapes, cv_nrmses = [], [], [], [], []
for _, val_idx in kf.split(X_tr):
    yv, pv = y_tr[val_idx], oof[val_idx]
    cv_r2s.append(r2_score(yv, pv))
    cv_rmses.append(np.sqrt(mean_squared_error(yv, pv)))
    cv_maes.append(mean_absolute_error(yv, pv))
    cv_mapes.append(np.mean(np.abs((yv - pv) / (np.abs(yv) + 1e-8))) * 100)
    cv_nrmses.append(np.sqrt(mean_squared_error(yv, pv)) / np.mean(yv))

model.fit(X_tr, y_tr)
pred    = model.predict(X_te)
test_r2 = r2_score(y_te, pred)
rmse    = np.sqrt(mean_squared_error(y_te, pred))
mae     = mean_absolute_error(y_te, pred)
mape    = np.mean(np.abs((y_te - pred) / (np.abs(y_te) + 1e-8))) * 100
nrmse   = rmse / np.mean(y_te)

SEP = "=" * 55
print(SEP)
print("  FINAL MODEL — HistGradientBoosting (Faba Bean, 19 Features)")
print(SEP)
print(f"  Feature count  : {len(FEATS)}")
print(f"  Train / Test   : {len(y_tr)} / {len(y_te)}")
print(SEP)
print(f"  CV R²    (Mean±Std) : {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")
print(f"  CV RMSE  (Mean±Std) : {np.mean(cv_rmses):.4f} ± {np.std(cv_rmses):.4f}")
print(f"  CV MAE   (Mean±Std) : {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")
print(f"  CV MAPE  (Mean±Std) : {np.mean(cv_mapes):.2f}% ± {np.std(cv_mapes):.2f}%")
print(f"  CV NRMSE (Mean±Std) : {np.mean(cv_nrmses):.4f} ± {np.std(cv_nrmses):.4f}")
print(SEP)
print(f"  Test R²             : {test_r2:.4f}")
print(f"  Test RMSE           : {rmse:.4f}")
print(f"  Test MAE            : {mae:.4f}")
print(f"  Test MAPE           : {mape:.2f}%")
print(f"  Test NRMSE          : {nrmse:.4f}")
print(SEP)
print("  Best parameters:")
for k, v in best_p.items():
    if k != 'random_state':
        print(f"    {k}: {v}")
print(SEP)

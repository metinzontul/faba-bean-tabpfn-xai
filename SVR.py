import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, optuna, time
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

print("NuSVR Optuna tuning (100 trials)...")
t0 = time.time()

def nusvr_obj(trial):
    p = dict(
        nu    = trial.suggest_float('nu',    0.01, 0.99),
        C     = trial.suggest_float('C',     0.1,  300.0, log=True),
        gamma = trial.suggest_float('gamma', 1e-4, 10.0,  log=True),
    )
    oof = cross_val_predict(NuSVR(kernel='rbf', **p), X_tr, y_tr, cv=kf, n_jobs=-1)
    return r2_score(y_tr, oof)

study = optuna.create_study(direction='maximize')
study.optimize(nusvr_obj, n_trials=100, show_progress_bar=True)

best_p = {**study.best_params, 'kernel': 'rbf'}
print(f"Best OOF CV R²: {study.best_value:.4f}")
print(f"Parameters: {best_p}")

model = NuSVR(**best_p)
oof = cross_val_predict(model, X_tr, y_tr, cv=kf, n_jobs=-1)
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
print("  FINAL MODEL — NuSVR (Faba Bean, 19 Features)")
print(SEP)
print(f"  Algorithm      : NuSVR (kernel=rbf)")
print(f"  Feature count  : {len(FEATS)}")
print(f"  Train / Test   : {len(y_tr)} / {len(y_te)}")
print(f"  Optuna trials  : 100")
print(f"  Time           : {time.time()-t0:.0f}s")
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

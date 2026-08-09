import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, optuna, time
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor

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

print("MLP Optuna tuning (100 trials)...")
t0 = time.time()

def mlp_obj(trial):
    n_layers = trial.suggest_int('n_layers', 1, 4)
    units    = trial.suggest_categorical('units', [32, 64, 128, 256])
    layers   = tuple([units] * n_layers)
    p = dict(
        hidden_layer_sizes  = layers,
        activation          = trial.suggest_categorical('activation', ['relu', 'tanh']),
        alpha               = trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        learning_rate_init  = trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        max_iter            = 500,
        random_state        = 42,
        early_stopping      = True,
        validation_fraction = 0.1,
    )
    scores = cross_val_score(MLPRegressor(**p), X_tr, y_tr, cv=kf, scoring='r2')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(mlp_obj, n_trials=100, show_progress_bar=True)

bp = study.best_params
n_layers    = bp['n_layers']
units       = bp['units']
best_layers = tuple([units] * n_layers)

best_p = dict(
    hidden_layer_sizes  = best_layers,
    activation          = bp['activation'],
    alpha               = bp['alpha'],
    learning_rate_init  = bp['lr'],
    max_iter            = 1000,
    random_state        = 42,
    early_stopping      = True,
    validation_fraction = 0.1,
)
print(f"Best OOF CV R²: {study.best_value:.4f}")
print(f"Parameters: layers={best_layers}, activation={bp['activation']}, alpha={bp['alpha']:.5f}, lr={bp['lr']:.5f}")

model = MLPRegressor(**best_p)
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
print("  FINAL MODEL — MLP (Faba Bean, 19 Features)")
print(SEP)
print(f"  Algorithm      : MLPRegressor (scikit-learn)")
print(f"  Layers         : {best_layers}")
print(f"  Activation     : {bp['activation']}")
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

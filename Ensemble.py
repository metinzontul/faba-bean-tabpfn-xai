import warnings; warnings.filterwarnings('ignore')
import sys, io, time
from pathlib import Path
import numpy as np, pandas as pd
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import NuSVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA      = Path(__file__).parent / "data" / "faba_bean_dataset.csv"
TARGET    = 'Single_Plant_Yield_g'
N_TRIALS  = 60
KF_SPLITS = 5
SEED      = 42

df    = pd.read_csv(DATA, sep=';')
FEATS = [c for c in df.columns if c != TARGET]
X_raw = df[FEATS].values
y     = df[TARGET].values

X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, test_size=0.20, random_state=SEED)

sc   = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=SEED)
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

kf = KFold(n_splits=KF_SPLITS, shuffle=True, random_state=SEED)

SEP = "=" * 60
print(SEP)
print(f"  Ensemble — Faba Bean (OOF Blend + Stacking)")
print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}  |  Features: {len(FEATS)}")
print(SEP)

def oof_score(oof):
    r2s = []
    for _, val_idx in kf.split(X_tr):
        r2s.append(r2_score(y_tr[val_idx], oof[val_idx]))
    return np.mean(r2s), np.std(r2s)

oof_dict   = {}
final_dict = {}

# 1. NuSVR
print("\n[1/7] NuSVR Optuna tuning...")
t0 = time.time()

def svr_obj(trial):
    p = dict(
        nu    = trial.suggest_float('nu',    0.01, 0.99),
        C     = trial.suggest_float('C',     0.1, 300.0, log=True),
        gamma = trial.suggest_float('gamma', 1e-4, 10.0,  log=True),
    )
    oof = cross_val_predict(NuSVR(kernel='rbf', **p), X_tr, y_tr, cv=kf, n_jobs=-1)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(svr_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'kernel': 'rbf'}
m = NuSVR(**bp)
oof_svr = cross_val_predict(m, X_tr, y_tr, cv=kf, n_jobs=-1)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_svr)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['SVR'] = oof_svr
final_dict['SVR'] = m

# 2. ExtraTrees
print("\n[2/7] ExtraTrees Optuna tuning...")
t0 = time.time()

def et_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        max_depth         = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 30]),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10),
        max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        random_state=SEED, n_jobs=-1,
    )
    oof = cross_val_predict(ExtraTreesRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(et_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_state': SEED, 'n_jobs': -1}
m = ExtraTreesRegressor(**bp)
oof_et = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_et)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['ExtraTrees'] = oof_et
final_dict['ExtraTrees'] = m

# 3. RandomForest
print("\n[3/7] RandomForest Optuna tuning...")
t0 = time.time()

def rf_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        max_depth         = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 30]),
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10),
        max_features      = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        random_state=SEED, n_jobs=-1,
    )
    oof = cross_val_predict(RandomForestRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(rf_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_state': SEED, 'n_jobs': -1}
m = RandomForestRegressor(**bp)
oof_rf = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_rf)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['RF'] = oof_rf
final_dict['RF'] = m

# 4. HistGradientBoosting
print("\n[4/7] HGB Optuna tuning...")
t0 = time.time()

def hgb_obj(trial):
    p = dict(
        max_iter          = trial.suggest_int('max_iter', 100, 600, step=50),
        max_depth         = trial.suggest_int('max_depth', 2, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        l2_regularization = trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True),
        min_samples_leaf  = trial.suggest_int('min_samples_leaf', 5, 50),
        random_state=SEED,
    )
    oof = cross_val_predict(HistGradientBoostingRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(hgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_state': SEED}
m = HistGradientBoostingRegressor(**bp)
oof_hgb = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_hgb)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['HGB'] = oof_hgb
final_dict['HGB'] = m

# 5. XGBoost
print("\n[5/7] XGBoost Optuna tuning...")
t0 = time.time()

def xgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        max_depth         = trial.suggest_int('max_depth', 2, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 10),
        random_state=SEED, n_jobs=-1, verbosity=0,
    )
    oof = cross_val_predict(XGBRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_state': SEED, 'n_jobs': -1, 'verbosity': 0}
m = XGBRegressor(**bp)
oof_xgb = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_xgb)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['XGBoost'] = oof_xgb
final_dict['XGBoost'] = m

# 6. CatBoost
print("\n[6/7] CatBoost Optuna tuning...")
t0 = time.time()

def cat_obj(trial):
    p = dict(
        iterations          = trial.suggest_int('iterations', 200, 800, step=50),
        depth               = trial.suggest_int('depth', 2, 8),
        learning_rate       = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        l2_leaf_reg         = trial.suggest_float('l2_leaf_reg', 1e-3, 20.0, log=True),
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0),
        random_strength     = trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        random_seed=SEED, verbose=0, thread_count=-1,
    )
    oof = cross_val_predict(CatBoostRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(cat_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_seed': SEED, 'verbose': 0, 'thread_count': -1}
m = CatBoostRegressor(**bp)
oof_cat = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_cat)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['CatBoost'] = oof_cat
final_dict['CatBoost'] = m

# 7. LightGBM
print("\n[7/7] LightGBM Optuna tuning...")
t0 = time.time()

def lgbm_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 800, step=50),
        num_leaves        = trial.suggest_int('num_leaves', 15, 127),
        learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_alpha         = trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        min_child_samples = trial.suggest_int('min_child_samples', 5, 50),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    oof = cross_val_predict(lgb.LGBMRegressor(**p), X_tr, y_tr, cv=kf)
    return r2_score(y_tr, oof)

s = optuna.create_study(direction='maximize')
s.optimize(lgbm_obj, n_trials=N_TRIALS, show_progress_bar=True)
bp = {**s.best_params, 'random_state': SEED, 'n_jobs': -1, 'verbose': -1}
m = lgb.LGBMRegressor(**bp)
oof_lgb = cross_val_predict(m, X_tr, y_tr, cv=kf)
m.fit(X_tr, y_tr)
r2m, r2s = oof_score(oof_lgb)
print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}  ({time.time()-t0:.0f}s)")
oof_dict['LightGBM'] = oof_lgb
final_dict['LightGBM'] = m

# TabPFN (optional)
print("\n[+] Trying TabPFN (optional)...")
try:
    from tabpfn import TabPFNRegressor
    oof_tab = np.zeros(len(y_tr))
    for tr_idx, val_idx in kf.split(X_tr):
        _m = TabPFNRegressor(n_estimators=4, random_state=SEED)
        _m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_tab[val_idx] = _m.predict(X_tr[val_idx])
    r2m, r2s = oof_score(oof_tab)
    print(f"  OOF CV R²: {r2m:.4f} ± {r2s:.4f}")
    oof_dict['TabPFN'] = oof_tab
    tab_final = TabPFNRegressor(n_estimators=4, random_state=SEED)
    tab_final.fit(X_tr, y_tr)
    final_dict['TabPFN'] = tab_final
except Exception as e:
    print(f"  TabPFN skipped: {e}")

# (A) OOF Blend — weight optimisation with Optuna
print("\n" + SEP)
print("  (A) Blend Weight Optimisation...")
print(SEP)

names = list(oof_dict.keys())
oof_matrix_tr = np.column_stack([oof_dict[n] for n in names])
oof_matrix_te = np.column_stack([final_dict[n].predict(X_te) for n in names])

def blend_obj(trial):
    raw = np.array([trial.suggest_float(f'w_{i}', 0.0, 1.0) for i in range(len(names))])
    w   = raw / raw.sum()
    blend = oof_matrix_tr @ w
    return r2_score(y_tr, blend)

s_blend = optuna.create_study(direction='maximize')
s_blend.optimize(blend_obj, n_trials=300, show_progress_bar=True)

raw_w = np.array([s_blend.best_params[f'w_{i}'] for i in range(len(names))])
w_opt = raw_w / raw_w.sum()

blend_oof_tr  = oof_matrix_tr @ w_opt
blend_pred_te = oof_matrix_te @ w_opt

blend_cv_r2, blend_cv_r2s = oof_score(blend_oof_tr)
blend_test_r2    = r2_score(y_te, blend_pred_te)
blend_test_rmse  = np.sqrt(mean_squared_error(y_te, blend_pred_te))
blend_test_mae   = mean_absolute_error(y_te, blend_pred_te)
blend_test_mape  = np.mean(np.abs((y_te - blend_pred_te) / (np.abs(y_te) + 1e-8))) * 100
blend_test_nrmse = blend_test_rmse / np.mean(y_te)

# (B) Ridge Stacking meta-learner
print("\n  (B) Ridge Stacking meta-learner...")
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
ridge.fit(oof_matrix_tr, y_tr)

stack_pred_te    = ridge.predict(oof_matrix_te)
stack_test_r2    = r2_score(y_te, stack_pred_te)
stack_test_rmse  = np.sqrt(mean_squared_error(y_te, stack_pred_te))
stack_test_mae   = mean_absolute_error(y_te, stack_pred_te)
stack_test_mape  = np.mean(np.abs((y_te - stack_pred_te) / (np.abs(y_te) + 1e-8))) * 100
stack_test_nrmse = stack_test_rmse / np.mean(y_te)

# Results summary
print("\n" + SEP)
print("  INDIVIDUAL MODEL OOF CV R² SUMMARY")
print(SEP)
single_results = []
for n in names:
    r2m, r2s = oof_score(oof_dict[n])
    te_r2   = r2_score(y_te, final_dict[n].predict(X_te))
    te_rmse = np.sqrt(mean_squared_error(y_te, final_dict[n].predict(X_te)))
    print(f"  {n:<12}  OOF CV R²: {r2m:.4f} ± {r2s:.4f}   Test R²: {te_r2:.4f}   RMSE: {te_rmse:.4f}")
    single_results.append({'Model': n, 'OOF_CV_R2': r2m, 'OOF_CV_R2_std': r2s,
                           'Test_R2': te_r2, 'Test_RMSE': te_rmse})

print(SEP)
print(f"  (A) OOF Blend    — CV R²: {blend_cv_r2:.4f} ± {blend_cv_r2s:.4f}")
print(f"                     Test R²: {blend_test_r2:.4f}   RMSE: {blend_test_rmse:.4f}")
print(f"                     MAE: {blend_test_mae:.4f}   MAPE: {blend_test_mape:.2f}%   NRMSE: {blend_test_nrmse:.4f}")
print(SEP)
print(f"  (B) Ridge Stack  — Test R²: {stack_test_r2:.4f}   RMSE: {stack_test_rmse:.4f}")
print(f"                     MAE: {stack_test_mae:.4f}   MAPE: {stack_test_mape:.2f}%   NRMSE: {stack_test_nrmse:.4f}")
print(SEP)

print("\n  Blend Weights:")
for n, w in zip(names, w_opt):
    print(f"    {n:<12} : {w:.4f} ({w*100:.1f}%)")

print("\n  Ridge Stacking Coefficients:")
for n, c in zip(names, ridge.coef_):
    print(f"    {n:<12} : {c:.4f}")

best_r2     = max(blend_test_r2, stack_test_r2)
best_method = "OOF Blend" if blend_test_r2 >= stack_test_r2 else "Ridge Stacking"
print(SEP)
print(f"  BEST METHOD    : {best_method}")
print(f"  BEST TEST R²   : {best_r2:.4f}")
print(SEP)

rows = single_results + [
    {'Model': 'Ensemble_Blend', 'OOF_CV_R2': blend_cv_r2, 'OOF_CV_R2_std': blend_cv_r2s,
     'Test_R2': blend_test_r2, 'Test_RMSE': blend_test_rmse},
    {'Model': 'Ensemble_Stack', 'OOF_CV_R2': None, 'OOF_CV_R2_std': None,
     'Test_R2': stack_test_r2, 'Test_RMSE': stack_test_rmse},
]
out_csv = Path(__file__).parent / "ensemble_results.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n  Results saved: {out_csv}")

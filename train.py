from my_utils.base_data_handler import BaseDataHandler
handler = BaseDataHandler('food_time_delivery_clean.csv')
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from my_utils import functions
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from optuna.integration import XGBoostPruningCallback, LightGBMPruningCallback, CatBoostPruningCallback

# ===============================
# User-configurable flag
# ===============================
USE_PCA = False
PCA_COMPONENTS = 23  # Only used if USE_PCA=True

# --- 1. Data Setup and Log Transformation ---
X, y = handler.get_training_data(target="target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split for tuning (Train/Validation)
X_train_part, X_valid, y_train_part, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# --- 2. Scaling ---
scaler = RobustScaler()
X_train_scaled_array = scaler.fit_transform(X_train)
X_test_scaled_array = scaler.transform(X_test)
X_train_part_scaled_array = scaler.transform(X_train_part)
X_valid_scaled_array = scaler.transform(X_valid)

# --- 3. Optional PCA ---
if USE_PCA:
    pca = PCA(n_components=PCA_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled_array)
    X_test_pca = pca.transform(X_test_scaled_array)
    X_train_part_pca = pca.transform(X_train_part_scaled_array)
    X_valid_pca = pca.transform(X_valid_scaled_array)

    # PCA feature names for DataFrames
    feature_names = [f'pca_comp_{i+1}' for i in range(pca.n_components_)]
else:
    # Skip PCA, use scaled arrays directly
    X_train_pca = X_train_scaled_array
    X_test_pca = X_test_scaled_array
    X_train_part_pca = X_train_part_scaled_array
    X_valid_pca = X_valid_scaled_array

    feature_names = X_train.columns.tolist()  # Keep original feature names

# --- 4. Convert to DataFrame for LightGBM/Stacking ---
X_train_scaled_df = pd.DataFrame(X_train_pca, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_pca, columns=feature_names)
X_train_part_scaled_df = pd.DataFrame(X_train_part_pca, columns=feature_names)
X_valid_scaled_df = pd.DataFrame(X_valid_pca, columns=feature_names)

# --- 5. Log transform targets ---
y_train_log = np.log1p(y_train_part)
y_valid_log = np.log1p(y_valid)
y_train_full_log = np.log1p(y_train)

# --- 6. XGBoost DMatrix (always NumPy) ---
dtrain_part = xgb.DMatrix(X_train_part_pca, label=y_train_log)
dvalid = xgb.DMatrix(X_valid_pca, label=y_valid_log)
dtrain_full = xgb.DMatrix(X_train_pca, label=y_train_full_log)
dtest = xgb.DMatrix(X_test_pca)

print(f"Using PCA: {USE_PCA}, feature shape: {X_train_scaled_df.shape}")


def objective_xgb(trial):
    ## --- 2. Search Space ---
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "verbosity": 0,
        "eval_metric": "rmse",
        
        # Hyperparameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 600, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }

    ## --- 3. Setup Pruning Callback ---
    # Monitors the RMSE of the validation set (named 'validation')
    pruning_callback = XGBoostPruningCallback(trial, "validation-rmse")

    ## --- 4. Run Training on Partial Data ---
    model = xgb.train(
        params,
        dtrain_part,
        num_boost_round=1000,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose_eval=False
    )

    # Return the best score (RMSE on log-prices)
    trial.set_user_attr("best_iteration", model.best_iteration)
    return model.best_score

def objective_lgbm(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 600, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }

    pruning_callback = LightGBMPruningCallback(trial, 'rmse')

    model = LGBMRegressor(**params)

    model.fit(
        X_train_part_scaled_df, y_train_log,
        eval_set=[(X_valid_scaled_df, y_valid_log)],
        eval_metric='rmse',
        callbacks=[pruning_callback, early_stopping(50, verbose=False)],
    )

    trial.set_user_attr('best_iteration', model.best_iteration_)
    return model.best_score_['valid_0']['rmse']


def objective_cat(trial):
    params = {
        'objective': 'RMSE',
        'verbose': 0,
        'random_seed': 42,
        'iterations': trial.suggest_int('iterations', 600, 1200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'rsm': trial.suggest_float('rsm', 0.6, 1.0),
        'allow_writing_files': False
    }

    pruning_callback = CatBoostPruningCallback(trial, 'RMSE')

    model = CatBoostRegressor(**params)

    model.fit(
        X_train_part_scaled_df, y_train_log,
        eval_set=[(X_valid_scaled_df, y_valid_log)],
        callbacks=[pruning_callback],
        early_stopping_rounds=50,
        verbose=0
    )

    trial.set_user_attr('best_iteration', model.best_iteration_)
    return model.get_best_score()['validation']['RMSE']

# --- Optimization Run ---
print("Optimizing XGBoost (using DMatrix)...")
study_xgb = optuna.create_study(direction="minimize", study_name="Food time delivery: XGBoost")
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)

print("Optimizing LightGBM (using DMatrix)...")
study_lgbm = optuna.create_study(direction="minimize", study_name="Food time delivery: LightGBM")
study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=False)
print("Optimizing CatBoost (using DMatrix)...")

study_cat = optuna.create_study(direction="minimize", study_name="Food time delivery: CatBoost")
study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=False)

# --- XGBoost ---
best_xgb_params = study_xgb.best_params.copy()
xgb_n_rounds = max(1, study_xgb.best_trial.user_attrs.get('best_iteration', 2000))
print(f'xgb n rounds: {xgb_n_rounds}')
best_xgb_params.update({
    'n_estimators': xgb_n_rounds,
    "n_jobs": -1,
    "device": "cuda",
    "seed" : 42})

# --- LightGBM ---
best_lgbm_params = study_lgbm.best_params.copy()
lgbm_n_rounds = max(1, study_lgbm.best_trial.user_attrs.get('best_iteration', 2000))
print(f'lgbm n rounds: {lgbm_n_rounds}')
best_lgbm_params.update({
    "n_estimators": lgbm_n_rounds,
    'n_jobs':-1,
    'verbose':-1,
    'random_state':42})

# --- CatBoost ---
best_cat_params = study_cat.best_params.copy()
cat_n_rounds = max(1, study_cat.best_trial.user_attrs.get('best_iteration', 2000))
print(f'cat n rounds: {cat_n_rounds}')
best_cat_params.update({
    "iterations": cat_n_rounds, 
    "verbose": 0, 
    "random_state": 42, 
    "allow_writing_files": False})


print("\n5. Assembling Best Models for Stacking...")
estimators = [
    ('xgb', xgb.XGBRegressor(**best_xgb_params)),
    ('lgbm', LGBMRegressor(**best_lgbm_params)),
    ('cat', CatBoostRegressor(**best_cat_params))
]

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from xgboost import XGBClassifier
from catboost import CatBoostRegressor


# ==========================================================
# 1. SMART SOFT MODEL SELECTOR – FIXED VERSION
# ==========================================================

class SmartSoftModelSelector(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, selector_model=None, cv=5):
        self.estimators = estimators
        self.model_names = [name for name, _ in estimators]
        self.cv = cv
        
        self.selector_model = selector_model if selector_model else XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            device='cuda',
            tree_method='hist',
            n_jobs=1,
            random_state=42
        )

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.array(y)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # OOF matrix: [n_samples, n_models]
        oof_preds = np.zeros((len(X), len(self.estimators)))

        print("\n=== Training Esperti con OOF ===")
        for m_idx, (name, model) in enumerate(self.estimators):
            print(f" -> {name}")
            for train_i, valid_i in kf.split(X):
                X_tr, X_val = X.iloc[train_i], X.iloc[valid_i]  # FIXED
                y_tr = y[train_i]
                
                model_fold = clone(model)
                model_fold.fit(X_tr, y_tr)
                oof_preds[valid_i, m_idx] = model_fold.predict(X_val)

        # Selettore target = il modello con errore minore per ciascun sample
        errors = np.abs(oof_preds - y.reshape(-1, 1))
        y_best = errors.argmin(axis=1)

        print("=== Training Gating Network ===")
        self.selector_model.fit(X, y_best)

        # Ora alleniamo definitivamente ogni esperto sul 100% del dataset
        print("=== Training Esperti Finali (100%) ===")
        self.fitted_estimators_ = []
        for name, model in self.estimators:
            m = clone(model)
            m.fit(X, y)
            self.fitted_estimators_.append(m)

        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        base_preds = np.column_stack([m.predict(X) for m in self.fitted_estimators_])
        weights = self.selector_model.predict_proba(X)
        return np.sum(base_preds * weights, axis=1)



# ==========================================================
# 2. RESIDUAL CORRECTOR – FIXED VERSION
# ==========================================================

class ResidualCorrectedMoE(BaseEstimator, RegressorMixin):
    def __init__(self, base_moe_model, corrector_model=None, cv=5):
        self.base_moe_model = base_moe_model
        self.cv = cv
        
        self.corrector_model = corrector_model if corrector_model else CatBoostRegressor(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            task_type='GPU',
            devices='0',
            verbose=0,
            random_state=42,
            allow_writing_files=False
        )

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = np.array(y)

        print("\n=== Calcolo Residui OOF del MoE ===")

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))

        for train_i, valid_i in kf.split(X):
            X_tr, X_val = X.iloc[train_i], X.iloc[valid_i]  # FIXED
            y_tr = y[train_i]

            # clone MoE e retrain sul fold
            moe_fold = clone(self.base_moe_model)
            moe_fold.fit(X_tr, y_tr)
            oof_preds[valid_i] = moe_fold.predict(X_val)

        residuals = y - oof_preds
        print(f" Residual mean: {residuals.mean():.4f}")
        print(f" Residual std : {residuals.std():.4f}")

        print("\n=== Training Correttore Residui ===")
        self.corrector_model.fit(X, residuals)

        # salva modello MoE finale (train 100%)
        self.base_moe_model.fit(X, y)

        return self

    def predict(self, X):
        X = pd.DataFrame(X)
        base_pred = self.base_moe_model.predict(X)
        correction = self.corrector_model.predict(X)
        return base_pred + correction


# Creiamo e addestriamo il sistema MoE
moe_model = SmartSoftModelSelector(estimators=estimators)
moe_model.fit(X_train_scaled_df, y_train_full_log)

# ---------------------------------------------------------
# 6. AGGIUNTA DEL CORRETTORE DEI RESIDUI (Residual Learning)
# ---------------------------------------------------------
print("\n6. Training Correttore dei Residui...")
# Costruzione Finale
# Nota: passiamo il moe_model già addestrato, ma la classe ResidualCorrectedMoE
# userà cross_val_predict che internamente farà cloni e fit su fold.
final_system = ResidualCorrectedMoE(base_moe_model=moe_model)
final_system.fit(X_train_scaled_df, y_train_full_log)

# Valutazione
y_pred_log = final_system.predict(X_test_scaled_df)
y_pred = np.expm1(y_pred_log)

final_rmse = root_mean_squared_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
# Plot veloce
corrections_test = final_system.corrector_model.predict(X_test_scaled_df)
base_preds_test = final_system.base_moe_model.predict(X_test_scaled_df)

import matplotlib.pyplot as plt
import seaborn as sns

print(f"\n==========================================")
print(f" RISULTATI FINALI OTTIMIZZATI")
print(f"==========================================")
print(f" RMSE: {final_rmse:.5f}")
print(f" R^2 : {final_r2:.5f}")
print(f"==========================================")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=base_preds_test, y=corrections_test, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Bias Correction vs Price (Optimized Models)")
plt.xlabel("Predicted Price")
plt.ylabel("Correction")
plt.show()
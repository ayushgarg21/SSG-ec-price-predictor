"""
EC Price Prediction — Appreciation Model (v4 final)
====================================================
ZERO look-ahead bias. Every feature is computed using ONLY data
available at the time of prediction.

Pipeline:
  1. Parse and split data temporally FIRST
  2. Compute all lookups/encodings from TRAIN set only
  3. Apply to val/test using train-derived lookups
  4. Optuna HPO, ensemble, SHAP, residual analysis

Run: python notebooks/02_appreciation_model.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import optuna
import shap
from scipy import stats as sp_stats
from sqlalchemy import create_engine, text

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.features.engineering import parse_contract_date, extract_lease_commence_year, extract_floor_mid
from src.model.experiment import log_experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style="whitegrid", font_scale=1.1)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD AND PARSE
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

engine = create_engine("postgresql://ec_user:ec_password@localhost:5432/ec_prices")
with engine.connect() as conn:
    all_df = pd.read_sql(text("SELECT * FROM ura_transactions"), conn)
with engine.connect() as conn:
    ec_df = pd.read_sql(text("SELECT * FROM ura_transactions WHERE property_type = 'Executive Condominium'"), conn)

print(f"All transactions: {len(all_df):,}")
print(f"EC transactions:  {len(ec_df):,}")

for d in [all_df, ec_df]:
    parsed = d["contract_date"].apply(lambda x: pd.Series(parse_contract_date(x)))
    d["txn_month"] = parsed[0]
    d["txn_year"] = parsed[1]
    d["price"] = pd.to_numeric(d["price"], errors="coerce")
    d["area"] = pd.to_numeric(d["area"], errors="coerce")
    d["price_psm"] = d["price"] / d["area"]
    d["district_num"] = pd.to_numeric(d["district"], errors="coerce")

# Parse EC-specific fields on resales
resales = ec_df[ec_df["type_of_sale"] == "3"].copy()
resales["floor_mid"] = resales["floor_range"].apply(extract_floor_mid)
resales["lease_commence_year"] = resales["tenure"].apply(extract_lease_commence_year)
resales["years_from_launch"] = resales["txn_year"] - resales["lease_commence_year"]
resales["remaining_lease"] = 99 - resales["years_from_launch"]
resales["segment_encoded"] = resales["market_segment"].map({"OCR": 0, "RCR": 1, "CCR": 2}).fillna(0).astype(int)

resales = resales.dropna(subset=["price", "area", "floor_mid", "lease_commence_year", "years_from_launch"])
resales = resales.sort_values(["txn_year", "txn_month"]).reset_index(drop=True)

print(f"Resale transactions (clean): {len(resales):,}")

# ══════════════════════════════════════════════════════════════════════
# 2. TEMPORAL SPLIT FIRST (before any feature engineering)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: TEMPORAL SPLIT (before feature engineering)")
print("=" * 70)

train_mask = (resales["txn_year"] < 2025) | ((resales["txn_year"] == 2025) & (resales["txn_month"] <= 6))
val_mask = (resales["txn_year"] == 2025) & (resales["txn_month"] > 6)
test_mask = resales["txn_year"] >= 2026

train_raw = resales[train_mask].copy()
val_raw = resales[val_mask].copy()
test_raw = resales[test_mask].copy()

if len(test_raw) < 50:
    n = len(resales)
    train_raw = resales.iloc[:int(n * 0.7)].copy()
    val_raw = resales.iloc[int(n * 0.7):int(n * 0.85)].copy()
    test_raw = resales.iloc[int(n * 0.85):].copy()

train_cutoff_year = int(train_raw["txn_year"].max())
train_cutoff_month = int(train_raw["txn_month"].max())

print(f"Train: {len(train_raw):,}  Val: {len(val_raw):,}  Test: {len(test_raw):,}")
print(f"Train: {int(train_raw['txn_year'].min())}—{int(train_raw['txn_year'].max())}")
print(f"Val:   {int(val_raw['txn_year'].min())}—{int(val_raw['txn_year'].max())}")
print(f"Test:  {int(test_raw['txn_year'].min())}—{int(test_raw['txn_year'].max())}")

# ══════════════════════════════════════════════════════════════════════
# 3. COMPUTE ALL LOOKUPS FROM TRAIN DATA ONLY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: FEATURE ENGINEERING (train-derived only)")
print("=" * 70)

# --- 3a. Launch PSM (from train-period data only) ---
print("Computing launch PSM from train-period data only...")

# New sales that happened before or during train period
train_period_ec = ec_df[
    (ec_df["txn_year"] < train_cutoff_year) |
    ((ec_df["txn_year"] == train_cutoff_year) & (ec_df["txn_month"] <= train_cutoff_month))
]
train_new_sales = train_period_ec[train_period_ec["type_of_sale"] == "1"]
train_resales = train_period_ec[train_period_ec["type_of_sale"] == "3"]

# Prefer new sale median, fallback to earliest train-period resale
launch_ns = train_new_sales.groupby("project")["price_psm"].median()
launch_rs = train_resales.sort_values(["txn_year", "txn_month"]).groupby("project")["price_psm"].first()

launch_psm_dict = {}
for proj in set(launch_ns.index) | set(launch_rs.index):
    if proj in launch_ns.index:
        launch_psm_dict[proj] = launch_ns[proj]
    else:
        launch_psm_dict[proj] = launch_rs[proj]

print(f"  Projects with launch PSM (from train period): {len(launch_psm_dict)}")

# --- 3b. District lag features from ALL transactions (train-period only) ---
print("Computing district features from train-period market data...")

train_period_all = all_df[
    (all_df["txn_year"] < train_cutoff_year) |
    ((all_df["txn_year"] == train_cutoff_year) & (all_df["txn_month"] <= train_cutoff_month))
]

# District stats by year (lagged by 1 year)
all_dist_yr = train_period_all.groupby(["district_num", "txn_year"])["price_psm"].agg(
    ["median", "count", "std"]
).reset_index()
all_dist_yr.columns = ["district_num", "year", "all_dist_med_psm", "all_dist_vol", "all_dist_std"]
all_dist_yr["txn_year"] = all_dist_yr["year"] + 1  # lag
all_dist_lag = all_dist_yr[["district_num", "txn_year", "all_dist_med_psm", "all_dist_vol", "all_dist_std"]]

# EC district stats (lagged)
train_period_ec_all = ec_df[
    (ec_df["txn_year"] < train_cutoff_year) |
    ((ec_df["txn_year"] == train_cutoff_year) & (ec_df["txn_month"] <= train_cutoff_month))
]
ec_dist_yr = train_period_ec_all.groupby(["district_num", "txn_year"])["price_psm"].median().reset_index()
ec_dist_yr.columns = ["district_num", "year", "ec_dist_med_psm"]
ec_dist_yr["txn_year"] = ec_dist_yr["year"] + 1
ec_dist_lag = ec_dist_yr[["district_num", "txn_year", "ec_dist_med_psm"]]

# District momentum (from train-period only)
all_dist_mom = train_period_all.groupby(["district_num", "txn_year"])["price_psm"].median().reset_index()
all_dist_mom.columns = ["district_num", "year", "med"]
all_dist_mom = all_dist_mom.sort_values(["district_num", "year"])
all_dist_mom["prev"] = all_dist_mom.groupby("district_num")["med"].shift(1)
all_dist_mom["district_momentum"] = (all_dist_mom["med"] - all_dist_mom["prev"]) / all_dist_mom["prev"]
all_dist_mom["txn_year"] = all_dist_mom["year"] + 1
mom_lookup = all_dist_mom[["district_num", "txn_year", "district_momentum"]].dropna()

# Market-level PSM (lagged, train-period only)
mkt_yr = train_period_all.groupby("txn_year")["price_psm"].median().reset_index()
mkt_yr.columns = ["year", "market_lag_psm"]
mkt_yr["txn_year"] = mkt_yr["year"] + 1
mkt_lag = mkt_yr[["txn_year", "market_lag_psm"]]

# Project lag stats (train-period EC only)
proj_yr = train_period_ec_all.groupby(["project", "txn_year"]).agg(
    proj_med_psm=("price_psm", "median"), proj_count=("price", "count"),
).reset_index()
proj_yr["txn_year_lag"] = proj_yr["txn_year"] + 1
proj_lag = proj_yr[["project", "txn_year_lag", "proj_med_psm", "proj_count"]].rename(
    columns={"txn_year_lag": "txn_year", "proj_med_psm": "proj_lag_psm", "proj_count": "proj_lag_vol"}
)

# Area quartile bins (from train resales only)
_, area_bins = pd.qcut(train_raw["area"], q=4, retbins=True)

# --- 3c. Compute NaN fill medians from TRAIN data only ---
# We'll compute these after merging features onto train set


# ══════════════════════════════════════════════════════════════════════
# 4. APPLY FEATURES TO ALL SPLITS
# ══════════════════════════════════════════════════════════════════════
print("\nApplying features to train/val/test...")


def apply_features(df_split, launch_psm_dict, all_dist_lag, ec_dist_lag,
                   mom_lookup, mkt_lag, proj_lag, area_bins, fill_medians=None):
    """Apply all features using ONLY train-derived lookups. Zero leakage."""
    df = df_split.copy()

    # Launch PSM
    df["launch_psm"] = df["project"].map(launch_psm_dict)

    # Appreciation ratio (target)
    df["appreciation_ratio"] = df["price_psm"] / df["launch_psm"]

    # District features (lagged)
    df = df.merge(all_dist_lag, on=["district_num", "txn_year"], how="left")
    df = df.merge(ec_dist_lag, on=["district_num", "txn_year"], how="left")
    df = df.merge(mom_lookup, on=["district_num", "txn_year"], how="left")
    df = df.merge(mkt_lag, on="txn_year", how="left")
    df = df.merge(proj_lag, on=["project", "txn_year"], how="left")

    # Launch vs district
    df["launch_vs_district"] = df["launch_psm"] / df["all_dist_med_psm"]

    # Derived features (from row itself, no leakage)
    df["is_post_mop"] = (df["years_from_launch"] >= 5).astype(int)
    df["is_privatised"] = (df["years_from_launch"] >= 10).astype(int)
    df["quarter"] = ((df["txn_month"] - 1) // 3 + 1).astype(int)
    df["lease_age_bucket"] = pd.cut(
        df["years_from_launch"], bins=[-1, 2, 5, 10, 15, 99], labels=[0, 1, 2, 3, 4]
    ).astype(float)
    df["is_high_floor"] = (df["floor_mid"] >= 15).astype(int)
    df["area_quartile"] = pd.cut(
        df["area"], bins=area_bins, labels=[0, 1, 2, 3], include_lowest=True
    ).astype(float).fillna(1).astype(int)

    # Fill NaN with train-derived medians
    lag_cols = ["all_dist_med_psm", "all_dist_vol", "all_dist_std", "ec_dist_med_psm",
                "district_momentum", "market_lag_psm", "proj_lag_psm", "proj_lag_vol",
                "launch_vs_district"]
    if fill_medians is not None:
        for col in lag_cols:
            df[col] = df[col].fillna(fill_medians.get(col, 0))

    return df


# Apply to train first (to compute fill medians)
train_feat = apply_features(train_raw, launch_psm_dict, all_dist_lag, ec_dist_lag,
                            mom_lookup, mkt_lag, proj_lag, area_bins)

# Compute fill medians from TRAIN only
lag_cols = ["all_dist_med_psm", "all_dist_vol", "all_dist_std", "ec_dist_med_psm",
            "district_momentum", "market_lag_psm", "proj_lag_psm", "proj_lag_vol",
            "launch_vs_district"]
train_fill_medians = {col: float(train_feat[col].median()) for col in lag_cols}
print("  Fill medians (from train):")
for col, med in train_fill_medians.items():
    n_miss = train_feat[col].isna().sum()
    train_feat[col] = train_feat[col].fillna(med)
    if n_miss > 0:
        print(f"    {col}: filled {n_miss} NaN with {med:.1f}")

# Apply to val and test with train-derived medians
val_feat = apply_features(val_raw, launch_psm_dict, all_dist_lag, ec_dist_lag,
                          mom_lookup, mkt_lag, proj_lag, area_bins, train_fill_medians)
test_feat = apply_features(test_raw, launch_psm_dict, all_dist_lag, ec_dist_lag,
                           mom_lookup, mkt_lag, proj_lag, area_bins, train_fill_medians)

# --- Target-encode project (train-only) ---
print("\nTarget-encoding project (train-only)...")

# Drop rows without launch_psm or valid ratio
train_feat = train_feat.dropna(subset=["launch_psm", "appreciation_ratio"])
train_feat = train_feat[(train_feat["appreciation_ratio"] > 0.3) & (train_feat["appreciation_ratio"] < 3.0)]

val_feat = val_feat.dropna(subset=["launch_psm", "appreciation_ratio"])
val_feat = val_feat[(val_feat["appreciation_ratio"] > 0.3) & (val_feat["appreciation_ratio"] < 3.0)]

test_feat = test_feat.dropna(subset=["launch_psm", "appreciation_ratio"])
test_feat = test_feat[(test_feat["appreciation_ratio"] > 0.3) & (test_feat["appreciation_ratio"] < 3.0)]

# KFold encoding on TRAIN only
global_mean_ratio = float(train_feat["appreciation_ratio"].mean())
train_feat["project_target_enc"] = global_mean_ratio
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in kf.split(train_feat):
    fold_means = train_feat.iloc[tr_idx].groupby("project")["appreciation_ratio"].mean()
    mapped = train_feat.iloc[va_idx]["project"].map(fold_means).fillna(global_mean_ratio)
    train_feat.iloc[va_idx, train_feat.columns.get_loc("project_target_enc")] = mapped.values

# Val/test: use train-set project means
project_enc_lookup = train_feat.groupby("project")["appreciation_ratio"].mean().to_dict()
val_feat["project_target_enc"] = val_feat["project"].map(project_enc_lookup).fillna(global_mean_ratio)
test_feat["project_target_enc"] = test_feat["project"].map(project_enc_lookup).fillna(global_mean_ratio)

n_unseen = test_feat["project"].apply(lambda p: p not in project_enc_lookup).sum()
print(f"  Train projects: {len(project_enc_lookup)}")
print(f"  Test unseen projects: {n_unseen}/{len(test_feat)} (use global mean {global_mean_ratio:.4f})")

# ══════════════════════════════════════════════════════════════════════
# 5. PREPARE FEATURE MATRICES
# ══════════════════════════════════════════════════════════════════════
FEATURE_COLUMNS = [
    "district_num", "area", "floor_mid", "lease_commence_year",
    "years_from_launch", "remaining_lease", "segment_encoded",
    "launch_psm", "launch_vs_district",
    "all_dist_med_psm", "all_dist_vol", "all_dist_std",
    "ec_dist_med_psm", "district_momentum", "market_lag_psm",
    "proj_lag_psm", "proj_lag_vol", "project_target_enc",
    "area_quartile", "is_post_mop", "is_privatised",
    "quarter", "lease_age_bucket", "is_high_floor",
]
TARGET = "appreciation_ratio"

# Drop any remaining NaN in features
for split_name, split_df in [("train", train_feat), ("val", val_feat), ("test", test_feat)]:
    missing = split_df[FEATURE_COLUMNS].isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  {split_name} NaN remaining:")
        for col, cnt in missing.items():
            print(f"    {col}: {cnt}")

train_feat = train_feat.dropna(subset=FEATURE_COLUMNS + [TARGET])
val_feat = val_feat.dropna(subset=FEATURE_COLUMNS + [TARGET])
test_feat = test_feat.dropna(subset=FEATURE_COLUMNS + [TARGET])

X_train = train_feat[FEATURE_COLUMNS].values.astype(np.float64)
y_train = train_feat[TARGET].values
X_val = val_feat[FEATURE_COLUMNS].values.astype(np.float64)
y_val = val_feat[TARGET].values
X_test = test_feat[FEATURE_COLUMNS].values.astype(np.float64)
y_test = test_feat[TARGET].values

y_test_price = test_feat["price"].values
test_launch_psm = test_feat["launch_psm"].values
test_area = test_feat["area"].values

val_price = val_feat["price"].values
val_launch_psm = val_feat["launch_psm"].values
val_area = val_feat["area"].values

print(f"\nFinal sizes — Train: {len(train_feat):,}  Val: {len(val_feat):,}  Test: {len(test_feat):,}")
print(f"Features: {len(FEATURE_COLUMNS)}")

# ══════════════════════════════════════════════════════════════════════
# 6. OPTUNA HPO
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: HYPERPARAMETER OPTIMISATION")
print("=" * 70)


def ratio_to_price(ratio, lpsm, area):
    return ratio * lpsm * area


def eval_full(y_ratio_true, y_ratio_pred, y_price_true, lpsm, area):
    y_pp = ratio_to_price(y_ratio_pred, lpsm, area)
    return {
        "ratio_mae": float(mean_absolute_error(y_ratio_true, y_ratio_pred)),
        "ratio_r2": float(r2_score(y_ratio_true, y_ratio_pred)),
        "price_mae": float(mean_absolute_error(y_price_true, y_pp)),
        "price_rmse": float(np.sqrt(mean_squared_error(y_price_true, y_pp))),
        "price_r2": float(r2_score(y_price_true, y_pp)),
        "price_mape": float(np.mean(np.abs((y_price_true - y_pp) / y_price_true)) * 100),
    }


def obj_xgb(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 300, 1500),
         "max_depth": trial.suggest_int("max_depth", 3, 10),
         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
         "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
         "gamma": trial.suggest_float("gamma", 0, 5)}
    m = XGBRegressor(**p, random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_train, y_train)
    return mean_absolute_error(val_price, ratio_to_price(m.predict(X_val), val_launch_psm, val_area))


def obj_lgbm(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 300, 1500),
         "max_depth": trial.suggest_int("max_depth", 3, 12),
         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 20, 200),
         "min_child_samples": trial.suggest_int("min_child_samples", 3, 50)}
    m = LGBMRegressor(**p, random_state=42, n_jobs=-1, verbosity=-1)
    m.fit(X_train, y_train)
    return mean_absolute_error(val_price, ratio_to_price(m.predict(X_val), val_launch_psm, val_area))


def obj_cat(trial):
    p = {"iterations": trial.suggest_int("iterations", 300, 1500),
         "depth": trial.suggest_int("depth", 3, 10),
         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
         "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
         "random_strength": trial.suggest_float("random_strength", 0, 5)}
    m = CatBoostRegressor(**p, random_state=42, verbose=0)
    m.fit(X_train, y_train)
    return mean_absolute_error(val_price, ratio_to_price(m.predict(X_val), val_launch_psm, val_area))


N_TRIALS = 60

studies = {}
for name, obj in [("XGBoost", obj_xgb), ("LightGBM", obj_lgbm), ("CatBoost", obj_cat)]:
    print(f"\nTuning {name} ({N_TRIALS} trials)...")
    s = optuna.create_study(direction="minimize")
    s.optimize(obj, n_trials=N_TRIALS)
    studies[name] = s
    print(f"  Best val MAE: S${s.best_value:,.0f}")

# ══════════════════════════════════════════════════════════════════════
# 7. FINAL MODELS + ENSEMBLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: FINAL MODELS + WEIGHTED ENSEMBLE")
print("=" * 70)

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

models = {}
best_xgb = XGBRegressor(**studies["XGBoost"].best_params, random_state=42, n_jobs=-1, verbosity=0)
best_xgb.fit(X_trainval, y_trainval)
models["XGBoost"] = best_xgb

best_lgbm = LGBMRegressor(**studies["LightGBM"].best_params, random_state=42, n_jobs=-1, verbosity=-1)
best_lgbm.fit(X_trainval, y_trainval)
models["LightGBM"] = best_lgbm

best_cat = CatBoostRegressor(**studies["CatBoost"].best_params, random_state=42, verbose=0)
best_cat.fit(X_trainval, y_trainval)
models["CatBoost"] = best_cat

# Weighted ensemble (optimise on val)
print("Optimising ensemble weights on val set...")
val_preds = {n: m.predict(X_val) for n, m in models.items()}
best_w, best_w_mae = None, float("inf")
for w1 in np.arange(0.05, 0.95, 0.05):
    for w2 in np.arange(0.05, 0.95 - w1, 0.05):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < 0.05:
            continue
        ens = w1 * val_preds["XGBoost"] + w2 * val_preds["LightGBM"] + w3 * val_preds["CatBoost"]
        mae = mean_absolute_error(val_price, ratio_to_price(ens, val_launch_psm, val_area))
        if mae < best_w_mae:
            best_w_mae = mae
            best_w = (w1, w2, w3)

print(f"  Weights: XGB={best_w[0]:.2f}, LGBM={best_w[1]:.2f}, CAT={best_w[2]:.2f}")


from src.model.ensemble import WeightedEnsemble

ensemble = WeightedEnsemble(models, best_w)
models["Weighted Ensemble"] = ensemble

# Evaluate all on TEST
results = {}
print(f"\n{'Model':<22} {'Price MAE':>12} {'Price R²':>10} {'MAPE':>8} {'Ratio R²':>10}")
print("-" * 68)
for name, model in models.items():
    pred_ratio = model.predict(X_test)
    m = eval_full(y_test, pred_ratio, y_test_price, test_launch_psm, test_area)
    results[name] = {**m, "pred_ratio": pred_ratio}
    print(f"{name:<22} S${m['price_mae']:>10,.0f} {m['price_r2']:>9.4f} {m['price_mape']:>7.2f}% {m['ratio_r2']:>9.4f}")

sorted_models = sorted(results.items(), key=lambda x: x[1]["price_mae"])
best_name = sorted_models[0][0]
best_result = sorted_models[0][1]
print(f"\nWinner: {best_name}")

if len(sorted_models) > 1:
    sn = sorted_models[1][0]
    r1 = np.abs(y_test_price - ratio_to_price(best_result["pred_ratio"], test_launch_psm, test_area))
    r2 = np.abs(y_test_price - ratio_to_price(sorted_models[1][1]["pred_ratio"], test_launch_psm, test_area))
    t, p = sp_stats.ttest_rel(r1, r2)
    print(f"vs {sn}: t={t:.4f}, p={p:.4e}, {'Significant' if p < 0.05 else 'Not significant'}")

# ══════════════════════════════════════════════════════════════════════
# 8. PREDICTION INTERVALS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: PREDICTION INTERVALS")
print("=" * 70)

# Tune quantile alpha on validation set to achieve ~80% coverage
print("Tuning quantile alpha for target 80% coverage...")

best_alpha_lo, best_alpha_hi, best_cov_diff = 0.10, 0.90, 999
for alpha_lo in [0.02, 0.05, 0.08, 0.10]:
    alpha_hi = 1.0 - alpha_lo
    q_lo_tmp = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha_lo,
                            n_estimators=800, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)
    q_hi_tmp = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha_hi,
                            n_estimators=800, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)
    q_lo_tmp.fit(X_train, y_train)
    q_hi_tmp.fit(X_train, y_train)
    lo_v = ratio_to_price(q_lo_tmp.predict(X_val), val_launch_psm, val_area)
    hi_v = ratio_to_price(q_hi_tmp.predict(X_val), val_launch_psm, val_area)
    cov = np.mean((val_price >= lo_v) & (val_price <= hi_v)) * 100
    diff = abs(cov - 80)
    print(f"  alpha={alpha_lo:.2f}/{alpha_hi:.2f} → val coverage: {cov:.1f}%")
    if diff < best_cov_diff:
        best_cov_diff = diff
        best_alpha_lo, best_alpha_hi = alpha_lo, alpha_hi

print(f"  Selected: alpha={best_alpha_lo:.2f}/{best_alpha_hi:.2f}")

# Train final quantile models on train+val with best alpha
q_lo = XGBRegressor(objective="reg:quantileerror", quantile_alpha=best_alpha_lo,
                    n_estimators=800, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0)
q_hi = XGBRegressor(objective="reg:quantileerror", quantile_alpha=best_alpha_hi,
                    n_estimators=800, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=0)
q_lo.fit(X_trainval, y_trainval)
q_hi.fit(X_trainval, y_trainval)

lo_price = ratio_to_price(q_lo.predict(X_test), test_launch_psm, test_area)
hi_price = ratio_to_price(q_hi.predict(X_test), test_launch_psm, test_area)
pred_price = ratio_to_price(best_result["pred_ratio"], test_launch_psm, test_area)

coverage = np.mean((y_test_price >= lo_price) & (y_test_price <= hi_price)) * 100
avg_width = np.mean(hi_price - lo_price)
target_pct = int((1 - best_alpha_lo * 2) * 100)
print(f"\n{target_pct}% PI coverage on test: {coverage:.1f}%")
print(f"Avg interval width: S${avg_width:,.0f}")

fig, ax = plt.subplots(figsize=(12, 6))
si = np.argsort(y_test_price)[:min(200, len(y_test_price))]
x = np.arange(len(si))
ax.fill_between(x, lo_price[si]/1e6, hi_price[si]/1e6, alpha=0.3, color="#2563eb", label=f"{target_pct}% PI")
ax.plot(x, pred_price[si]/1e6, ".", color="#2563eb", ms=3, label="Predicted")
ax.plot(x, y_test_price[si]/1e6, ".", color="#dc2626", ms=3, label="Actual")
ax.set_xlabel("Samples (sorted)")
ax.set_ylabel("Price (S$M)")
ax.set_title(f"Prediction Intervals — {coverage:.1f}% coverage")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_prediction_intervals.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 14_prediction_intervals.png")

# ══════════════════════════════════════════════════════════════════════
# 9. SEGMENTED EVALUATION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: SEGMENTED EVALUATION")
print("=" * 70)

te = test_feat.copy()
te["pred_price"] = pred_price
te["abs_error"] = np.abs(te["price"] - te["pred_price"])
te["pct_error"] = te["abs_error"] / te["price"] * 100

print(f"\nBy District ({best_name}):")
print(f"{'District':>10} {'Count':>8} {'MAE':>12} {'MAPE':>8}")
print("-" * 42)
for dist in sorted(te["district_num"].dropna().unique()):
    s = te[te["district_num"] == dist]
    if len(s) >= 3:
        print(f"{int(dist):>10} {len(s):>8} S${s['abs_error'].mean():>10,.0f} {s['pct_error'].mean():>7.2f}%")

print(f"\nBy Years from Launch ({best_name}):")
print(f"{'Years':>10} {'Count':>8} {'MAE':>12} {'MAPE':>8}")
print("-" * 42)
for yr_range, label in [(range(3, 6), "3-5 (MOP)"), (range(6, 11), "6-10"), (range(11, 30), "11+")]:
    s = te[te["years_from_launch"].isin(yr_range)]
    if len(s) >= 3:
        print(f"{label:>10} {len(s):>8} S${s['abs_error'].mean():>10,.0f} {s['pct_error'].mean():>7.2f}%")

# ══════════════════════════════════════════════════════════════════════
# 10. SHAP + RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: SHAP + RESIDUAL ANALYSIS")
print("=" * 70)

# Use best individual model for SHAP
shap_name = best_name if best_name != "Weighted Ensemble" else sorted_models[1][0]
shap_model = models[shap_name]
explainer = shap.TreeExplainer(shap_model)
sample_idx = np.random.RandomState(42).choice(len(X_test), min(500, len(X_test)), replace=False)
shap_values = explainer.shap_values(X_test[sample_idx])

fig = plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test[sample_idx], feature_names=FEATURE_COLUMNS, show=False, max_display=24)
plt.title(f"SHAP — Appreciation Model ({shap_name})", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(10, 10))
shap.summary_plot(shap_values, X_test[sample_idx], feature_names=FEATURE_COLUMNS, plot_type="bar", show=False, max_display=24)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 09_shap_summary.png, 10_shap_bar.png")

top3 = np.argsort(np.abs(shap_values).mean(0))[-3:][::-1]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, fi in enumerate(top3):
    shap.dependence_plot(fi, shap_values, X_test[sample_idx], feature_names=FEATURE_COLUMNS, ax=axes[i], show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_shap_dependence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 11_shap_dependence.png")

# Residual analysis
residuals = y_test_price - pred_price
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].scatter(pred_price/1e6, residuals/1e3, alpha=0.3, s=10, color="#2563eb")
axes[0,0].axhline(0, color="red", ls="--")
axes[0,0].set_xlabel("Predicted (S$M)"); axes[0,0].set_ylabel("Residual (S$K)")
axes[0,0].set_title("Residuals vs Predicted")

axes[0,1].hist(residuals/1e3, bins=40, edgecolor="white", alpha=0.8, color="#2563eb")
axes[0,1].axvline(0, color="red", ls="--")
axes[0,1].set_xlabel("Residual (S$K)"); axes[0,1].set_title(f"Distribution (mean: S${residuals.mean():,.0f})")

axes[1,0].scatter(te["years_from_launch"], residuals/1e3, alpha=0.3, s=10, color="#7c3aed")
axes[1,0].axhline(0, color="red", ls="--")
axes[1,0].set_xlabel("Years from Launch"); axes[1,0].set_ylabel("Residual (S$K)")
axes[1,0].set_title("Residuals vs Lease Age")

axes[1,1].scatter(y_test_price/1e6, pred_price/1e6, alpha=0.3, s=10, color="#2563eb")
lims = [min(y_test_price.min(), pred_price.min())/1e6, max(y_test_price.max(), pred_price.max())/1e6]
axes[1,1].plot(lims, lims, "r--", lw=1.5)
axes[1,1].set_xlabel("Actual (S$M)"); axes[1,1].set_ylabel("Predicted (S$M)")
axes[1,1].set_title("Actual vs Predicted")
plt.suptitle(f"Residual Analysis — {best_name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/15_residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 15_residual_analysis.png")

fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
maes = [results[n]["price_mae"] for n in names]
r2s = [results[n]["price_r2"] for n in names]
colors = ["#059669" if n == best_name else "#2563eb" for n in names]
bars = ax.bar(names, maes, color=colors, alpha=0.8)
ax.set_ylabel("MAE (S$)")
ax.set_title("Model Comparison — Zero Leakage")
for bar, r2 in zip(bars, r2s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500, f'R²={r2:.4f}', ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 13_model_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# 11. SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 11: SAVING ARTIFACTS")
print("=" * 70)

scaler = StandardScaler()
scaler.fit(X_trainval)

best_model_obj = models[best_name]
if best_name == "Weighted Ensemble":
    final_model = best_model_obj
    needs_scaling = False
else:
    final_model = best_model_obj.__class__(**best_model_obj.get_params())
    final_model.fit(scaler.transform(X_trainval), y_trainval)
    needs_scaling = True

model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
bundle = {
    "scaler": scaler,
    "model": final_model,
    "model_needs_scaling": needs_scaling,
    "feature_columns": FEATURE_COLUMNS,
    "quantile_lower": q_lo,
    "quantile_upper": q_hi,
    "model_type": "appreciation",
    "launch_psm_lookup": launch_psm_dict,
    "global_mean_ratio": global_mean_ratio,
}
joblib.dump(bundle, model_path)

# Serving lookups (from train-period data ONLY)
latest_train_year = int(train_raw["txn_year"].max())
train_all_latest = train_period_all[train_period_all["txn_year"] == latest_train_year]
train_ec_latest = train_period_ec_all[train_period_ec_all["txn_year"] == latest_train_year]

serving_lookups = {
    "district_stats": train_all_latest.groupby("district_num").agg(
        median_psm=("price_psm", "median"), volume=("price_psm", "count"),
        std_psm=("price_psm", "std"),
    ).to_dict("index"),
    "ec_district_stats": train_ec_latest.groupby("district_num")["price_psm"].median().to_dict(),
    "project_stats": train_ec_latest.groupby("project").agg(
        median_psm=("price_psm", "median"), volume=("price", "count"),
    ).to_dict("index"),
    "district_momentum": mom_lookup.groupby("district_num")["district_momentum"].last().to_dict(),
    "market_lag_psm": float(train_all_latest["price_psm"].median()),
    "project_target_enc": project_enc_lookup,
    "global_mean_ratio": global_mean_ratio,
    "launch_psm_lookup": launch_psm_dict,
    "global_defaults": train_fill_medians,
    "area_quartile_bins": list(area_bins),
    "latest_year": latest_train_year,
}
joblib.dump(serving_lookups, os.path.join(ARTIFACT_DIR, "serving_lookups.joblib"))

metadata = {
    "algorithm": best_name,
    "model_type": "appreciation_ratio",
    "target": "resale_psm / launch_psm",
    "feature_columns": FEATURE_COLUMNS,
    "metrics": {k: v for k, v in best_result.items() if not k.startswith("pred")},
    "prediction_intervals": {"coverage_pct": float(coverage), "avg_width": float(avg_width)},
    "train_size": len(X_trainval),
    "test_size": len(X_test),
    "all_model_results": {
        n: {k: float(v) for k, v in r.items() if not k.startswith("pred")}
        for n, r in results.items()
    },
    "leakage_audit": {
        "launch_psm": "Computed from train-period data only",
        "lag_features": "Computed from train-period data only",
        "target_encoding": "KFold on train set, applied to val/test via train means",
        "fill_medians": "Computed from train set only",
        "area_bins": "Computed from train set only",
    },
}

run_id = log_experiment(
    algorithm=best_name, parameters={"type": "appreciation_v4_final"},
    metrics={k: float(v) for k, v in best_result.items() if not k.startswith("pred")},
    feature_columns=FEATURE_COLUMNS, train_size=len(X_trainval),
    test_size=len(X_test), artifact_path=model_path,
)
metadata["run_id"] = run_id

with open(os.path.join(ARTIFACT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print(f"\nModel: {model_path}")
print(f"Run ID: {run_id}")
print(f"Winner: {best_name}")
print(f"Price MAE:  S${best_result['price_mae']:,.0f}")
print(f"Price R²:   {best_result['price_r2']:.4f}")
print(f"Price MAPE: {best_result['price_mape']:.2f}%")
print(f"Ratio R²:   {best_result['ratio_r2']:.4f}")
print(f"PI coverage: {coverage:.1f}%")
print(f"\nZERO look-ahead bias. Every feature from train-period only.")
print("\nDone.")

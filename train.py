"""
Train reward-regression models on the counterfactual (all-9-candidate)
dataset, with raw flow_reward as the target (NOT the staleness-shaped
value), and select the winner using decision-quality metrics computed
over the true 9-candidate groups: mean policy value, mean regret, and
top-1 action accuracy -- overall and per category. RMSE/MAE/Spearman are
kept as secondary diagnostics only.
"""

from __future__ import annotations

import json
import os

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from features import CATEGORICAL_COLS, FEATURE_COLS, GROUP_COL, TARGET_COL

DATA_PATH = "data/aiko_counterfactual_training_data.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
N_FOLDS = 3
RANDOM_STATE = 7
N_ESTIMATORS = 300
EARLY_STOP = 25


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return df


def decision_level_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    """
    Group predictions back into their 9-candidate decisions and compute
    policy value / regret / top-1 accuracy, overall and per category of the
    TRUE best action (so we can see, e.g., 'when memory was truly best, did
    we pick it').
    """
    out = {}
    grouped = df.groupby("decision_id")

    picked_rewards = []
    best_rewards = []
    regrets = []
    top1_hits = []
    true_best_categories = []
    picked_categories = []

    for _, g in grouped:
        best_idx = g["reward"].idxmax()
        pred_idx = g[pred_col].idxmax()
        best_reward = g.loc[best_idx, "reward"]
        picked_reward = g.loc[pred_idx, "reward"]

        picked_rewards.append(picked_reward)
        best_rewards.append(best_reward)
        regrets.append(best_reward - picked_reward)
        top1_hits.append(int(best_idx == pred_idx))
        true_best_categories.append(g.loc[best_idx, "game_category"])
        picked_categories.append(g.loc[pred_idx, "game_category"])

    out["mean_policy_value"] = float(np.mean(picked_rewards))
    out["mean_oracle_value"] = float(np.mean(best_rewards))
    out["mean_regret"] = float(np.mean(regrets))
    out["median_regret"] = float(np.median(regrets))
    out["p90_regret"] = float(np.percentile(regrets, 90))
    out["top1_accuracy"] = float(np.mean(top1_hits))

    cat_df = pd.DataFrame(
        {
            "regret": regrets,
            "top1_hit": top1_hits,
            "true_best_category": true_best_categories,
            "picked_category": picked_categories,
        }
    )
    by_true_best_cat = cat_df.groupby("true_best_category").agg(
        mean_regret=("regret", "mean"),
        top1_accuracy=("top1_hit", "mean"),
        n=("regret", "size"),
    )
    out["by_true_best_category"] = by_true_best_cat.round(4).to_dict(orient="index")
    out["picked_category_distribution"] = (
        cat_df["picked_category"].value_counts(normalize=True).round(4).to_dict()
    )
    out["true_best_category_distribution"] = (
        cat_df["true_best_category"].value_counts(normalize=True).round(4).to_dict()
    )
    return out


def pointwise_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    spearman = float(spearmanr(y_true, y_pred).correlation)
    return {"rmse": rmse, "mae": mae, "spearman": spearman}


def train_lightgbm(X_tr, y_tr, X_val, y_val, cat_cols):
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_xgboost(X_tr, y_tr, X_val, y_val, cat_cols):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_val[c] = X_val[c].astype("category")
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOP,
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_catboost(X_tr, y_tr, X_val, y_val, cat_cols):
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)
    model = cb.CatBoostRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        cat_features=cat_cols,
        early_stopping_rounds=EARLY_STOP,
        verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return model


def predict(model, X, lib, cat_cols):
    X = X.copy()
    if lib == "xgboost":
        for c in cat_cols:
            X[c] = X[c].astype("category")
        return model.predict(X)
    if lib == "catboost":
        for c in cat_cols:
            X[c] = X[c].astype(str)
        return model.predict(X)
    return model.predict(X)


def run():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_data()
    groups = df[GROUP_COL]
    gkf = GroupKFold(n_splits=N_FOLDS)

    trainers = {"lightgbm": train_lightgbm, "xgboost": train_xgboost, "catboost": train_catboost}
    libs = list(trainers.keys())

    oof_preds = {lib: np.zeros(len(df)) for lib in libs}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        X_tr, y_tr = df.iloc[tr_idx][FEATURE_COLS], df.iloc[tr_idx][TARGET_COL]
        X_val, y_val = df.iloc[val_idx][FEATURE_COLS], df.iloc[val_idx][TARGET_COL]

        for lib in libs:
            model = trainers[lib](X_tr, y_tr, X_val, y_val, CATEGORICAL_COLS)
            oof_preds[lib][val_idx] = predict(model, X_val, lib, CATEGORICAL_COLS)
        print(f"fold {fold} done")

    report = {}
    for lib in libs:
        df[f"pred_{lib}"] = oof_preds[lib]
        pw = pointwise_metrics(df[TARGET_COL].values, oof_preds[lib])
        dl = decision_level_metrics(df, f"pred_{lib}")
        report[lib] = {"pointwise": pw, "decision_level": dl}

    # selection criterion: highest mean_policy_value on OOF predictions
    # (equivalently, lowest mean_regret) -- NOT grouped rank concordance.
    best_lib = max(report, key=lambda k: report[k]["decision_level"]["mean_policy_value"])

    # also report the oracle and the ORIGINAL observed-only-trained baseline
    # for comparison, and what "always predict global mean" (a naive
    # baseline) would score, to sanity check the metrics are meaningful.
    naive_pred = np.full(len(df), df[TARGET_COL].mean())
    df["pred_naive"] = naive_pred
    report["naive_mean_baseline"] = {
        "decision_level": decision_level_metrics(df, "pred_naive"),
    }

    with open(f"{REPORT_DIR}/model_selection_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== Decision-level summary (out-of-fold) ===")
    for lib in libs:
        dl = report[lib]["decision_level"]
        print(
            f"{lib:10s}  policy_value={dl['mean_policy_value']:.4f}  "
            f"regret={dl['mean_regret']:.4f}  top1_acc={dl['top1_accuracy']:.4f}  "
            f"rmse={report[lib]['pointwise']['rmse']:.4f}"
        )
    print(f"\nnaive mean baseline policy_value="
          f"{report['naive_mean_baseline']['decision_level']['mean_policy_value']:.4f}")
    print(f"oracle (true best every time) policy_value="
          f"{report[libs[0]]['decision_level']['mean_oracle_value']:.4f}")
    print(f"\nBEST MODEL by policy value: {best_lib}")

    # ---- refit the winning model on ALL data for deployment ----
    X_all, y_all = df[FEATURE_COLS], df[TARGET_COL]
    # simple holdout carve-out from the full data for early stopping on the
    # final refit (not used anywhere else)
    gkf2 = GroupKFold(n_splits=N_FOLDS)
    tr_idx, val_idx = next(gkf2.split(df, groups=groups))
    final_model = trainers[best_lib](
        X_all.iloc[tr_idx], y_all.iloc[tr_idx], X_all.iloc[val_idx], y_all.iloc[val_idx], CATEGORICAL_COLS
    )

    import joblib

    joblib.dump({"model": final_model, "lib": best_lib, "feature_cols": FEATURE_COLS, "cat_cols": CATEGORICAL_COLS},
                f"{MODEL_DIR}/reward_model.joblib")

    print(f"\nSaved winning model ({best_lib}) to {MODEL_DIR}/reward_model.joblib")
    return report, best_lib


if __name__ == "__main__":
    run()

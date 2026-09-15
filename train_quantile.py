"""
Approach 1 -- quantile regression: predict an upper quantile of reward
(e.g. the 75th percentile) instead of the mean, and argmax on that. This
targets the differential-shrinkage problem directly: a quantile loss does
not average away the high end of a noisy distribution the way squared-error
does.

Approach 3 -- UCB-style uncertainty-aware selection: predict q10/q50/q90
per candidate and score with `q50 + k * (q90 - q10)`, i.e. reward candidates
proportionally to how much upside their own predicted uncertainty carries.
This is meant to give memory/attention (the noisier categories) credit for
their uncertainty rather than being penalized for it, but calibrated by the
model's own per-context uncertainty rather than a fixed quota.

Both are evaluated with real GroupKFold OOF predictions (no leakage) using
the same decision-level metrics as the baseline regressor.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import CATEGORICAL_COLS, FEATURE_COLS, GROUP_COL, TARGET_COL
from train import DATA_PATH, EARLY_STOP, N_ESTIMATORS, N_FOLDS, RANDOM_STATE, decision_level_metrics

ALPHAS = [0.1, 0.5, 0.75, 0.9]


def train_quantile_model(X_tr, y_tr, X_val, y_val, alpha, cat_cols):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
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


def run():
    df = pd.read_csv(DATA_PATH)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")

    groups = df[GROUP_COL]
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof = {a: np.zeros(len(df)) for a in ALPHAS}

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        X_tr, y_tr = df.iloc[tr_idx][FEATURE_COLS], df.iloc[tr_idx][TARGET_COL]
        X_val = df.iloc[val_idx][FEATURE_COLS]
        for a in ALPHAS:
            model = train_quantile_model(X_tr, y_tr, X_val, df.iloc[val_idx][TARGET_COL], a, CATEGORICAL_COLS)
            oof[a][val_idx] = model.predict(X_val)
        print(f"fold {fold} done")

    for a in ALPHAS:
        df[f"pred_q{int(a*100)}"] = oof[a]

    report = {}

    # --- Approach 1: direct upper-quantile decision rule ---
    for a in [0.75, 0.9]:
        col = f"pred_q{int(a*100)}"
        report[f"quantile_{int(a*100)}"] = decision_level_metrics(df, col)

    # --- Approach 3: UCB-style, sweep k ---
    spread = df["pred_q90"] - df["pred_q10"]
    for k in [0.25, 0.5, 1.0, 1.5]:
        df["pred_ucb"] = df["pred_q50"] + k * spread
        report[f"ucb_k{k}"] = decision_level_metrics(df, "pred_ucb")

    print("\n=== Approach 1: quantile decision rules ===")
    for name in ["quantile_75", "quantile_90"]:
        d = report[name]
        print(f"{name:14s} policy_value={d['mean_policy_value']:.4f} regret={d['mean_regret']:.4f} "
              f"top1={d['top1_accuracy']:.4f}  picked={d['picked_category_distribution']}")

    print("\n=== Approach 3: UCB decision rules ===")
    for k in [0.25, 0.5, 1.0, 1.5]:
        d = report[f"ucb_k{k}"]
        print(f"k={k:<5} policy_value={d['mean_policy_value']:.4f} regret={d['mean_regret']:.4f} "
              f"top1={d['top1_accuracy']:.4f}  picked={d['picked_category_distribution']}")

    with open("reports/quantile_ucb_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # save quantile models fit on a train/val split, for deployment of
    # whichever strategy wins
    import joblib

    tr_idx, val_idx = next(GroupKFold(n_splits=N_FOLDS).split(df, groups=groups))
    models = {}
    for a in ALPHAS:
        models[a] = train_quantile_model(
            df.iloc[tr_idx][FEATURE_COLS], df.iloc[tr_idx][TARGET_COL],
            df.iloc[val_idx][FEATURE_COLS], df.iloc[val_idx][TARGET_COL],
            a, CATEGORICAL_COLS,
        )
    joblib.dump(
        {"models": models, "feature_cols": FEATURE_COLS, "cat_cols": CATEGORICAL_COLS},
        "models/quantile_models.joblib",
    )
    print("\nSaved quantile models to models/quantile_models.joblib")
    return report


if __name__ == "__main__":
    run()

"""
Train a LightGBM LambdaRank ranker over each 9-candidate decision group,
as a candidate fix for the differential-shrinkage category bias found in
the pointwise regressors (see reports/model_selection_report.json): all
three regressors under-predict noisier, high-value candidates (memory,
attention) more than the less-noisy problem_solving candidates, which
biases argmax toward problem_solving regardless of the true best category.

A ranking loss only needs the relative order of the 9 candidates to be
right -- it never tries to match the exact noisy peak value -- so it
should not inherit the same bias. We evaluate this empirically with the
same decision-level metrics used for the regressors, rather than assuming
it will help.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import CATEGORICAL_COLS, FEATURE_COLS, GROUP_COL, TARGET_COL
from train import DATA_PATH, EARLY_STOP, N_ESTIMATORS, N_FOLDS, RANDOM_STATE, decision_level_metrics


def make_rank_labels(df: pd.DataFrame) -> pd.Series:
    # integer rank 0..8 within each decision group (8 = best true reward).
    # Using rank rather than raw reward as the relevance label makes the
    # objective purely about relative order, robust to the reward's noisy
    # absolute scale.
    return df.groupby("decision_id")[TARGET_COL].rank(method="first").astype(int) - 1


def fold_groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("decision_id", sort=False).size().values


def run():
    df = pd.read_csv(DATA_PATH).sort_values("decision_id").reset_index(drop=True)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    df["rank_label"] = make_rank_labels(df)

    groups = df[GROUP_COL]
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(df))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        tr_df = df.iloc[tr_idx].sort_values("decision_id")
        val_df = df.iloc[val_idx].sort_values("decision_id")

        X_tr, y_tr = tr_df[FEATURE_COLS], tr_df["rank_label"]
        X_val, y_val = val_df[FEATURE_COLS], val_df["rank_label"]
        group_tr = fold_groups(tr_df)
        group_val = fold_groups(val_df)

        model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        model.fit(
            X_tr,
            y_tr,
            group=group_tr,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=[1],
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
        )
        preds = model.predict(X_val)
        oof_pred[val_df.index.values] = preds
        print(f"fold {fold} done")

    df["pred_ranker"] = oof_pred
    dl = decision_level_metrics(df, "pred_ranker")

    print("\n=== LambdaRank decision-level summary (out-of-fold) ===")
    print(f"policy_value={dl['mean_policy_value']:.4f}  regret={dl['mean_regret']:.4f}  "
          f"top1_acc={dl['top1_accuracy']:.4f}")
    print("picked_category_distribution:", dl["picked_category_distribution"])
    print("true_best_category_distribution:", dl["true_best_category_distribution"])
    print("by_true_best_category:")
    for k, v in dl["by_true_best_category"].items():
        print(" ", k, v)

    with open("reports/ranker_report.json", "w") as f:
        json.dump({"lightgbm_ranker": {"decision_level": dl}}, f, indent=2, default=str)

    # refit on a train/val split for deployment
    tr_idx, val_idx = next(GroupKFold(n_splits=N_FOLDS).split(df, groups=groups))
    tr_df = df.iloc[tr_idx].sort_values("decision_id")
    val_df = df.iloc[val_idx].sort_values("decision_id")
    final_model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    final_model.fit(
        tr_df[FEATURE_COLS],
        tr_df["rank_label"],
        group=fold_groups(tr_df),
        eval_set=[(val_df[FEATURE_COLS], val_df["rank_label"])],
        eval_group=[fold_groups(val_df)],
        eval_at=[1],
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(0)],
    )

    import joblib

    joblib.dump(
        {"model": final_model, "lib": "lightgbm_ranker", "feature_cols": FEATURE_COLS, "cat_cols": CATEGORICAL_COLS},
        "models/ranker_model.joblib",
    )
    print("\nSaved ranker model to models/ranker_model.joblib")
    return dl


if __name__ == "__main__":
    run()

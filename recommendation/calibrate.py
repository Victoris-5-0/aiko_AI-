"""
Approach 2 -- category-conditional post-hoc calibration.

The baseline pointwise regressor under-predicts high-value outcomes more
severely for the noisier accuracy-based categories (memory, attention) than
for problem_solving (see README). Fit one IsotonicRegression per category,
mapping raw predicted reward -> a recalibrated value, using genuinely
held-out data (half of the OOF decisions) so this isn't circular.

Decision rule: argmax over the recalibrated, category-specific value.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from recommendation.train import decision_level_metrics

OOF_PATH = "data/oof_predictions.csv"
BASE_PRED_COL = "pred_catboost"  # the winning baseline model from train.py


def run():
    df = pd.read_csv(OOF_PATH)

    # split by decision_id (not row) so a whole 9-candidate group stays together
    decision_ids = df["decision_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(decision_ids)
    half = len(decision_ids) // 2
    fit_ids, eval_ids = set(decision_ids[:half]), set(decision_ids[half:])

    fit_df = df[df["decision_id"].isin(fit_ids)]
    eval_df = df[df["decision_id"].isin(eval_ids)].copy()

    calibrators = {}
    for category, g in fit_df.groupby("game_category"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(g[BASE_PRED_COL], g["reward"])
        calibrators[category] = iso

    eval_df["pred_calibrated"] = 0.0
    for category, iso in calibrators.items():
        mask = eval_df["game_category"] == category
        eval_df.loc[mask, "pred_calibrated"] = iso.predict(eval_df.loc[mask, BASE_PRED_COL])

    dl_baseline = decision_level_metrics(eval_df, BASE_PRED_COL)
    dl_calibrated = decision_level_metrics(eval_df, "pred_calibrated")

    print("=== baseline (uncalibrated) on calibration-eval half ===")
    print(f"policy_value={dl_baseline['mean_policy_value']:.4f} regret={dl_baseline['mean_regret']:.4f} "
          f"top1={dl_baseline['top1_accuracy']:.4f}")
    print("picked_category_distribution:", dl_baseline["picked_category_distribution"])

    print("\n=== calibrated on calibration-eval half ===")
    print(f"policy_value={dl_calibrated['mean_policy_value']:.4f} regret={dl_calibrated['mean_regret']:.4f} "
          f"top1={dl_calibrated['top1_accuracy']:.4f}")
    print("picked_category_distribution:", dl_calibrated["picked_category_distribution"])

    # refit calibrators on ALL OOF data for deployment (still leakage-free
    # w.r.t. the original training folds -- OOF predictions were never seen
    # by the models that produced them)
    deploy_calibrators = {}
    for category, g in df.groupby("game_category"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(g[BASE_PRED_COL], g["reward"])
        deploy_calibrators[category] = iso

    joblib.dump(deploy_calibrators, "models/category_calibrators.joblib")

    report = {
        "baseline_on_eval_half": dl_baseline,
        "calibrated_on_eval_half": dl_calibrated,
    }
    with open("reports/calibration_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    run()

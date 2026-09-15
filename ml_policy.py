"""
ML inference policy.

Uses features.build_learner_context / build_candidate_row -- the exact
same functions used to build the training data -- so there is no
feature-construction skew between training and serving.

IMPORTANT train/serve parity point: `current_level_*` is a piece of
observable state that the training data generator updates via
`update_behavior_policy_state` (PASS_THRESHOLD=0.50 / HARD_FAIL_THRESHOLD
=0.20) after every real play, regardless of which action was taken. The ML
policy at serving time must update its own child's level state the SAME
way, not with the rule engine's separate 0.85/0.55 ratchet -- otherwise
`current_level_*` would mean something different at serve time than it
meant during training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulator import GAMES, Child, update_behavior_policy_state
from features import ALL_CANDIDATES, build_candidate_row, build_learner_context


def decide_action_ml(child: Child, timestamp, bundle: dict) -> tuple[str, int, str]:
    if child.onboarding_queue:
        game_id = child.onboarding_queue.pop(0)
        return game_id, 1, GAMES[game_id]["category"]

    learner_ctx = build_learner_context(child, timestamp)
    rows = [build_candidate_row(learner_ctx, gid, d) for gid, d in ALL_CANDIDATES]
    X = pd.DataFrame(rows)[bundle["feature_cols"]]

    lib = bundle["lib"]
    for c in bundle["cat_cols"]:
        if lib == "catboost":
            X[c] = X[c].astype(str)
        else:
            X[c] = X[c].astype("category")

    if bundle.get("type") == "ucb":
        # UCB-style combination of q10/q50/q90 models (approach 3)
        q10 = bundle["models"][0.1].predict(X)
        q50 = bundle["models"][0.5].predict(X)
        q90 = bundle["models"][0.9].predict(X)
        preds = q50 + bundle["k"] * (q90 - q10)
    else:
        preds = bundle["model"].predict(X)

    # optional post-hoc, category-conditional recalibration (approach 2)
    calibrators = bundle.get("calibrators")
    if calibrators:
        categories = [row["game_category"] for row in rows]
        preds = np.array(
            [calibrators[cat].predict([p])[0] if cat in calibrators else p for p, cat in zip(preds, categories)]
        )

    best_idx = int(preds.argmax())
    game_id, difficulty = ALL_CANDIDATES[best_idx]
    category = GAMES[game_id]["category"]
    return game_id, difficulty, category


def ml_update_level(child: Child, category: str, score: float) -> None:
    """Same update rule the training-data generator used -- keeps
    current_level_* consistent between train and serve time."""
    update_behavior_policy_state(child, category, score)

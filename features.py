"""
Shared feature engineering.

CRITICAL: this module is imported by BOTH generate_data.py (to build training
rows) and ml_policy.py (to build live scoring rows). Using the exact same
functions in both places is what guarantees train/serve parity for
`current_level_*`, rolling stats, the estimated-ability proxy, etc.

No function here ever reads Child.ability, Child.learning_rate, or
Child.fatigue_factor (the hidden simulator ground truth). Everything is
derived from observable history only.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from simulator import (
    CATEGORIES,
    CATEGORY_METRIC,
    DIFFICULTY_B,
    DISCRIMINATION_EASY,
    GAME_BY_CATEGORY,
    GAME_IDS,
    GAMES,
    ROLLING_WINDOWS,
    Child,
    logit,
)

# ---------------------------------------------------------------------------
# Rolling / history helpers
# ---------------------------------------------------------------------------


def _metric_key(category: str) -> str:
    return "accuracy_score" if CATEGORY_METRIC[category] == "accuracy" else "efficiency_score"


def rolling_values(history, category, n):
    key = _metric_key(category)
    vals = [r[key] for r in history if r["game_category"] == category and pd.notna(r[key])]
    return vals[-n:]


def rolling_mean(history, category, n):
    vals = rolling_values(history, category, n)
    return float(np.mean(vals)) if vals else np.nan


def rolling_std(history, category, n):
    vals = rolling_values(history, category, n)
    return float(np.std(vals)) if len(vals) >= 2 else np.nan


def rolling_mean_by_game(history, game_id, n):
    cfg = GAMES[game_id]
    key = "accuracy_score" if cfg["metric"] == "accuracy" else "efficiency_score"
    vals = [r[key] for r in history if r["game_id"] == game_id and pd.notna(r[key])]
    vals = vals[-n:]
    return float(np.mean(vals)) if vals else np.nan


def weakest_category(history, excluded: set) -> str:
    eligible = [c for c in CATEGORIES if c not in excluded] or list(CATEGORIES)
    scores = {}
    for c in eligible:
        m = rolling_mean(history, c, 5)
        scores[c] = m if not np.isnan(m) else -1.0
    return min(scores, key=scores.get)


def pick_game_in_category(history, category) -> str:
    candidates = GAME_BY_CATEGORY[category]
    if len(candidates) == 1:
        return candidates[0]

    def game_score(gid):
        v = rolling_mean_by_game(history, gid, 5)
        return v if not np.isnan(v) else -1.0

    return min(candidates, key=game_score)


# ---------------------------------------------------------------------------
# Estimated ability (observable proxy for hidden theta — NOT ground truth)
# ---------------------------------------------------------------------------


def estimate_theta(history, category) -> float:
    """
    Statistical point estimate of the child's theta in `category`, derived
    only from observed (difficulty, score) pairs in history. Uses a simple
    inverse-IRT approximation (assumes the "easy" discrimination constant,
    which is an approximation but a reasonable, stable proxy — this is a
    feature, not a claim of ground truth).

    Falls back to 0.0 (== ability 0.5) when there is no history yet.
    """
    metric = CATEGORY_METRIC[category]
    key = "accuracy_score" if metric == "accuracy" else "efficiency_score"
    obs = [
        (r["difficulty"], r[key])
        for r in history
        if r["game_category"] == category and pd.notna(r[key])
    ]
    obs = obs[-10:]  # recency-weighted via truncation, simple and stable
    if not obs:
        return 0.0
    thetas = []
    for difficulty, score in obs:
        b = DIFFICULTY_B[difficulty]
        thetas.append(b + logit(score) / DISCRIMINATION_EASY)
    return float(np.mean(thetas))


# ---------------------------------------------------------------------------
# Learner-state (context) features — identical across all candidates for a
# given decision point.
# ---------------------------------------------------------------------------


def build_learner_context(child: Child, timestamp: datetime) -> dict:
    feats = {
        "age": child.age,
        "plays_in_session": child.plays_in_session,
        "is_onboarding": int(bool(child.onboarding_queue)),
        "last_game_category": (child.history[-1]["game_category"] if child.history else "none"),
    }

    for gid in GAME_IDS:
        idx = child.game_last_index.get(gid)
        feats[f"games_since_played_{gid}"] = (
            child.games_played_count - idx if idx is not None else np.nan
        )

    for cat in CATEGORIES:
        idx = child.category_last_index.get(cat)
        since_played = child.games_played_count - idx if idx is not None else np.nan
        feats[f"category_since_played_{cat}"] = since_played
        feats[f"is_first_play_{cat}"] = int(idx is None)

        for n in ROLLING_WINDOWS:
            feats[f"rolling_{cat}_last{n}"] = rolling_mean(child.history, cat, n)

        feats[f"rolling_std_{cat}_last5"] = rolling_std(child.history, cat, 5)
        last3 = feats[f"rolling_{cat}_last3"]
        last10 = feats[f"rolling_{cat}_last10"]
        feats[f"trend_{cat}"] = (
            last3 - last10 if not (np.isnan(last3) or np.isnan(last10)) else 0.0
        )

        feats[f"sessions_played_{cat}"] = len(child.category_sessions_played[cat])

        last_ts = child.category_last_ts.get(cat)
        feats[f"days_since_last_played_{cat}"] = (
            (timestamp - last_ts).total_seconds() / 86400.0 if last_ts is not None else np.nan
        )

        feats[f"current_level_{cat}"] = child.category_level[cat]
        feats[f"estimated_theta_{cat}"] = estimate_theta(child.history, cat)

    return feats


# ---------------------------------------------------------------------------
# Candidate-action + interaction features
# ---------------------------------------------------------------------------


def build_candidate_row(learner_ctx: dict, game_id: str, difficulty: int) -> dict:
    cfg = GAMES[game_id]
    category = cfg["category"]
    b = DIFFICULTY_B[difficulty]
    est_theta = learner_ctx[f"estimated_theta_{category}"]
    current_level = learner_ctx[f"current_level_{category}"]

    row = dict(learner_ctx)
    row.update(
        {
            "game_id": game_id,
            "game_category": category,
            "difficulty": difficulty,
            "difficulty_b": b,
            "theta_minus_difficulty_b": est_theta - b,
            "abs_theta_minus_difficulty_b": abs(est_theta - b),
            "candidate_level_diff": difficulty - current_level,
            "candidate_matches_current_level": int(difficulty == current_level),
            "candidate_category_staleness": learner_ctx[f"category_since_played_{category}"],
            "candidate_category_trend": learner_ctx[f"trend_{category}"],
        }
    )
    return row


ALL_CANDIDATES = [(gid, d) for gid in GAME_IDS for d in (1, 2, 3)]


FEATURE_COLS = [
    "game_id",
    "game_category",
    "difficulty",
    "difficulty_b",
    "theta_minus_difficulty_b",
    "abs_theta_minus_difficulty_b",
    "candidate_level_diff",
    "candidate_matches_current_level",
    "candidate_category_staleness",
    "candidate_category_trend",
    "age",
    "is_onboarding",
    "plays_in_session",
    "last_game_category",
] + [f"games_since_played_{g}" for g in GAME_IDS] + sum(
    (
        [
            f"category_since_played_{c}",
            f"is_first_play_{c}",
            f"rolling_{c}_last3",
            f"rolling_{c}_last5",
            f"rolling_{c}_last10",
            f"rolling_std_{c}_last5",
            f"trend_{c}",
            f"sessions_played_{c}",
            f"days_since_last_played_{c}",
            f"current_level_{c}",
            f"estimated_theta_{c}",
        ]
        for c in CATEGORIES
    ),
    [],
)

CATEGORICAL_COLS = ["game_id", "game_category", "last_game_category"]
TARGET_COL = "reward"
GROUP_COL = "learner_id"

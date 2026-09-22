import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from recommendation.simulator import (
    GAMES,
    CATEGORIES,
    GAME_IDS,
    DIFFICULTIES,
)
from recommendation.features import build_candidate_row

logger = logging.getLogger("recommendation")

router = APIRouter()

ALL_CANDIDATES = [
    (game_id, difficulty)
    for game_id in GAME_IDS
    for difficulty in DIFFICULTIES
]

VALID_LAST_GAME_CATEGORY = set(CATEGORIES) | {"none"}

# ============================================================
# Model
# ============================================================

# Quantile-75 is the deployed policy (chosen for category balance --
# 14.4% memory coverage vs. 3.2% for the mean-regression baseline, at a
# measured, accepted reward trade-off). Do NOT point this back at
# reward_model.joblib (the mean baseline) without re-approving that
# trade-off first.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "deployed_model.joblib"

try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    bundle = None
    logger.error(f"Could not load model from {MODEL_PATH}: {e}")


# ============================================================
# Request schema
# ============================================================

class LearnerFeatures(BaseModel):
    """
    The 33 learner-state columns not already top-level on
    RecommendationRequest (age, plays_in_session, last_game_category) and
    not part of current_level. Field names match features.py:FEATURE_COLS
    exactly. Nulls are accepted only where BACKEND_FEATURE_SPEC.md allows
    them (first play in a category / game never played).
    """

    games_since_played_prob_maze: Optional[int] = None
    games_since_played_att_odd_one_out: Optional[int] = None
    games_since_played_mem_recognition: Optional[int] = None

    category_since_played_attention: Optional[int] = None
    category_since_played_memory: Optional[int] = None
    category_since_played_problem_solving: Optional[int] = None

    is_first_play_attention: int
    is_first_play_memory: int
    is_first_play_problem_solving: int

    rolling_attention_last3: Optional[float] = None
    rolling_memory_last3: Optional[float] = None
    rolling_problem_solving_last3: Optional[float] = None

    rolling_attention_last5: Optional[float] = None
    rolling_memory_last5: Optional[float] = None
    rolling_problem_solving_last5: Optional[float] = None

    rolling_attention_last10: Optional[float] = None
    rolling_memory_last10: Optional[float] = None
    rolling_problem_solving_last10: Optional[float] = None

    rolling_std_attention_last5: Optional[float] = None
    rolling_std_memory_last5: Optional[float] = None
    rolling_std_problem_solving_last5: Optional[float] = None

    trend_attention: float = 0.0
    trend_memory: float = 0.0
    trend_problem_solving: float = 0.0

    sessions_played_attention: int
    sessions_played_memory: int
    sessions_played_problem_solving: int

    days_since_last_played_attention: Optional[float] = None
    days_since_last_played_memory: Optional[float] = None
    days_since_last_played_problem_solving: Optional[float] = None

    estimated_theta_attention: float = 0.0
    estimated_theta_memory: float = 0.0
    estimated_theta_problem_solving: float = 0.0


class RecommendationRequest(BaseModel):
    learner_id: str
    # Accepted for logging/audit only. days_since_last_played_* is computed
    # by the backend and sent inside `features`, so timestamp does not feed
    # into any feature computation here.
    timestamp: str
    age: int
    plays_in_session: int = 0
    last_game_category: str = "none"
    features: LearnerFeatures
    current_level: Dict[str, int]

    @validator("last_game_category")
    def _valid_last_game_category(cls, v):
        if v not in VALID_LAST_GAME_CATEGORY:
            raise ValueError(
                f"last_game_category must be one of {sorted(VALID_LAST_GAME_CATEGORY)}, got {v!r}"
            )
        return v

    @validator("current_level")
    def _valid_current_level(cls, v):
        missing = [c for c in CATEGORIES if c not in v]
        if missing:
            raise ValueError(f"current_level is missing required categories: {missing}")
        bad = {c: lvl for c, lvl in v.items() if lvl not in (1, 2, 3)}
        if bad:
            raise ValueError(f"current_level values must be 1, 2 or 3, got: {bad}")
        return v


# ============================================================
# Health check
# ============================================================

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_path": str(MODEL_PATH),
    }


# ============================================================
# Recommendation
# ============================================================

@router.post("/recommend")
def recommend(request: RecommendationRequest):

    if bundle is None:
        raise HTTPException(
            status_code=500,
            detail="ML model is not loaded.",
        )

    # ----------------------------------------------------
    # 1. Onboarding check -- bypass the model entirely.
    #
    # The model was trained on data where the first play of each of the
    # 3 games was ALWAYS forced (never chosen by the reward model), so
    # calling the model for a learner who hasn't played all 3 games yet
    # would feed it a context distribution it never saw in training.
    # "Never played" is read from games_since_played_<game> being null,
    # rather than trusting a client-supplied is_onboarding flag.
    # ----------------------------------------------------

    games_since_played = {
        "prob_maze": request.features.games_since_played_prob_maze,
        "att_odd_one_out": request.features.games_since_played_att_odd_one_out,
        "mem_recognition": request.features.games_since_played_mem_recognition,
    }
    unplayed_games = [g for g in GAME_IDS if games_since_played[g] is None]

    if unplayed_games:
        # Deterministic order (GAME_IDS order) rather than the random
        # shuffle used in training data generation -- onboarding order
        # doesn't affect the reward objective, only which game happens
        # to go first, so determinism here is a reasonable simplification.
        game_id = unplayed_games[0]
        difficulty = 1
        category = GAMES[game_id]["category"]

        logger.info(f"learner {request.learner_id}: onboarding -> {game_id} d{difficulty}")

        return {
            "learner_id": request.learner_id,
            "game_id": game_id,
            "category": category,
            "difficulty": difficulty,
            "predicted_reward": None,
            "is_onboarding": True,
            "candidates": None,
        }

    try:
        # ----------------------------------------------------
        # 2. Build learner context
        # ----------------------------------------------------

        learner_ctx = {
            "age": request.age,
            "plays_in_session": request.plays_in_session,
            "is_onboarding": 0,  # always 0 here -- onboarding returned above
            "last_game_category": request.last_game_category,
        }
        learner_ctx.update(request.features.dict())

        for category in CATEGORIES:
            learner_ctx[f"current_level_{category}"] = request.current_level[category]

        # ----------------------------------------------------
        # 3. Generate candidate rows
        # ----------------------------------------------------

        rows = [
            build_candidate_row(learner_ctx, game_id, difficulty)
            for game_id, difficulty in ALL_CANDIDATES
        ]

        # ----------------------------------------------------
        # 4. Build feature matrix, same column order as training
        # ----------------------------------------------------

        X = pd.DataFrame(rows)
        X = X[bundle["feature_cols"]]

        # ----------------------------------------------------
        # 5. Handle categorical features
        # ----------------------------------------------------

        lib = bundle.get("lib")

        for column in bundle.get("cat_cols", []):
            if lib == "catboost":
                X[column] = X[column].astype(str)
            else:
                X[column] = X[column].astype("category")

        # ----------------------------------------------------
        # 6. Generate predictions (quantile/mean/UCB bundle types supported;
        #    Softmax changes selection policy, not reward prediction)
        # ----------------------------------------------------

        if bundle.get("type") == "ucb":
            q10 = bundle["models"][0.1].predict(X)
            q50 = bundle["models"][0.5].predict(X)
            q90 = bundle["models"][0.9].predict(X)
            predictions = q50 + bundle["k"] * (q90 - q10)
        else:
            predictions = bundle["model"].predict(X)

        # ----------------------------------------------------
        # 7. Apply category calibrators if this bundle uses them
        # ----------------------------------------------------

        calibrators = bundle.get("calibrators")

        if calibrators:
            categories = [row["game_category"] for row in rows]
            predictions = np.array(
                [
                    calibrators[category].predict([prediction])[0]
                    if category in calibrators
                    else prediction
                    for prediction, category in zip(predictions, categories)
                ]
            )

        # ----------------------------------------------------
        # 8. Select candidate
        # ----------------------------------------------------
        # Default policy: deterministic argmax.
        # Softmax policy: sample from a probability distribution derived
        # from the predicted rewards. The model still predicts a reward
        # for every candidate; Softmax only changes how the final action
        # is selected.
        #
        # Expected bundle format for Softmax:
        # {
        #     "type": "softmax",
        #     "T": 0.05,
        #     ...
        # }
        if bundle.get("type") == "softmax":
            T = float(bundle["T"])
            if T <= 0:
                raise ValueError("Softmax temperature T must be > 0.")

            # Subtract max for numerical stability.
            z = (predictions - predictions.max()) / T
            exp_z = np.exp(z)
            probabilities = exp_z / exp_z.sum()

            best_idx = int(
                np.random.choice(
                    len(ALL_CANDIDATES),
                    p=probabilities,
                )
            )
        else:
            best_idx = int(np.argmax(predictions))

        game_id, difficulty = ALL_CANDIDATES[best_idx]
        category = GAMES[game_id]["category"]

        candidates = [
            {
                "game_id": gid,
                "difficulty": diff,
                "predicted_reward": round(float(pred), 4),
            }
            for (gid, diff), pred in zip(ALL_CANDIDATES, predictions)
        ]

        logger.info(
            f"learner {request.learner_id}: recommended {game_id} d{difficulty} "
            f"(reward={predictions[best_idx]:.4f})"
        )

        # ----------------------------------------------------
        # 9. Return recommendation
        # ----------------------------------------------------

        return {
            "learner_id": request.learner_id,
            "game_id": game_id,
            "category": category,
            "difficulty": difficulty,
            "predicted_reward": round(float(predictions[best_idx]), 4),
            "is_onboarding": False,
            "candidates": candidates,
        }

    except KeyError as e:
        # a required feature/category was missing after validation --
        # should be rare given the schema above, but keep this specific
        # so it's obviously a data problem, not a model problem
        logger.exception(f"learner {request.learner_id}: missing field {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Request is missing a required field: {e}",
        )

    except Exception as e:
        # log the real exception server-side for debugging, but don't
        # leak internal details (stack traces, file paths, library
        # internals) to the caller
        logger.exception(f"learner {request.learner_id}: recommendation failed")
        raise HTTPException(
            status_code=500,
            detail="Recommendation failed due to an internal error.",
        )

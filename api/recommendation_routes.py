from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from recommendation.simulator import (
    GAMES,
    CATEGORIES,
    GAME_IDS,
    DIFFICULTIES,
)
from recommendation.features import build_candidate_row


router = APIRouter()

ALL_CANDIDATES = [
    (game_id, difficulty)
    for game_id in GAME_IDS
    for difficulty in DIFFICULTIES
]

# ============================================================
# Model
# ============================================================


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "reward_model.joblib"


try:
    bundle = joblib.load(MODEL_PATH)
except Exception as e:
    bundle = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")


# ============================================================
# Request Schema
# ============================================================

class RecommendationRequest(BaseModel):
    learner_id: str
    timestamp: str
    age: int
    plays_in_session: int = 0
    last_game_category: str = "none"
    features: Dict[str, Any]
    current_level: Dict[str, int]


# ============================================================
# Health Check
# ============================================================

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
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

    try:

        # ----------------------------------------------------
        # 1. Build learner context
        # ----------------------------------------------------

        learner_ctx = {
            "age": request.age,
            "plays_in_session": request.plays_in_session,
            "is_onboarding": 0,
            "last_game_category": request.last_game_category,
        }

        # Add additional learner features
        learner_ctx.update(request.features)

        # Add current level for each category
        for category in CATEGORIES:
            if category in request.current_level:
                learner_ctx[f"current_level_{category}"] = (
                    request.current_level[category]
                )

        # ----------------------------------------------------
        # 2. Generate candidate rows
        # ----------------------------------------------------

        rows = [
            build_candidate_row(
                learner_ctx,
                game_id,
                difficulty,
            )
            for game_id, difficulty in ALL_CANDIDATES
        ]

        # ----------------------------------------------------
        # 3. Build feature matrix
        # ----------------------------------------------------

        X = pd.DataFrame(rows)

        # Make sure inference features are exactly
        # the same features used during training
        X = X[bundle["feature_cols"]]

        # ----------------------------------------------------
        # 4. Handle categorical features
        # ----------------------------------------------------

        lib = bundle.get("lib")

        for column in bundle.get("cat_cols", []):

            if lib == "catboost":
                X[column] = X[column].astype(str)

            else:
                X[column] = X[column].astype("category")

        # ----------------------------------------------------
        # 5. Generate predictions
        # ----------------------------------------------------

        if bundle.get("type") == "ucb":

            q10 = bundle["models"][0.1].predict(X)
            q50 = bundle["models"][0.5].predict(X)
            q90 = bundle["models"][0.9].predict(X)

            predictions = (
                q50
                + bundle["k"] * (q90 - q10)
            )

        else:

            predictions = bundle["model"].predict(X)

        # ----------------------------------------------------
        # 6. Apply category calibrators if available
        # ----------------------------------------------------

        calibrators = bundle.get("calibrators")

        if calibrators:

            categories = [
                row["game_category"]
                for row in rows
            ]

            predictions = np.array([
                calibrators[category].predict([prediction])[0]
                if category in calibrators
                else prediction
                for prediction, category
                in zip(predictions, categories)
            ])

        # ----------------------------------------------------
        # 7. Select best candidate
        # ----------------------------------------------------

        best_idx = int(np.argmax(predictions))

        game_id, difficulty = ALL_CANDIDATES[best_idx]

        category = GAMES[game_id]["category"]

        # ----------------------------------------------------
        # 8. Return recommendation
        # ----------------------------------------------------

        return {
            "learner_id": request.learner_id,
            "game_id": game_id,
            "category": category,
            "difficulty": difficulty,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )
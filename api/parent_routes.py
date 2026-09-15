from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

from parent_llm.analyzer import analyze_adventure
from parent_llm.stats import build_adventure_stats


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class GameRecord(BaseModel):
    """
    A single game play record.

    memory / attention -> accuracy_score
    problem_solving    -> efficiency_score
    """

    game_id: str = Field(
        ...,
        example="mem_recognition"
    )

    game_category: str = Field(
        ...,
        example="memory"
    )

    level: int = Field(
        ...,
        ge=1,
        le=3,
        example=1
    )

    accuracy_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        example=0.75
    )

    efficiency_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        example=0.75
    )

    def to_stats_dict(self) -> dict:
        """
        Convert the API request model into the dictionary format
        expected by parent_llm.stats.py.
        """

        data = {
            "game_id": self.game_id,
            "game_category": self.game_category,
            "level": self.level,
        }

        if self.accuracy_score is not None:
            data["accuracy_score"] = self.accuracy_score

        if self.efficiency_score is not None:
            data["efficiency_score"] = self.efficiency_score

        return data


class AnalyzeAdventureRequest(BaseModel):
    """
    Request sent by the backend when an Adventure is completed.
    """

    adventure_id: str = Field(
        ...,
        example="adv_123"
    )

    age: int = Field(
        ...,
        ge=2,
        le=12,
        example=6
    )

    games: List[GameRecord] = Field(
        ...,
        min_length=1
    )


class CategoryAnalysis(BaseModel):
    """
    Analysis for one category.
    """

    current_level: int
    starting_level: int
    level_progression: str
    performance: str
    trend: str


class AnalyzeAdventureResponse(BaseModel):
    """
    Response returned to the backend/frontend after
    the Parent LLM analysis.
    """

    status: str = Field(
        ...,
        example="success",
        description=(
            "success | partial_success | llm_error | "
            "llm_invalid_response | insufficient_data | validation_error"
        )
    )

    summary: Optional[str] = None

    current_levels: Optional[Dict[str, int]] = None

    strengths: Optional[List[str]] = None

    areas_to_practice: Optional[List[str]] = None

    category_analysis: Optional[Dict[str, CategoryAnalysis]] = None

    recommendations: Optional[List[str]] = None

    parent_message: Optional[str] = None

    llm_unavailable: Optional[bool] = False

    excluded_games: Optional[int] = None

    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """
    Check whether the Parent LLM API route is available.
    """

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------------------------
# Analyze Adventure
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeAdventureResponse
)
async def analyze_adventure_endpoint(
    request: AnalyzeAdventureRequest
):
    """
    Analyze a completed Adventure and generate a parent-facing summary.

    Flow:

        Backend
           |
           v
        /parent/analyze
           |
           v
        parent_routes.py
           |
           v
        stats.py
           |
           v
        analyzer.py
           |
           v
        LLM
           |
           v
        Parent-facing response
    """

    try:

        # ---------------------------------------------------------------
        # Convert Pydantic request into the format expected by
        # parent_llm.analyzer.py
        # ---------------------------------------------------------------

        raw_adventure = {
            "adventure_id": request.adventure_id,
            "age": request.age,
            "games": [
                game.to_stats_dict()
                for game in request.games
            ]
        }

        logger.info(
            "Analyzing adventure %s for age %s",
            request.adventure_id,
            request.age
        )

        # ---------------------------------------------------------------
        # Run Parent LLM pipeline
        # ---------------------------------------------------------------

        result = analyze_adventure(raw_adventure)

        logger.info(
            "Adventure %s analysis completed with status=%s",
            request.adventure_id,
            result.get("status")
        )

        return result

    except Exception as e:

        logger.exception(
            "Error analyzing adventure %s",
            request.adventure_id
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Analyze Adventure + Raw Statistics
# ---------------------------------------------------------------------------

@router.post("/analyze-with-stats")
async def analyze_adventure_with_raw_stats(
    request: AnalyzeAdventureRequest
):
    """
    Debug/development endpoint.

    Returns both:

    1. Statistics generated by stats.py
    2. Final Parent LLM analysis

    This endpoint is useful during development/testing.
    """

    try:

        raw_adventure = {
            "adventure_id": request.adventure_id,
            "age": request.age,
            "games": [
                game.to_stats_dict()
                for game in request.games
            ]
        }

        # ---------------------------------------------------------------
        # Build deterministic statistics
        # ---------------------------------------------------------------

        stats_result = build_adventure_stats(
            raw_adventure
        )

        # ---------------------------------------------------------------
        # Run LLM analysis
        # ---------------------------------------------------------------

        analysis_result = analyze_adventure(
            raw_adventure
        )

        return {
            "adventure_id": request.adventure_id,
            "stats": stats_result,
            "analysis": analysis_result,
        }

    except Exception as e:

        logger.exception(
            "Error in analyze-with-stats for adventure %s",
            request.adventure_id
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Demo Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyze/demo")
async def demo_analyze():
    """
    Demo endpoint for testing the Parent LLM without
    needing the real backend.
    """

    demo_request = AnalyzeAdventureRequest(
        adventure_id="demo_adventure_001",
        age=6,
        games=[
            GameRecord(
                game_id="mem_recognition",
                game_category="memory",
                level=1,
                accuracy_score=0.60
            ),
            GameRecord(
                game_id="att_odd_one_out",
                game_category="attention",
                level=1,
                accuracy_score=0.70
            ),
            GameRecord(
                game_id="mem_recognition",
                game_category="memory",
                level=1,
                accuracy_score=0.63
            ),
            GameRecord(
                game_id="att_odd_one_out",
                game_category="attention",
                level=2,
                accuracy_score=0.67
            ),
            GameRecord(
                game_id="mem_recognition",
                game_category="memory",
                level=2,
                accuracy_score=0.67
            ),
        ]
    )

    return await analyze_adventure_endpoint(
        demo_request
    )
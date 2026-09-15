"""
Statistics layer for the Adventure AI analyzer.

Pure, deterministic, no network calls. Responsible for:
  - validating raw per-game records
  - computing per-category / per-game statistics for a completed Adventure
  - applying level-aware "improvement vs decline" logic
  - sanity-checking its own output before it's handed to the LLM

The LLM never sees anything that hasn't passed through here first, and never
produces numbers that could contradict what's computed here.
"""

from __future__ import annotations
from typing import Any, Optional
import math

# ---------------------------------------------------------------------------
# Constants — the single source of truth for game rules
# ---------------------------------------------------------------------------

LEVEL_MIN = 1
LEVEL_MAX = 3

CATEGORIES = ("memory", "attention", "problem_solving")

# Which raw score field each category uses
CATEGORY_SCORE_FIELD = {
    "memory": "accuracy_score",
    "attention": "accuracy_score",
    "problem_solving": "efficiency_score",
}

# "Slight decrease" tolerance when a level-up coincides with a score drop.
# Kept per score-type since accuracy and efficiency are different signals
# (efficiency in particular tends to vary more run to run). Values are in
# score units on the 0-1 scale, i.e. 0.05 == 5 percentage points.
SLIGHT_DECREASE_THRESHOLD = {
    "accuracy_score": 0.05,
    "efficiency_score": 0.08,
}

# Minimum score-change magnitude (at the SAME level) to call something
# "improving"/"declining" rather than "stable". Below this, treat as noise.
STABLE_EPSILON = 0.03

# If more than this fraction of games in an Adventure are invalid, the
# overall result is considered unreliable and no analysis is generated.
MAX_INVALID_RATIO = 0.5

VALID_TRENDS = {
    "improving",
    "declining",
    "stable",
    "strong_improvement",
    "progressing_with_higher_difficulty",
    "challenged_by_difficulty",
    "insufficient_data",
}


# ---------------------------------------------------------------------------
# 1. Raw data validation
# ---------------------------------------------------------------------------

def validate_games(adventure: dict) -> tuple[list[dict], list[dict], list[str]]:
    """
    Validate `adventure["games"]`. Returns (valid_games, invalid_games, warnings).

    Each entry in invalid_games is {"index": int, "game": <original record>,
    "reason": str}. `warnings` holds adventure-level (not per-game) problems,
    e.g. a bad `age` field.
    """
    warnings: list[str] = []
    valid: list[dict] = []
    invalid: list[dict] = []

    age = adventure.get("age")
    if not isinstance(age, (int, float)) or isinstance(age, bool) or age <= 0:
        warnings.append(f"invalid or missing adventure-level 'age': {age!r}")

    games = adventure.get("games")
    if not isinstance(games, list):
        warnings.append("'games' is missing or not a list")
        games = []

    for idx, game in enumerate(games):
        reason = _validate_single_game(game)
        if reason is None:
            valid.append(game)
        else:
            invalid.append({"index": idx, "game": game, "reason": reason})

    return valid, invalid, warnings


def _validate_single_game(game: Any) -> Optional[str]:
    """Return None if valid, else a human-readable rejection reason."""
    if not isinstance(game, dict):
        return "record is not an object"

    game_id = game.get("game_id")
    if not isinstance(game_id, str) or not game_id.strip():
        return "missing or invalid 'game_id'"

    category = game.get("game_category")
    if category not in CATEGORIES:
        return f"invalid 'game_category': {category!r}"

    level = game.get("level")
    if not isinstance(level, int) or isinstance(level, bool) or not (LEVEL_MIN <= level <= LEVEL_MAX):
        return f"'level' must be an integer between {LEVEL_MIN} and {LEVEL_MAX}, got {level!r}"

    expected_field = CATEGORY_SCORE_FIELD[category]
    score = game.get(expected_field)
    if score is None:
        return f"missing required '{expected_field}' for category '{category}'"
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return f"'{expected_field}' must be numeric, got {score!r}"
    if math.isnan(score) or math.isinf(score):
        return f"'{expected_field}' is NaN or infinite"
    if not (0.0 <= score <= 1.0):
        return f"'{expected_field}' out of range [0,1]: {score!r}"

    return None


# ---------------------------------------------------------------------------
# 2. Statistics computation
# ---------------------------------------------------------------------------

def _trend_label(level_change: int, score_change: float, score_field: str) -> str:
    threshold = SLIGHT_DECREASE_THRESHOLD[score_field]

    if level_change > 0:
        if score_change >= 0:
            return "strong_improvement"
        if score_change >= -threshold:
            return "progressing_with_higher_difficulty"
        return "challenged_by_difficulty"

    if level_change < 0:
        # Not expected in normal play, but handled defensively rather than
        # assumed impossible.
        if score_change > STABLE_EPSILON:
            return "improving"
        if score_change < -STABLE_EPSILON:
            return "declining"
        return "stable"

    # level_change == 0
    if score_change > STABLE_EPSILON:
        return "improving"
    if score_change < -STABLE_EPSILON:
        return "declining"
    return "stable"


def _category_stats(category: str, plays: list[dict]) -> dict:
    """
    `plays` is the ordered (chronological, as given in the input list) list
    of valid game records for one category.
    """
    score_field = CATEGORY_SCORE_FIELD[category]
    n = len(plays)

    if n == 0:
        return {"games_played": 0, "status": "no_data"}

    def as_point(g: dict) -> dict:
        return {"game_id": g["game_id"], "level": g["level"], "score": g[score_field]}

    first_play = as_point(plays[0])
    latest_play = as_point(plays[-1])
    average_score = sum(g[score_field] for g in plays) / n

    # per-game breakdown, so a category average never hides which specific
    # game is dragging performance up or down
    per_game: dict[str, dict] = {}
    for g in plays:
        gid = g["game_id"]
        entry = per_game.setdefault(gid, {"plays": 0, "_scores": [], "_levels": []})
        entry["plays"] += 1
        entry["_scores"].append(g[score_field])
        entry["_levels"].append(g["level"])
    for gid, entry in per_game.items():
        scores = entry.pop("_scores")
        levels = entry.pop("_levels")
        entry["avg_score"] = round(sum(scores) / len(scores), 4)
        entry["level_range"] = [min(levels), max(levels)]

    if n == 1:
        return {
            "games_played": 1,
            "first_play": first_play,
            "latest_play": latest_play,
            "average_score": round(average_score, 4),
            "level_change": 0,
            "score_change": 0.0,
            "trend": "insufficient_data",
            "trend_basis": "single_play",
            "per_game": per_game,
        }

    if n == 2:
        level_change = latest_play["level"] - first_play["level"]
        score_change = round(latest_play["score"] - first_play["score"], 4)
        trend_basis = "first_vs_latest"
    else:
        # 3+ plays: compare average of first half vs second half, rather
        # than relying on a single first/last data point that could be a
        # fluke. For an odd count, the middle play is dropped from BOTH
        # halves (not fabricated a fractional weight, not silently kept in
        # one half only) — simplest rule that stays symmetric.
        half = n // 2
        first_half = plays[:half]
        second_half = plays[-half:]
        first_half_avg_score = sum(g[score_field] for g in first_half) / half
        second_half_avg_score = sum(g[score_field] for g in second_half) / half
        first_half_avg_level = sum(g["level"] for g in first_half) / half
        second_half_avg_level = sum(g["level"] for g in second_half) / half

        level_change = round(second_half_avg_level - first_half_avg_level)
        score_change = round(second_half_avg_score - first_half_avg_score, 4)
        trend_basis = "first_half_vs_second_half"

    trend = _trend_label(level_change, score_change, score_field)

    return {
        "games_played": n,
        "first_play": first_play,
        "latest_play": latest_play,
        "average_score": round(average_score, 4),
        "level_change": level_change,
        "score_change": score_change,
        "trend": trend,
        "trend_basis": trend_basis,
        "per_game": per_game,
    }


def compute_statistics(age: float, valid_games: list[dict]) -> dict:
    """Build the full `current_adventure` block from validated game records."""
    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for g in valid_games:
        by_category[g["game_category"]].append(g)

    current_adventure = {
        category: _category_stats(category, plays)
        for category, plays in by_category.items()
    }

    current_levels = {
        category: cstats["latest_play"]["level"]
        for category, cstats in current_adventure.items()
        if cstats.get("games_played", 0) > 0
    }

    return {
        "age": age,
        "max_level": LEVEL_MAX,
        "current_levels": current_levels,
        "current_adventure": current_adventure,
    }


# ---------------------------------------------------------------------------
# 3. Statistics self-validation (sanity check before this goes to the LLM)
# ---------------------------------------------------------------------------

def validate_statistics(stats: dict) -> list[str]:
    """Return a list of problems found; an empty list means the stats are sound."""
    problems: list[str] = []
    current_adventure = stats.get("current_adventure", {})

    for category, cat_stats in current_adventure.items():
        if category not in CATEGORIES:
            problems.append(f"unexpected category key: {category!r}")
            continue

        if cat_stats.get("status") == "no_data":
            if cat_stats.get("games_played", 0) != 0:
                problems.append(f"{category}: marked no_data but games_played != 0")
            continue

        n = cat_stats.get("games_played", 0)
        if not isinstance(n, int) or n <= 0:
            problems.append(f"{category}: invalid games_played {n!r}")
            continue

        for point_key in ("first_play", "latest_play"):
            point = cat_stats.get(point_key)
            if not point:
                problems.append(f"{category}: missing {point_key}")
                continue
            level = point.get("level")
            score = point.get("score")
            if not (isinstance(level, int) and LEVEL_MIN <= level <= LEVEL_MAX):
                problems.append(f"{category}.{point_key}: level {level!r} out of range")
            if not (isinstance(score, (int, float)) and 0.0 <= score <= 1.0
                    and not math.isnan(score) and not math.isinf(score)):
                problems.append(f"{category}.{point_key}: score {score!r} invalid")

        trend = cat_stats.get("trend")
        if trend not in VALID_TRENDS:
            problems.append(f"{category}: invalid trend {trend!r}")

        if n == 1:
            if trend != "insufficient_data":
                problems.append(f"{category}: single play must have trend 'insufficient_data'")
            continue

        score_change = cat_stats.get("score_change")
        level_change = cat_stats.get("level_change")
        if not isinstance(score_change, (int, float)) or math.isnan(score_change) or math.isinf(score_change):
            problems.append(f"{category}: invalid score_change {score_change!r}")
        if not isinstance(level_change, int):
            problems.append(f"{category}: invalid level_change {level_change!r}")

        # Arithmetic consistency check — only meaningful for the direct
        # first-vs-latest case. The half-vs-half case (3+ plays) reports an
        # average-of-averages, not a direct subtraction of first/latest, so
        # it isn't re-derived here; the half-split logic itself is what's
        # under test in compute_statistics, not re-verified downstream.
        if cat_stats.get("trend_basis") == "first_vs_latest":
            first = cat_stats["first_play"]["score"]
            latest = cat_stats["latest_play"]["score"]
            first_level = cat_stats["first_play"]["level"]
            latest_level = cat_stats["latest_play"]["level"]
            expected_score_change = round(latest - first, 4)
            expected_level_change = latest_level - first_level
            if abs(expected_score_change - score_change) > 1e-6:
                problems.append(
                    f"{category}: score_change {score_change} inconsistent with "
                    f"first/latest scores ({first} -> {latest})"
                )
            if expected_level_change != level_change:
                problems.append(
                    f"{category}: level_change {level_change} inconsistent with "
                    f"first/latest levels ({first_level} -> {latest_level})"
                )

    avg_scores = [
        c.get("average_score") for c in current_adventure.values()
        if c.get("games_played", 0) > 0
    ]
    for s in avg_scores:
        if not isinstance(s, (int, float)) or math.isnan(s) or math.isinf(s) or not (0.0 <= s <= 1.0):
            problems.append(f"invalid average_score in output: {s!r}")

    return problems


# ---------------------------------------------------------------------------
# 4. Top-level entry point
# ---------------------------------------------------------------------------

def build_adventure_stats(raw_adventure: dict) -> dict:
    """
    Full stage-1 pipeline: validate raw games -> compute stats -> validate stats.

    Returns one of:
      {"status": "insufficient_data", "reason": str, ...}
      {"status": "validation_error", "reason": str, "problems": [...], ...}
      {"status": "ready", "valid_games": int, "invalid_games": [...],
       "warnings": [...], "stats": {...}}
    """
    valid_games, invalid_games, warnings = validate_games(raw_adventure)

    total = len(valid_games) + len(invalid_games)
    if total == 0 or len(valid_games) == 0:
        return {
            "status": "insufficient_data",
            "reason": "no valid game records in this Adventure",
            "invalid_games": invalid_games,
            "warnings": warnings,
        }

    invalid_ratio = len(invalid_games) / total
    if invalid_ratio > MAX_INVALID_RATIO:
        return {
            "status": "validation_error",
            "reason": (
                f"{len(invalid_games)}/{total} game records were invalid "
                f"({invalid_ratio:.0%}), exceeding the {MAX_INVALID_RATIO:.0%} threshold"
            ),
            "invalid_games": invalid_games,
            "warnings": warnings,
        }

    age = raw_adventure.get("age")
    stats = compute_statistics(age, valid_games)

    problems = validate_statistics(stats)
    if problems:
        return {
            "status": "validation_error",
            "reason": "computed statistics failed internal validation",
            "problems": problems,
            "invalid_games": invalid_games,
            "warnings": warnings,
        }

    return {
        "status": "ready",
        "valid_games": len(valid_games),
        "invalid_games": invalid_games,
        "warnings": warnings,
        "stats": stats,
    }

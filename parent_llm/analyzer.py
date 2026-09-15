"""
LLM interpretation layer for the Adventure AI analyzer.

Calls the LLM exactly once per completed Adventure, using only the
statistics produced by stats.py. Never computes numbers itself, never lets
the LLM's prose override those numbers, and always degrades to a
statistics-only summary if the LLM is unavailable.
"""

from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import groq

from parent_llm.stats import build_adventure_stats, VALID_TRENDS

logger = logging.getLogger("aiko.analyzer")

# Model choice: a small/fast model is enough here — the task is following
# strict output/content constraints over already-computed numbers, not open
# -ended reasoning. openai/gpt-oss-20b on Groq is fast, cheap, and has
# documented JSON-mode support, which is what this call actually leans on.
GROQ_MODEL = "openai/gpt-oss-20b"
load_dotenv(Path(__file__).parent / ".env")
_client: groq.Groq | None = None


def _get_client() -> groq.Groq:
    """Lazy singleton so importing this module doesn't require GROQ_API_KEY
    to be set (e.g. when only running stats.py or the offline fallback)."""
    global _client
    if _client is None:
        _client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an assistant that writes a short, warm, parent-facing summary of a child's
progress in an educational game app, based ONLY on statistics that have already
been calculated for you. You do not calculate anything yourself.

DATA RULES
- Use only the numbers and facts given to you in the user message. Never invent
  games, scores, levels, trends, or prior performance not present in the data.
- Levels range from 1 to 3, with 3 being the highest level currently available.
  Reaching level 3 means the child is at the current ceiling, not "still progressing."
- A category with "status": "no_data" was not played this Adventure. Do not mention
  it in strengths, areas_to_practice, or category_analysis. Do not guess how the
  child might have performed.
- A category with "trend": "insufficient_data" has too few plays for a trend. Say
  so plainly (e.g. "not enough data yet to determine a trend") -- never guess
  improvement or decline.
- Never treat a score decrease that coincides with a level increase as automatic
  decline. A level increase with a small score drop typically reflects the child
  maintaining performance under harder difficulty, not regression. The data will
  tell you the trend label already (e.g. "progressing_with_higher_difficulty") --
  interpret it, don't override it.
- Do not recalculate, adjust, or contradict any number given to you (scores,
  levels, changes, trends). Your job is language, not math.

WHAT YOU MUST NEVER DO
- Never diagnose, suggest, or imply any medical, psychological, or developmental
  condition (e.g. ADHD, learning disorder, memory disorder).
- Never call a score "normal" or "abnormal" -- no reference population has been
  supplied.
- Never compare this child to other children.
- Never say a child "has poor memory/attention/etc." Describe performance in this
  Adventure only, using language like "showed," "performance increased,"
  "may benefit from more practice."
- Never overinterpret a single data point or a category with very few plays.

OUTPUT FORMAT
Return ONLY valid JSON matching this exact schema -- no markdown, no commentary,
no text outside the JSON object:

{
  "status": "success",
  "summary": string,
  "current_levels": { "<category>": integer },
  "strengths": [string],
  "areas_to_practice": [string],
  "category_analysis": {
    "<category>": {
      "current_level": integer,
      "starting_level": integer,
      "level_progression": string,
      "performance": string,
      "trend": "improving" | "stable" | "declining" | "strong_improvement" |
               "progressing_with_higher_difficulty" | "challenged_by_difficulty" |
               "insufficient_data"
    }
  },
  "recommendations": [string],
  "parent_message": string
}

Only include categories that have data (status != "no_data") in current_levels,
strengths, areas_to_practice, and category_analysis. Omit "no_data" categories
entirely -- never invent an entry for them.

EXAMPLE

Input data:
{
  "age": 4,
  "max_level": 3,
  "current_adventure": {
    "memory": {
      "games_played": 3,
      "first_play": { "game_id": "mem_recognition", "level": 1, "score": 0.60 },
      "latest_play": { "game_id": "mem_sequence", "level": 2, "score": 0.67 },
      "average_score": 0.64,
      "level_change": 1,
      "score_change": 0.07,
      "trend": "improving",
      "trend_basis": "first_vs_latest"
    },
    "attention": {
      "games_played": 2,
      "first_play": { "game_id": "att_odd_one_out", "level": 1, "score": 0.70 },
      "latest_play": { "game_id": "att_odd_one_out", "level": 2, "score": 0.67 },
      "average_score": 0.685,
      "level_change": 1,
      "score_change": -0.03,
      "trend": "progressing_with_higher_difficulty",
      "trend_basis": "first_vs_latest"
    },
    "problem_solving": { "games_played": 0, "status": "no_data" }
  }
}

Good response:
{
  "status": "success",
  "summary": "The child made strong progress across both activities played in this
    adventure, taking on harder levels in memory and attention while keeping
    performance steady or improving.",
  "current_levels": { "memory": 2, "attention": 2 },
  "strengths": [
    "Improved accuracy in memory tasks while also moving up a difficulty level",
    "Maintained strong performance in attention tasks despite the difficulty increasing"
  ],
  "areas_to_practice": [
    "Continued practice at the current attention difficulty could help build
      more consistent scores before moving up further"
  ],
  "category_analysis": {
    "memory": {
      "current_level": 2,
      "starting_level": 1,
      "level_progression": "Moved from level 1 to level 2 during this adventure",
      "performance": "Accuracy improved from 60% to 67% alongside the level increase",
      "trend": "improving"
    },
    "attention": {
      "current_level": 2,
      "starting_level": 1,
      "level_progression": "Moved from level 1 to level 2 during this adventure",
      "performance": "Accuracy stayed nearly the same (70% to 67%) even as the
        difficulty increased, suggesting the child adapted well to the harder level",
      "trend": "progressing_with_higher_difficulty"
    }
  },
  "recommendations": [
    "Continue offering attention activities at level 2 to build confidence before
      introducing level 3",
    "Memory activities at level 2 or a gentle step up to level 3 could be a good
      next challenge"
  ],
  "parent_message": "Great adventure today! Your child took on harder challenges in
    memory and attention and handled them well -- a small dip in one score while the
    difficulty went up is a completely normal part of learning, not something to
    worry about."
}

Note: problem_solving is entirely absent from current_levels, strengths,
areas_to_practice, and category_analysis above -- this is correct, since it had
"status": "no_data". Never fill in a placeholder entry for it.
"""


def build_user_prompt(stats: dict) -> str:
    return (
        "Here is the statistics summary for this completed Adventure. Write your "
        "analysis based only on this data.\n\n"
        f"{json.dumps(stats, indent=2)}"
    )


# ---------------------------------------------------------------------------
# LLM call — isolated so the provider can be swapped without touching the
# rest of the pipeline. Provider-specific exceptions should be mapped into
# LLMTransientError (worth retrying) or LLMPermanentError (not worth retrying).
# ---------------------------------------------------------------------------

class LLMTransientError(Exception):
    """Timeout, rate limit, 5xx — worth retrying."""


class LLMPermanentError(Exception):
    """Bad credentials, invalid request schema — retrying won't help."""


def _call_llm_raw(system_prompt: str, user_prompt: str) -> str:
    """
    Groq call. Kept isolated in this one function so switching providers
    later only means rewriting this function — nothing else in the file
    changes. response_format json_object is enough here since we run our
    own strict schema/content validation right after (validate_llm_response)
    rather than trusting provider-side enforcement alone.
    """
    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            timeout=15,
        )
        return response.choices[0].message.content
    except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as exc:
        # covers timeouts (APITimeoutError is a subclass of
        # APIConnectionError), rate limits, and 5xx — all worth retrying
        raise LLMTransientError(str(exc)) from exc
    except (groq.AuthenticationError, groq.PermissionDeniedError,
            groq.NotFoundError, groq.BadRequestError, groq.UnprocessableEntityError) as exc:
        # bad credentials / malformed request — retrying won't help
        raise LLMPermanentError(str(exc)) from exc


def call_llm_with_retry(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Exponential backoff retry for transient failures only."""
    attempt = 0
    while True:
        try:
            return _call_llm_raw(system_prompt, user_prompt)
        except LLMPermanentError:
            raise
        except LLMTransientError as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "LLM transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

REQUIRED_TOP_LEVEL_FIELDS = {
    "status", "summary", "current_levels", "strengths", "areas_to_practice",
    "category_analysis", "recommendations", "parent_message",
}


def validate_llm_response(raw_text: str, stats: dict) -> dict:
    """
    Parse and validate the LLM's JSON against the schema AND against the
    stats it was given (no_data categories must not appear, current_level
    must match, etc). Raises ValueError on any problem; callers treat that
    as llm_invalid_response rather than trying to repair the output.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")

    missing = REQUIRED_TOP_LEVEL_FIELDS - parsed.keys()
    if missing:
        raise ValueError(f"missing required fields: {missing}")

    current_adventure = stats["current_adventure"]
    no_data_categories = {c for c, s in current_adventure.items() if s.get("status") == "no_data"}
    has_data_categories = set(current_adventure) - no_data_categories

    for field in ("current_levels", "category_analysis"):
        if not isinstance(parsed[field], dict):
            raise ValueError(f"'{field}' must be an object")
        keys = set(parsed[field].keys())
        leaked = keys & no_data_categories
        if leaked:
            raise ValueError(f"{field} includes no_data categories: {leaked}")
        unknown = keys - has_data_categories
        if unknown:
            raise ValueError(f"{field} includes unrecognized categories: {unknown}")

    for category, analysis in parsed["category_analysis"].items():
        trend = analysis.get("trend")
        if trend not in VALID_TRENDS:
            raise ValueError(f"category_analysis[{category}].trend invalid: {trend!r}")
        expected_level = current_adventure[category]["latest_play"]["level"]
        if analysis.get("current_level") != expected_level:
            raise ValueError(
                f"category_analysis[{category}].current_level "
                f"({analysis.get('current_level')}) doesn't match stats ({expected_level})"
            )

    for field in ("strengths", "areas_to_practice", "recommendations"):
        if not isinstance(parsed[field], list):
            raise ValueError(f"'{field}' must be a list")

    return parsed


# ---------------------------------------------------------------------------
# Fallback: statistics-only summary, no LLM involved
# ---------------------------------------------------------------------------

_TREND_PHRASES = {
    "improving": "improved",
    "declining": "declined",
    "stable": "stayed about the same",
    "strong_improvement": "improved strongly, including reaching a harder level",
    "progressing_with_higher_difficulty": "held steady while taking on a harder level",
    "challenged_by_difficulty": "was challenged by a harder level",
    "insufficient_data": "doesn't have enough data yet for a trend",
}


def build_fallback_summary(stats: dict) -> dict:
    """
    Pure statistics-only summary used when the LLM is unavailable. No
    network call, no LLM — generated directly from stats.py's output so the
    app keeps working during a provider outage.
    """
    current_adventure = stats["current_adventure"]
    current_levels = {
        c: s["latest_play"]["level"] for c, s in current_adventure.items()
        if s.get("games_played", 0) > 0
    }

    category_analysis = {}
    for category, s in current_adventure.items():
        if s.get("status") == "no_data":
            continue
        phrase = _TREND_PHRASES.get(s.get("trend"), "was recorded")
        category_analysis[category] = {
            "current_level": s["latest_play"]["level"],
            "starting_level": s["first_play"]["level"],
            "level_progression": f"Level {s['first_play']['level']} to {s['latest_play']['level']}",
            "performance": f"Performance {phrase} across {s['games_played']} play(s).",
            "trend": s.get("trend", "insufficient_data"),
        }

    played = [c for c in current_adventure if current_adventure[c].get("games_played", 0) > 0]

    return {
        "status": "success",
        "llm_unavailable": True,
        "summary": (
            "This is a basic summary generated from recorded statistics; a fuller "
            "written analysis wasn't available this time."
        ),
        "current_levels": current_levels,
        "strengths": [],
        "areas_to_practice": [],
        "category_analysis": category_analysis,
        "recommendations": [],
        "parent_message": (
            f"Your child played {', '.join(played) if played else 'no'} activities "
            "in this adventure. A detailed write-up wasn't available right now, but "
            "the numbers have been saved."
        ),
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def analyze_adventure(raw_adventure: dict) -> dict:
    """
    Full pipeline: raw data -> stats -> LLM -> validated analysis.

    Always returns a dict with a "status" field:
      insufficient_data | validation_error | success | partial_success |
      llm_error | llm_invalid_response
    Never raises for expected failure modes; a crash here would be a bug.
    """
    adventure_id = raw_adventure.get("adventure_id")
    stage1 = build_adventure_stats(raw_adventure)

    if stage1["status"] != "ready":
        logger.info("adventure_id=%s stats_status=%s reason=%s",
                     adventure_id, stage1["status"], stage1.get("reason"))
        return {"status": stage1["status"], "reason": stage1.get("reason")}

    stats = stage1["stats"]
    logger.info("adventure_id=%s valid_games=%d invalid_games=%d",
                 adventure_id, stage1["valid_games"], len(stage1["invalid_games"]))

    user_prompt = build_user_prompt(stats)

    try:
        raw_response = call_llm_with_retry(SYSTEM_PROMPT, user_prompt)
    except (LLMTransientError, LLMPermanentError) as exc:
        logger.error("adventure_id=%s llm_call_failed error=%s", adventure_id, exc)
        fallback = build_fallback_summary(stats)
        fallback["status"] = "llm_error"
        fallback["reason"] = "LLM call failed; showing statistics-only summary"
        return fallback

    try:
        analysis = validate_llm_response(raw_response, stats)
    except ValueError as exc:
        logger.error("adventure_id=%s llm_response_invalid error=%s", adventure_id, exc)
        fallback = build_fallback_summary(stats)
        fallback["status"] = "llm_invalid_response"
        fallback["reason"] = f"LLM response failed validation: {exc}"
        return fallback

    if stage1["invalid_games"]:
        analysis["status"] = "partial_success"
        analysis["excluded_games"] = len(stage1["invalid_games"])

    return analysis

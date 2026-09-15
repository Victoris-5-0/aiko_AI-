"""
Rule-based pedagogical baseline. Reconstructed from the description:
  category priority = 0.5 * staleness + 0.5 * weakness, plus a repeat
  penalty against the last-played category; difficulty via a simple
  ratchet: performance >= 0.85 -> increase, <= 0.55 -> decrease, else hold.

This policy is intentionally left untouched by the ML investigation -- it
must remain an honest, unmodified baseline for the comparison to mean
anything.
"""

from __future__ import annotations

import numpy as np

from recommendation.simulator import CATEGORIES, STALENESS_SATURATION_GAMES, Child, GAMES
from recommendation.features import pick_game_in_category, rolling_mean

REPEAT_PENALTY = 0.15
RULE_INCREASE_THRESHOLD = 0.85
RULE_DECREASE_THRESHOLD = 0.55


def _staleness_norm(child: Child, category: str) -> float:
    idx = child.category_last_index.get(category)
    if idx is None:
        return 1.0
    since = child.games_played_count - idx
    return min(1.0, since / STALENESS_SATURATION_GAMES)


def _weakness(child: Child, category: str) -> float:
    perf = rolling_mean(child.history, category, 5)
    if np.isnan(perf):
        return 1.0
    return 1.0 - perf


def rule_update_level(child: Child, category: str) -> None:
    """Rule engine's own difficulty ratchet (distinct from the generic
    behavior-policy ratchet used elsewhere)."""
    perf = rolling_mean(child.history, category, 5)
    if np.isnan(perf):
        return  # not enough data yet, hold current level
    if perf >= RULE_INCREASE_THRESHOLD:
        child.category_level[category] = min(3, child.category_level[category] + 1)
    elif perf <= RULE_DECREASE_THRESHOLD:
        child.category_level[category] = max(1, child.category_level[category] - 1)


def decide_action_rule(child: Child) -> tuple[str, int, str]:
    if child.onboarding_queue:
        game_id = child.onboarding_queue.pop(0)
        return game_id, 1, GAMES[game_id]["category"]

    last_category = child.history[-1]["game_category"] if child.history else None

    priorities = {}
    for cat in CATEGORIES:
        priority = 0.5 * _staleness_norm(child, cat) + 0.5 * _weakness(child, cat)
        if cat == last_category:
            priority -= REPEAT_PENALTY
        priorities[cat] = priority

    category = max(priorities, key=priorities.get)
    game_id = pick_game_in_category(child.history, category)

    # difficulty ratchet must be evaluated BEFORE we mutate anything this turn
    rule_update_level(child, category)
    difficulty = child.category_level[category]

    return game_id, difficulty, category

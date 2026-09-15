"""
Behavior policy: decides which of the 9 actions is actually played at each
decision point, so that logged contexts reflect a realistic operational
distribution. This is intentionally separate from features.py — it is allowed
to use hidden ground-truth ability (a plausible stand-in for "a reasonably
good historical policy"), because it only drives *which* action gets played
and how the simulator state evolves. It is NEVER used as a training feature.
"""

from __future__ import annotations

import numpy as np

from simulator import (
    CATEGORIES,
    DIFFICULTIES,
    DIFFICULTY_B,
    GAME_IDS,
    GAMES,
    Child,
    ability_to_theta,
    natural_level,
)
from features import pick_game_in_category, weakest_category

EXPLORATION_RATE = 0.30


def stratified_exploration_difficulty(child: Child, game_id: str, rng: np.random.Generator) -> int:
    cfg = GAMES[game_id]
    theta = ability_to_theta(child.ability[cfg["category"]])
    level = natural_level(theta, cfg)

    bin_choice = rng.choice(["match", "mild", "extreme"], p=[0.2, 0.3, 0.5])
    if bin_choice == "match":
        return level
    if bin_choice == "mild":
        options = [d for d in DIFFICULTIES if abs(d - level) == 1] or list(DIFFICULTIES)
        return int(rng.choice(options))
    options = sorted(DIFFICULTIES, key=lambda d: abs(d - level), reverse=True)
    return int(options[0])


def choose_action(child: Child, rng: np.random.Generator) -> tuple[str, int, str]:
    if child.onboarding_queue:
        game_id = child.onboarding_queue.pop(0)
        return game_id, 1, "onboarding"

    if rng.random() < EXPLORATION_RATE:
        game_id = rng.choice(GAME_IDS)
        difficulty = stratified_exploration_difficulty(child, game_id, rng)
        cfg = GAMES[game_id]
        theta = ability_to_theta(child.ability[cfg["category"]])
        level = natural_level(theta, cfg)
        bin_tag = "match" if difficulty == level else ("mild" if abs(difficulty - level) == 1 else "extreme")
        return game_id, difficulty, bin_tag

    category = weakest_category(child.history, excluded=set())
    game_id = pick_game_in_category(child.history, category)
    difficulty = child.category_level[category]
    return game_id, difficulty, "exploit"

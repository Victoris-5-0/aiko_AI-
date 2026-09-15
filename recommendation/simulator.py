"""
Aiko environment / simulator core.

This module preserves the original simulator's IRT-based game mechanics
and behavior policy unchanged (per the review's instruction to keep
proven simulator concepts intact). What changes lives in generate_data.py
(counterfactual, all-9-candidate labeling) and features.py (feature
engineering shared between training-time and inference-time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.stats import truncnorm

DIFFICULTIES = (1, 2, 3)

GAMES = {
    "prob_maze": {
        "category": "problem_solving",
        "metric": "efficiency",
        "comfort_target": 0.75,
        "optimal_moves": {1: 5, 2: 9, 3: 14},
        "seconds_per_move_mean": 3.0,
        "difficulty_b": {1: -1.45, 2: -0.55, 3: 0.35},
    },
    "att_odd_one_out": {
        "category": "attention",
        "metric": "accuracy",
        "comfort_target": 0.78,
        "n_trials_range": (10, 16),
        "difficulty_b": {1: -1.45, 2: -0.55, 3: 0.35},
    },
    "mem_recognition": {
        "category": "memory",
        "metric": "accuracy",
        "comfort_target": 0.72,
        "n_trials_range": (10, 18),
        "difficulty_b": {1: -1.45, 2: -0.55, 3: 0.35},
    },
}

DIFFICULTY_B = {1: -1.45, 2: -0.55, 3: 0.35}

CATEGORIES = sorted({g["category"] for g in GAMES.values()})
GAME_IDS = list(GAMES.keys())
GAME_BY_CATEGORY = {c: [g for g, cfg in GAMES.items() if cfg["category"] == c] for c in CATEGORIES}
CATEGORY_METRIC = {c: GAMES[GAME_BY_CATEGORY[c][0]]["metric"] for c in CATEGORIES}
CATEGORY_OF_GAME = {g: cfg["category"] for g, cfg in GAMES.items()}

EXPLORATION_RATE = 0.30
PASS_THRESHOLD = 0.50
HARD_FAIL_THRESHOLD = 0.20

FLOW_WIDTH_LOW = 0.13
FLOW_WIDTH_HIGH = 0.20
REWARD_FLOOR = 0.2

STALENESS_SATURATION_GAMES = 20.0
STALENESS_REWARD_WEIGHT = 0.15  # kept only as a diagnostic/feature weight now, not the training target

THETA_SCALE = 4.0
DISCRIMINATION_HARD = 3.5
DISCRIMINATION_EASY = 2.0

ROLLING_WINDOWS = (3, 5, 10)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def ability_to_theta(ability: float) -> float:
    return (ability - 0.5) * THETA_SCALE


def irt_probability(theta: float, b: float) -> float:
    diff = theta - b
    a = DISCRIMINATION_HARD if diff < 0 else DISCRIMINATION_EASY
    return sigmoid(a * diff)


for _cfg in GAMES.values():
    _cfg["match_offset"] = logit(_cfg["comfort_target"]) / DISCRIMINATION_EASY


def natural_level(theta: float, cfg: dict) -> int:
    adjusted = theta - cfg["match_offset"]
    return min(DIFFICULTIES, key=lambda lvl: abs(adjusted - cfg["difficulty_b"][lvl]))


def flow_reward(score: float, target: float) -> float:
    width = FLOW_WIDTH_LOW if score < target else FLOW_WIDTH_HIGH
    gaussian = np.exp(-((score - target) ** 2) / (2 * width ** 2))
    return REWARD_FLOOR + (1 - REWARD_FLOOR) * gaussian


def shape_reward_with_staleness(flow: float, category_since_played) -> float:
    """Diagnostic-only shaped reward (NOT the training target anymore)."""
    if category_since_played is None or (
        isinstance(category_since_played, float) and np.isnan(category_since_played)
    ):
        staleness = 1.0
    else:
        staleness = min(1.0, float(category_since_played) / STALENESS_SATURATION_GAMES)
    base = (1.0 - STALENESS_REWARD_WEIGHT) + STALENESS_REWARD_WEIGHT * staleness
    return float(np.clip(flow * base, REWARD_FLOOR, 1.0))


def trunc_normal(mean, sd, low, high, size, rng):
    a, b = (low - mean) / sd, (high - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng)


@dataclass
class Child:
    learner_id: str
    age: int
    ability: dict  # hidden ground truth — simulator-only, never exposed to model
    learning_rate: float  # hidden ground truth — simulator-only
    fatigue_factor: float  # hidden ground truth — simulator-only

    category_level: dict = field(default_factory=lambda: {c: 1 for c in CATEGORIES})
    onboarding_queue: list = field(default_factory=list)

    games_played_count: int = 0
    game_last_index: dict = field(default_factory=dict)
    category_last_index: dict = field(default_factory=dict)
    category_last_ts: dict = field(default_factory=dict)
    category_sessions_played: dict = field(default_factory=lambda: {c: set() for c in CATEGORIES})
    history: list = field(default_factory=list)

    session_id: int = -1
    position_in_session: int = 0
    plays_in_session: int = 0


def sample_child(idx: int, rng: np.random.Generator) -> Child:
    age = int(rng.integers(4, 10))
    ability = {c: float(trunc_normal(0.5, 0.16, 0.05, 0.95, None, rng)) for c in CATEGORIES}
    learning_rate = float(trunc_normal(0.02, 0.012, 0.0, 0.08, None, rng))
    fatigue_factor = float(trunc_normal(0.03, 0.015, 0.0, 0.1, None, rng))

    child = Child(
        learner_id=f"child_{idx:04d}",
        age=age,
        ability=ability,
        learning_rate=learning_rate,
        fatigue_factor=fatigue_factor,
    )
    order = list(GAME_IDS)
    rng.shuffle(order)
    child.onboarding_queue = order
    return child


def simulate_accuracy_game(child: Child, game_id: str, level: int, fatigue: float, rng: np.random.Generator) -> dict:
    cfg = GAMES[game_id]
    ability = child.ability[cfg["category"]]
    theta = ability_to_theta(ability) - fatigue * THETA_SCALE
    b = cfg["difficulty_b"][level]
    p_correct = float(np.clip(irt_probability(theta, b), 0.02, 0.98))
    n_trials = int(rng.integers(*cfg["n_trials_range"]))
    correct = int(rng.binomial(n_trials, p_correct))
    accuracy_score = correct / n_trials
    return {
        "accuracy_score": accuracy_score,
        "efficiency_score": np.nan,
        "time_taken_seconds": np.nan,
        "n_trials": n_trials,
    }


def simulate_efficiency_game(child: Child, game_id: str, level: int, fatigue: float, rng: np.random.Generator) -> dict:
    cfg = GAMES[game_id]
    ability = child.ability[cfg["category"]]
    theta = ability_to_theta(ability) - fatigue * THETA_SCALE
    b = cfg["difficulty_b"][level]
    base_efficiency = irt_probability(theta, b)
    noise = float(trunc_normal(0.0, 0.06, -0.2, 0.2, None, rng))
    efficiency_score = float(np.clip(base_efficiency + noise, 0.03, 1.0))
    effective_ability = float(np.clip(ability - fatigue, 0.05, 0.98))
    optimal_moves = cfg["optimal_moves"][level]
    moves_taken = max(optimal_moves, int(round(optimal_moves / efficiency_score)))
    per_move_time = max(
        0.5,
        float(
            trunc_normal(
                cfg["seconds_per_move_mean"] * (1.15 - 0.3 * effective_ability), 0.6, 0.5, 8.0, None, rng
            )
        ),
    )
    time_taken_seconds = round(moves_taken * per_move_time, 1)
    return {
        "accuracy_score": np.nan,
        "efficiency_score": efficiency_score,
        "time_taken_seconds": time_taken_seconds,
        "n_trials": np.nan,
    }


def simulate_game(child: Child, game_id: str, level: int, fatigue: float, rng: np.random.Generator) -> dict:
    cfg = GAMES[game_id]
    if cfg["metric"] == "accuracy":
        return simulate_accuracy_game(child, game_id, level, fatigue, rng)
    return simulate_efficiency_game(child, game_id, level, fatigue, rng)


def score_and_reward(game_id: str, outcome: dict) -> tuple[float, float]:
    """Return (raw_score, raw_flow_reward) for a simulated outcome."""
    cfg = GAMES[game_id]
    score = outcome["accuracy_score"] if cfg["metric"] == "accuracy" else outcome["efficiency_score"]
    reward = flow_reward(score, cfg["comfort_target"])
    return float(score), float(reward)


def update_behavior_policy_state(child: Child, category: str, score: float) -> None:
    if score >= PASS_THRESHOLD:
        child.category_level[category] = min(3, child.category_level[category] + 1)
    elif score < HARD_FAIL_THRESHOLD:
        child.category_level[category] = max(1, child.category_level[category] - 1)


def apply_learning(child: Child, categories_played) -> None:
    for cat in categories_played:
        child.ability[cat] = min(0.98, child.ability[cat] + child.learning_rate * (1 - child.ability[cat]))

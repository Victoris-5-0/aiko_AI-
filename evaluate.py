"""
Head-to-head evaluator: rule baseline vs ML policy, on FRESH unseen
children (a different seed range from training), with:
  - a session schedule (session count, plays-per-session, gaps) generated
    ONCE per child and shared by both arms, so trajectory length/timing
    can't differ between arms;
  - independently-instantiated but identically-seeded outcome RNGs per
    arm, so the two arms' random draws are positionally aligned (common
    random numbers) rather than one arm consuming draws that the other
    then inherits, which was the original evaluator's pairing bug.

Also reports, for every decision, the TRUE oracle reward (via the same
all-9-candidate simulation used in training) so we can report regret for
both policies, not just their raw selected-action reward.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from simulator import (
    GAMES,
    apply_learning,
    sample_child,
    score_and_reward,
    simulate_game,
    HARD_FAIL_THRESHOLD,
)
from features import ALL_CANDIDATES, build_learner_context
from rule_policy import decide_action_rule, rule_update_level
from ml_policy import decide_action_ml, ml_update_level

EVAL_SEED_BASE = 900_000  # far outside the training seed's draw range
N_EVAL_CHILDREN = 120
MIN_SESSIONS = 6
MAX_SESSIONS = 16
MIN_PLAYS = 1
MAX_PLAYS = 6
SESSION_GAP_DAYS = (1, 4)
FLOW_THRESHOLD = 0.75


def make_schedule(idx: int) -> list[int]:
    rng = np.random.default_rng(EVAL_SEED_BASE + 2 * idx)
    n_sessions = int(rng.integers(MIN_SESSIONS, MAX_SESSIONS + 1))
    return [int(rng.integers(MIN_PLAYS, MAX_PLAYS + 1)) for _ in range(n_sessions)]


def oracle_reward(child, fatigue, rng) -> float:
    """True best achievable reward for this exact pre-action state,
    without mutating the child (diagnostic only)."""
    best = -1.0
    for game_id, difficulty in ALL_CANDIDATES:
        snapshot_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        outcome = simulate_game(child, game_id, difficulty, fatigue, snapshot_rng)
        _, reward = score_and_reward(game_id, outcome)
        best = max(best, reward)
    return best


def run_trajectory(child, schedule, outcome_rng, decide_fn, is_ml: bool, bundle=None):
    records = []
    timestamp = datetime(2026, 1, 1)
    child.games_played_count = 0

    for session_id, n_plays in enumerate(schedule):
        child.session_id = session_id
        child.plays_in_session = 0

        for _ in range(n_plays):
            fatigue = child.fatigue_factor * child.plays_in_session

            if is_ml:
                game_id, difficulty, category = decide_fn(child, timestamp, bundle)
            else:
                game_id, difficulty, category = decide_fn(child)

            # oracle regret diagnostic from the identical pre-action state,
            # using a RNG fork so it never perturbs the main outcome stream
            oracle = oracle_reward(child, fatigue, outcome_rng)

            outcome = simulate_game(child, game_id, difficulty, fatigue, outcome_rng)
            score, reward = score_and_reward(game_id, outcome)

            if is_ml:
                ml_update_level(child, category, score)
            # rule's own level ratchet already updated inside decide_action_rule

            apply_learning(child, {category})
            child.history.append(
                {
                    "game_id": game_id,
                    "game_category": category,
                    "difficulty": difficulty,
                    "accuracy_score": outcome["accuracy_score"],
                    "efficiency_score": outcome["efficiency_score"],
                    "reward": reward,
                    "timestamp": timestamp,
                }
            )
            child.game_last_index[game_id] = child.games_played_count
            child.category_last_index[category] = child.games_played_count
            child.category_last_ts[category] = timestamp
            child.category_sessions_played[category].add(session_id)
            child.games_played_count += 1
            child.plays_in_session += 1

            records.append(
                {
                    "learner_id": child.learner_id,
                    "session_id": session_id,
                    "category": category,
                    "difficulty": difficulty,
                    "score": score,
                    "reward": reward,
                    "oracle_reward": oracle,
                    "regret": oracle - reward,
                }
            )
        timestamp = timestamp + timedelta(days=int(outcome_rng.integers(*SESSION_GAP_DAYS)))

    return pd.DataFrame(records)


def paired_stats(rule_child_means, ml_child_means):
    diffs = np.array(ml_child_means) - np.array(rule_child_means)
    t_stat, t_p = stats.ttest_rel(ml_child_means, rule_child_means)
    try:
        w_stat, w_p = stats.wilcoxon(ml_child_means, rule_child_means)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    cohens_d = float(diffs.mean() / diffs.std(ddof=1))

    rng = np.random.default_rng(1234)
    boot_means = [
        rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(5000)
    ]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "mean_diff_ml_minus_rule": float(diffs.mean()),
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "paired_t_test_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "cohens_d": cohens_d,
    }


def run(bundle_path="models/deployed_model.joblib", out_report="reports/head_to_head_report.json", label="ml"):
    bundle = joblib.load(bundle_path)

    rule_frames, ml_frames = [], []

    for idx in range(N_EVAL_CHILDREN):
        creation_rng = np.random.default_rng(EVAL_SEED_BASE + 4 * idx + 1)
        base_child = sample_child(idx, creation_rng)
        schedule = make_schedule(idx)

        outcome_seed = EVAL_SEED_BASE + 4 * idx + 2
        rng_rule = np.random.default_rng(outcome_seed)
        rng_ml = np.random.default_rng(outcome_seed)  # identical seed: CRN

        rule_child = copy.deepcopy(base_child)
        ml_child = copy.deepcopy(base_child)

        rule_df = run_trajectory(rule_child, schedule, rng_rule, decide_action_rule, is_ml=False)
        ml_df = run_trajectory(ml_child, schedule, rng_ml, decide_action_ml, is_ml=True, bundle=bundle)

        rule_frames.append(rule_df)
        ml_frames.append(ml_df)
        if (idx + 1) % 20 == 0:
            print(f"{idx + 1}/{N_EVAL_CHILDREN} children evaluated")

    rule_all = pd.concat(rule_frames, ignore_index=True)
    ml_all = pd.concat(ml_frames, ignore_index=True)

    report = {}
    for name, df in [("rule", rule_all), (label, ml_all)]:
        report[name] = {
            "overall_reward": float(df["reward"].mean()),
            "flow_rate": float((df["reward"] >= FLOW_THRESHOLD).mean()),
            "frustration_rate": float((df["score"] < HARD_FAIL_THRESHOLD).mean()),
            "mean_regret": float(df["regret"].mean()),
            "category_reward": df.groupby("category")["reward"].mean().round(4).to_dict(),
            "category_coverage": (df["category"].value_counts(normalize=True) * 100).round(2).to_dict(),
            "category_regret": df.groupby("category")["regret"].mean().round(4).to_dict(),
        }

    rule_child_means = rule_all.groupby("learner_id")["reward"].mean()
    ml_child_means = ml_all.groupby("learner_id")["reward"].mean()
    common_ids = sorted(set(rule_child_means.index) & set(ml_child_means.index))
    report["paired_comparison"] = paired_stats(
        rule_child_means.loc[common_ids].values, ml_child_means.loc[common_ids].values
    )

    with open(out_report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== OVERALL REWARD ===")
    print(f"model: {report[label]['overall_reward']:.6f}")
    print(f"rule:  {report['rule']['overall_reward']:.6f}")
    print(f"\nmodel mean regret: {report[label]['mean_regret']:.4f}")
    print(f"rule  mean regret: {report['rule']['mean_regret']:.4f}")
    print("\ncategory coverage (model):", report[label]["category_coverage"])
    print("category coverage (rule): ", report["rule"]["category_coverage"])
    print("\ncategory reward (model):", report[label]["category_reward"])
    print("category reward (rule): ", report["rule"]["category_reward"])
    print("\npaired comparison:", report["paired_comparison"])

    return report


if __name__ == "__main__":
    run()

"""
Generate the Aiko counterfactual training dataset.

For every decision point:
  1. Take a pre-action snapshot of the child's observable context
     (build_learner_context) -- identical for all 9 candidates.
  2. Simulate ALL 9 (game_id, difficulty) candidates from that same
     pre-action child state, without mutating child state between
     candidates.
  3. Log all 9 rows with their true raw flow_reward (the training target)
     plus a diagnostic staleness-shaped reward (NOT used as a target).
  4. Advance the real trajectory using only the outcome of the action the
     behavior policy actually chose (reusing that candidate's already-
     simulated outcome rather than drawing again, so we don't spend a
     second, divergent random draw on the same context+action).

Output:
  data/aiko_counterfactual_training_data.csv  (9 rows per decision)
  data/aiko_observed_only.csv                 (1 row/decision -- the
                                                historically-taken action;
                                                kept only for comparison
                                                diagnostics)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from recommendation.simulator import (
    CATEGORIES,
    GAMES,
    Child,
    apply_learning,
    sample_child,
    score_and_reward,
    shape_reward_with_staleness,
    simulate_game,
    update_behavior_policy_state,
)
from recommendation.features import ALL_CANDIDATES, build_candidate_row, build_learner_context
from recommendation.behavior_policy import choose_action

SEED = 11
N_CHILDREN = 300
MIN_SESSIONS_PER_CHILD = 6
MAX_SESSIONS_PER_CHILD = 30
# Widened vs. the original 1-3: a real child can plausibly play more than 3
# games in one sitting, and capping the *simulation* at 3 was what made the
# position-in-session-style signal look artificially bounded. plays_in_session
# is now free to range higher, which is also why we dropped a hard
# "position_in_session" feature in features.py rather than keep a signal
# that could never be observed above ~2 in training.
MIN_PLAYS_PER_SESSION = 1
MAX_PLAYS_PER_SESSION = 8
SESSION_GAP_DAYS = (1, 4)

OUT_COUNTERFACTUAL = "data/aiko_counterfactual_training_data.csv"
OUT_OBSERVED = "data/aiko_observed_only.csv"


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    decision_id = 0

    for child_idx in range(N_CHILDREN):
        child = sample_child(child_idx, rng)
        timestamp = datetime(2025, 1, 1) + timedelta(days=int(rng.integers(0, 30)))
        n_sessions = int(rng.integers(MIN_SESSIONS_PER_CHILD, MAX_SESSIONS_PER_CHILD + 1))

        for session_id in range(n_sessions):
            child.session_id = session_id
            child.plays_in_session = 0
            n_plays = int(rng.integers(MIN_PLAYS_PER_SESSION, MAX_PLAYS_PER_SESSION + 1))

            for _play in range(n_plays):
                # ---- 1. pre-action snapshot (shared by all 9 candidates) ----
                learner_ctx = build_learner_context(child, timestamp)

                # ---- 2. decide the action that will actually be played ----
                chosen_game_id, chosen_difficulty, decision_tag = choose_action(child, rng)

                # ---- 3. simulate ALL 9 candidates from the identical snapshot ----
                candidate_outcomes = {}
                for game_id, difficulty in ALL_CANDIDATES:
                    outcome = simulate_game(
                        child, game_id, difficulty, fatigue=child.fatigue_factor * child.plays_in_session, rng=rng
                    )
                    score, raw_reward = score_and_reward(game_id, outcome)
                    category = GAMES[game_id]["category"]
                    shaped_reward = shape_reward_with_staleness(
                        raw_reward, learner_ctx[f"category_since_played_{category}"]
                    )
                    candidate_outcomes[(game_id, difficulty)] = {
                        "outcome": outcome,
                        "score": score,
                        "reward": raw_reward,
                        "shaped_reward_diagnostic": shaped_reward,
                    }

                true_best_reward = max(c["reward"] for c in candidate_outcomes.values())

                for game_id, difficulty in ALL_CANDIDATES:
                    c = candidate_outcomes[(game_id, difficulty)]
                    row = build_candidate_row(learner_ctx, game_id, difficulty)
                    row.update(
                        {
                            "learner_id": child.learner_id,
                            "decision_id": decision_id,
                            "session_id": session_id,
                            "timestamp": timestamp,
                            "decision_tag": decision_tag,
                            "reward": c["reward"],
                            "shaped_reward_diagnostic": c["shaped_reward_diagnostic"],
                            "is_taken_action": int(
                                (game_id == chosen_game_id) and (difficulty == chosen_difficulty)
                            ),
                            "is_true_best_action": int(c["reward"] == true_best_reward),
                        }
                    )
                    rows.append(row)

                # ---- 4. advance the REAL trajectory using only the chosen action ----
                chosen = candidate_outcomes[(chosen_game_id, chosen_difficulty)]
                chosen_category = GAMES[chosen_game_id]["category"]
                chosen_outcome = chosen["outcome"]

                update_behavior_policy_state(child, chosen_category, chosen["score"])
                apply_learning(child, {chosen_category})

                child.history.append(
                    {
                        "game_id": chosen_game_id,
                        "game_category": chosen_category,
                        "difficulty": chosen_difficulty,
                        "accuracy_score": chosen_outcome["accuracy_score"],
                        "efficiency_score": chosen_outcome["efficiency_score"],
                        "reward": chosen["reward"],
                        "timestamp": timestamp,
                    }
                )
                child.game_last_index[chosen_game_id] = child.games_played_count
                child.category_last_index[chosen_category] = child.games_played_count
                child.category_last_ts[chosen_category] = timestamp
                child.category_sessions_played[chosen_category].add(session_id)
                child.games_played_count += 1
                child.plays_in_session += 1

                decision_id += 1

            timestamp = timestamp + timedelta(days=int(rng.integers(*SESSION_GAP_DAYS)))

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os

    os.makedirs("data", exist_ok=True)
    df = run()
    df.to_csv(OUT_COUNTERFACTUAL, index=False)

    observed = df[df["is_taken_action"] == 1].copy()
    observed.to_csv(OUT_OBSERVED, index=False)

    print(f"decisions: {df['decision_id'].nunique()}")
    print(f"counterfactual rows: {len(df)} (9 per decision)")
    print(f"observed-only rows: {len(observed)}")
    print()
    print("category coverage of the OBSERVED (behavior-policy) actions:")
    print((observed["game_category"].value_counts(normalize=True) * 100).round(2))
    print()
    print("mean TRUE reward by category (across ALL 9 candidates, i.e. environment structure):")
    print(df.groupby("game_category")["reward"].mean().round(4))
    print()
    print("mean TRUE reward by category x difficulty:")
    print(df.groupby(["game_category", "difficulty"])["reward"].mean().round(4))

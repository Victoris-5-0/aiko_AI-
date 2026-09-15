# Aiko ML Policy — Rebuilt Pipeline

**Deployed model: Quantile-75** (`models/deployed_model.joblib`, a copy of
`models/quantile75_bundle.joblib`). This was chosen over the raw
mean-regression baseline specifically to give the memory category
meaningful, non-token exposure (14.4% vs. the baseline's 3.2%), at a real
but acceptable cost in aggregate reward — see the comparison table below for
the full trade-off this decision made. `evaluate.py` now defaults to this
bundle; the other three bundles (`reward_model.joblib` / mean baseline,
`calibrated_bundle.joblib`, `ucb_k05_bundle.joblib`) are kept for reference
and can still be passed explicitly to `evaluate.run(bundle_path=...)`.

Implements the approved plan: counterfactual (all-9-candidate) training data,
raw-flow-reward objective (no staleness shaping in the label), decision-quality
model selection (policy value / regret / top-1 accuracy instead of grouped
rank concordance), train/serve feature parity, and a CRN-paired evaluator.

## Result (deployed: Quantile-75)

```
OVERALL REWARD
model (quantile-75): 0.752779
rule:                 0.694685
margin: +0.058106   (still a clear, statistically solid win -- see
                      reports/head_to_head_quantile75.json for the full
                      paired-comparison stats)

mean regret
model: 0.2066
rule:  0.2656

category reward
                attention   memory   problem_solving
model           0.7036      0.6733   0.8055
rule            0.6753      0.6699   0.7379

category coverage (%)
                attention   memory   problem_solving
model           33.08       14.38    52.54
rule            32.26       33.87    33.87
```

For reference, the raw mean-regression baseline scored 0.7986 overall with
only 3.18% memory coverage — see the full four-way comparison further down.
Quantile-75 was chosen to trade some of that margin for materially better
category balance.

## What changed vs. the original system, and why

1. **Counterfactual, all-9-candidate training data** (`generate_data.py`).
   Every decision point snapshots the child's pre-action state and simulates
   all 9 `(game_id, difficulty)` candidates from that identical snapshot
   (no sequential mutation between candidates), then advances the real
   trajectory using only the actually-chosen action's outcome. This removes
   the selection bias inherent in regressing single logged actions and then
   arg-maxing over unseen counterfactuals.
2. **Training target is raw flow reward**, not the staleness-shaped value.
   Staleness is now a *feature* (`candidate_category_staleness`,
   `trend_<category>`), not folded into the label — the shaped reward is
   still computed and logged as `shaped_reward_diagnostic` for audit, but is
   never fit against.
3. **New features**: a statistical (non-leaky) `estimated_theta_<category>`
   derived only from the child's own observed (difficulty, score) history,
   `theta_minus_difficulty_b` / `abs_theta_minus_difficulty_b` (ability-
   difficulty gap for the specific candidate), per-category trend
   (`last3 - last10`), rolling std, and `is_first_play_<category>` flags so
   "never played" is explicit rather than relying on NaN routing.
   `features.py` is imported by *both* the data generator and the live
   inference policy, so there is no train/serve feature skew.
4. **Model selection by decision quality**, not grouped rank concordance:
   mean policy value, mean/median/p90 regret, and top-1 action accuracy,
   computed over true 9-candidate groups, overall and per true-best-category
   (`train.py: decision_level_metrics`). RMSE/MAE/Spearman are kept as
   secondary diagnostics only.
5. **A learning-to-rank alternative was tried and rejected on evidence**
   (`train_ranker.py`). Diagnostics on the pointwise regressors showed a
   differential-shrinkage bias: predictions under-shoot the true reward most
   at the high-value tail, and the under-shoot is worst for the noisier
   accuracy-based categories (memory, attention) and smallest for the
   less-noisy efficiency-based category (problem_solving) — which pulls
   arg-max toward problem_solving regardless of which category is truly
   best. A LambdaRank objective over each 9-candidate group was tested as a
   principled fix (rank loss cares only about relative order, not the exact
   noisy peak value). **It made collapse worse, not better** (problem_solving
   picked 85% of the time vs. the regressor's 73%; see
   `reports/ranker_report.json`), so the pointwise CatBoost regressor
   remains the production model. This is flagged as an open item below,
   not silently dropped.
6. **CRN-paired evaluator** (`evaluate.py`): each eval child gets an
   identical session schedule (session count, plays/session, gaps) generated
   once and shared by both arms, and both arms' outcome RNGs are seeded
   identically (independent `Generator` objects with the same seed), so
   random draws are positionally aligned until the two policies' actions
   diverge — instead of the original evaluator's single shared RNG consumed
   sequentially by one arm and then the other.
7. **`current_level_<category>` train/serve parity**: the ML arm updates its
   own level tracker with the *same* rule the training generator used
   (`update_behavior_policy_state`, 0.50/0.20 thresholds) — not the rule
   baseline's own 0.85/0.55 ratchet, which the rule baseline keeps
   independently (`rule_policy.rule_update_level`). Mixing these up would
   have silently broken the meaning of `current_level_*` at serving time.
8. Dropped `position_in_session` as a feature per your note — it was
   structurally capped by the original 1–3 session-length assumption and
   wouldn't generalize to longer real sessions. Session length was also
   widened (1–8 plays/session in the generator) so `plays_in_session`
   isn't artificially bounded either.

## Three balance-fix approaches tried (`calibrate.py`, `train_quantile.py`)

Requested fixes for near-zero memory coverage, all measured via full CRN
head-to-head runs against the rule baseline (`reports/head_to_head_*.json`):

| Strategy | Overall reward | Margin over rule (0.6947) | Memory coverage | Mean regret |
|---|---|---|---|---|
| Baseline (mean regression) | 0.7986 | +0.104 | 3.18% | 0.161 |
| Calibrated (isotonic, per-category) | 0.7982 | +0.104 | 2.96% | 0.162 |
| Quantile-75 | 0.7528 | +0.058 | 14.38% | 0.207 |
| UCB (q50 + 0.5·(q90-q10)) | 0.7403 | +0.045 | 11.36% | 0.220 |

**Isotonic recalibration (approach 2) does not fix collapse** — it corrects
within-category calibration but not between-category comparability, and
slightly worsened memory's share on the held-out eval half. **Quantile
regression and UCB (approaches 1 and 3) both do increase memory's share**,
substantially, but at a real reward cost — some of the added memory picks
are genuinely not the best action for that context, so this is an actual
reward-vs-balance trade-off, not a free improvement. All four options still
beat the rule baseline by a wide, statistically solid margin. Which one to
ship depends on whether "no category near-zero" is a hard product
requirement (favor quantile-75) or reward is what's actually scored (favor
the baseline). See `models/quantile75_bundle.joblib`,
`models/calibrated_bundle.joblib`, and `models/ucb_k05_bundle.joblib` for
the corresponding deployable bundles, and `ml_policy.py`'s `decide_action_ml`
for how each bundle type is scored.

## Honest open items (not fixed, flagged rather than hidden)

- **Category collapse is reduced in severity but not eliminated**, and it is
  now better understood: memory's true mean reward really is lower across
  all 9 candidates in this environment (0.58 vs. ~0.62), so *some*
  memory-avoidance is a rational, regret-minimizing response, not a bug.
  But the model still picks memory (3–5%) far less than memory is actually
  the best action (~28% of decisions in the training diagnostics), and when
  memory truly is best, the model catches it only ~3.5% of the time,
  vs. ~55% for problem_solving. That gap is a real, unresolved limitation —
  the ranking-objective fix didn't work; a next iteration should probably
  try a category-conditional recalibration step or a quantile/asymmetric
  loss that stops penalizing high-value, high-noise predictions as harshly
  as the current squared-error objective does. This is exactly the kind of
  change that should go through the same "hypothesize, test, keep only if
  the metrics improve" loop used above.
- The current model selection / final refit uses `N_ESTIMATORS=300` and
  `N_FOLDS=3` to fit this environment's compute budget; a production run
  should widen both (and re-validate the winning library) since more trees
  and folds were cheap wins left on the table here purely for turnaround
  time, not for a principled reason.
- `estimate_theta` is a simple inverse-IRT point estimate (assumes the
  "easy" discrimination constant) — adequate as a feature, but not a
  calibrated ability estimator; worth revisiting if regret plateaus.

## How to reproduce

```bash
pip install lightgbm xgboost catboost scikit-learn scipy pandas numpy joblib

python3 generate_data.py     # writes data/aiko_counterfactual_training_data.csv
python3 train.py             # trains + selects the reward regressor -> models/reward_model.joblib
python3 train_ranker.py      # (optional) the rejected ranking alternative, for reference
python3 evaluate.py          # CRN-paired head-to-head -> reports/head_to_head_report.json
```

## File guide

| file | role |
|---|---|
| `simulator.py` | Environment/IRT mechanics, `Child` state, game simulation — preserved from the original generator |
| `features.py` | Shared, leakage-safe feature engineering (used by both training and inference) |
| `behavior_policy.py` | Logging/behavior policy used only to pick which action gets *actually played* while generating training data |
| `generate_data.py` | Builds the all-9-candidate counterfactual training set |
| `train.py` | Trains LightGBM/XGBoost/CatBoost regressors, selects winner by policy value/regret |
| `train_ranker.py` | LambdaRank alternative (tested, rejected — kept for the record) |
| `rule_policy.py` | Unmodified-in-spirit rule baseline (staleness+weakness priority, its own difficulty ratchet) |
| `ml_policy.py` | Live scoring of all 9 candidates with the trained model, train/serve-consistent state updates |
| `evaluate.py` | CRN-paired head-to-head comparison + regret diagnostics |
| `reports/*.json` | Model-selection, ranker, and head-to-head numeric results |
| `models/reward_model.joblib` | The winning (CatBoost) reward regressor |

Note: the generated training CSVs (~120MB) are not included in the delivered
package — run `generate_data.py` to regenerate them (deterministic, seed=11).

import json
import analyzer
import stats as stats_mod

# ---------------------------------------------------------------------------
# 1. The exact example from the spec: memory has 3 plays (tests half/half
#    logic), attention has 2 (tests first/latest), problem_solving has 0.
# ---------------------------------------------------------------------------

raw_adventure = {
    "adventure_id": "adv_123",
    "age": 4,
    "games": [
        {"game_id": "mem_recognition", "game_category": "memory", "level": 1, "accuracy_score": 0.60},
        {"game_id": "att_odd_one_out", "game_category": "attention", "level": 1, "accuracy_score": 0.70},
        {"game_id": "mem_sequence", "game_category": "memory", "level": 1, "accuracy_score": 0.63},
        {"game_id": "att_odd_one_out", "game_category": "attention", "level": 2, "accuracy_score": 0.67},
        {"game_id": "mem_sequence", "game_category": "memory", "level": 2, "accuracy_score": 0.67},
        # one malformed record: level out of range -- should be excluded, not crash
        {"game_id": "bad_game", "game_category": "memory", "level": 5, "accuracy_score": 0.5},
    ],
}

stage1 = stats_mod.build_adventure_stats(raw_adventure)
assert stage1["status"] == "ready", stage1
assert stage1["valid_games"] == 5
assert len(stage1["invalid_games"]) == 1
assert stage1["invalid_games"][0]["reason"].startswith("'level' must be")
print("stage1 OK:", json.dumps(stage1["stats"], indent=2))

stats = stage1["stats"]
mem = stats["current_adventure"]["memory"]
assert mem["games_played"] == 3
assert mem["trend_basis"] == "first_half_vs_second_half"
assert mem["trend"] in stats_mod.VALID_TRENDS

att = stats["current_adventure"]["attention"]
assert att["games_played"] == 2
assert att["trend_basis"] == "first_vs_latest"
assert att["trend"] == "progressing_with_higher_difficulty", att["trend"]

ps = stats["current_adventure"]["problem_solving"]
assert ps == {"games_played": 0, "status": "no_data"}

problems = stats_mod.validate_statistics(stats)
assert problems == [], problems
print("stage1 + self-validation passed\n")

# ---------------------------------------------------------------------------
# 2. Insufficient data: no games at all
# ---------------------------------------------------------------------------

empty_result = analyzer.analyze_adventure({"adventure_id": "adv_empty", "age": 5, "games": []})
assert empty_result["status"] == "insufficient_data", empty_result
print("insufficient_data case OK:", empty_result)

# ---------------------------------------------------------------------------
# 3. validation_error: too many invalid games
# ---------------------------------------------------------------------------

mostly_bad = {
    "adventure_id": "adv_bad",
    "age": 5,
    "games": [
        {"game_id": "x", "game_category": "memory", "level": 9, "accuracy_score": 0.5},
        {"game_id": "y", "game_category": "memory", "level": 9, "accuracy_score": 0.5},
        {"game_id": "z", "game_category": "memory", "level": 1, "accuracy_score": 0.5},
    ],
}
bad_result = analyzer.analyze_adventure(mostly_bad)
assert bad_result["status"] == "validation_error", bad_result
print("validation_error case OK:", bad_result["reason"], "\n")

# ---------------------------------------------------------------------------
# 4. LLM transient failure -> retries then falls back to stats-only summary
# ---------------------------------------------------------------------------

call_count = {"n": 0}

def flaky_then_fail(system_prompt, user_prompt):
    call_count["n"] += 1
    raise analyzer.LLMTransientError("simulated timeout")

analyzer._call_llm_raw = flaky_then_fail
result = analyzer.analyze_adventure(raw_adventure)
assert result["status"] == "llm_error", result
assert result["llm_unavailable"] is True
assert call_count["n"] == 4  # 1 initial + 3 retries
print(f"llm_error fallback OK after {call_count['n']} attempts:", result["parent_message"], "\n")

# ---------------------------------------------------------------------------
# 5. LLM returns malformed JSON -> llm_invalid_response, falls back
# ---------------------------------------------------------------------------

analyzer._call_llm_raw = lambda s, u: "not json at all"
result = analyzer.analyze_adventure(raw_adventure)
assert result["status"] == "llm_invalid_response", result
print("llm_invalid_response case OK:", result["reason"], "\n")

# ---------------------------------------------------------------------------
# 6. LLM tries to invent a no_data category -> rejected
# ---------------------------------------------------------------------------

def good_response_with_leak(system_prompt, user_prompt):
    return json.dumps({
        "status": "success",
        "summary": "test",
        "current_levels": {"memory": 2, "attention": 2, "problem_solving": 1},  # leak!
        "strengths": [],
        "areas_to_practice": [],
        "category_analysis": {},
        "recommendations": [],
        "parent_message": "test",
    })

analyzer._call_llm_raw = good_response_with_leak
result = analyzer.analyze_adventure(raw_adventure)
assert result["status"] == "llm_invalid_response", result
assert "no_data" in result["reason"]
print("no_data leak correctly rejected:", result["reason"], "\n")

# ---------------------------------------------------------------------------
# 7. A clean, valid LLM response -> success (partial_success, since one
#    game in raw_adventure was invalid)
# ---------------------------------------------------------------------------

def good_response(system_prompt, user_prompt):
    return json.dumps({
        "status": "success",
        "summary": "Great progress in memory and attention this adventure.",
        "current_levels": {"memory": 2, "attention": 2},
        "strengths": ["Memory accuracy improved across the adventure"],
        "areas_to_practice": ["Attention could use more practice at level 2"],
        "category_analysis": {
            "memory": {
                "current_level": 2, "starting_level": 1,
                "level_progression": "1 to 2",
                "performance": "improved",
                "trend": mem["trend"],
            },
            "attention": {
                "current_level": 2, "starting_level": 1,
                "level_progression": "1 to 2",
                "performance": "held steady",
                "trend": "progressing_with_higher_difficulty",
            },
        },
        "recommendations": ["Keep practicing at current levels"],
        "parent_message": "Nice work today!",
    })

analyzer._call_llm_raw = good_response
result = analyzer.analyze_adventure(raw_adventure)
assert result["status"] == "partial_success", result
assert result["excluded_games"] == 1
print("success (partial_success) case OK:", json.dumps(result, indent=2))

print("\nALL TESTS PASSED")

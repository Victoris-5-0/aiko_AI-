"""
Simple test to verify the Groq API call works with real data.
Run this to test the API: python test_api_call.py
"""

import sys
import json
from analyzer import analyze_adventure

# Test data with a child's game performance
test_adventure = {
    "adventure_id": "adv_real_test_001",
    "age": 4,
    "games": [
        {"game_id": "mem_recognition", "game_category": "memory", "level": 1, "accuracy_score": 0.60},
        {"game_id": "att_odd_one_out", "game_category": "attention", "level": 1, "accuracy_score": 0.70},
        {"game_id": "mem_sequence", "game_category": "memory", "level": 1, "accuracy_score": 0.63},
        {"game_id": "att_odd_one_out", "game_category": "attention", "level": 2, "accuracy_score": 0.67},
        {"game_id": "mem_sequence", "game_category": "memory", "level": 2, "accuracy_score": 0.67},
    ],
}

print("=" * 80)
print("Testing Groq API with real adventure data")
print("=" * 80)
print(f"\nAdventure ID: {test_adventure['adventure_id']}")
print(f"Age: {test_adventure['age']}")
print(f"Games: {len(test_adventure['games'])} games\n")

try:
    result = analyze_adventure(test_adventure)
    
    print(f"Status: {result.get('status')}")
    print(f"\nSummary:\n{result.get('summary')}\n")
    print(f"Parent Message:\n{result.get('parent_message')}\n")
    print(f"Strengths: {result.get('strengths')}")
    print(f"Areas to Practice: {result.get('areas_to_practice')}")
    print(f"\n✓ API CALL SUCCESSFUL!")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ API CALL FAILED: {e}")
    sys.exit(1)

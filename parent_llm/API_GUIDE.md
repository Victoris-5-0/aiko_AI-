# Testing the API & Running the Backend

## Option 1: Simple API Test (Python)

**Test if the Groq API is working:**

```bash
cd /home/dallash/code/home/aiko/parent_llm
/home/dallash/code/myenv/bin/python test_api_call.py
```

This will:
- Send real game data to the Groq API
- Display the full analysis with parent message
- Confirm the API is working

---

## Option 2: FastAPI Backend Server

**Start the backend server:**

```bash
cd /home/dallash/code/home/aiko/parent_llm
/home/dallash/code/myenv/bin/pip install fastapi uvicorn  # First time only
/home/dallash/code/myenv/bin/python -m uvicorn app:app --reload --port 8000
```

Then:
- **API Docs**: http://localhost:8000/docs (interactive Swagger UI)
- **API**: http://localhost:8000/

### Available Endpoints:

#### 1. Health Check
```
GET /health
```
Returns: `{"status": "healthy"}`

#### 2. Analyze Adventure (Main Endpoint)
```
POST /analyze
Content-Type: application/json

{
  "adventure_id": "adv_123",
  "age": 4,
  "games": [
    {
      "game_id": "mem_recognition",
      "game_category": "memory",
      "level": 1,
      "accuracy_score": 0.60
    },
    {
      "game_id": "att_odd_one_out",
      "game_category": "attention",
      "level": 2,
      "accuracy_score": 0.67
    }
  ]
}
```

Returns: Full analysis with parent message and recommendations

#### 3. Demo Endpoint (for testing)
```
POST /analyze/demo
```
No input needed — uses pre-configured test data

#### 4. Analyze with Raw Stats (Debug)
```
POST /analyze-with-stats
```
Same input as `/analyze`, but returns both stats AND analysis

---

## Testing with cURL

```bash
# Test the demo endpoint
curl -X POST http://localhost:8000/analyze/demo

# Test with custom data
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "adventure_id": "adv_test_001",
    "age": 4,
    "games": [
      {"game_id": "mem_recognition", "game_category": "memory", "level": 1, "accuracy_score": 0.60},
      {"game_id": "att_odd_one_out", "game_category": "attention", "level": 2, "accuracy_score": 0.67},
      {"game_id": "mem_sequence", "game_category": "memory", "level": 2, "accuracy_score": 0.67}
    ]
  }'
```

---

## Expected Response Format

```json
{
  "status": "success",
  "summary": "The child made strong progress in memory and attention...",
  "current_levels": {
    "memory": 2,
    "attention": 2
  },
  "strengths": [
    "Improved accuracy in memory tasks while also moving up a difficulty level",
    "Maintained strong performance in attention tasks despite the difficulty increasing"
  ],
  "areas_to_practice": [
    "Continued practice at the current attention difficulty could help build more consistent scores"
  ],
  "category_analysis": {
    "memory": {
      "current_level": 2,
      "starting_level": 1,
      "level_progression": "1 to 2",
      "performance": "improved",
      "trend": "strong_improvement"
    }
  },
  "recommendations": ["Keep practicing at current levels"],
  "parent_message": "Great adventure today! Your child took on harder challenges...",
  "llm_unavailable": false,
  "excluded_games": 0
}
```

---

## Response Status Codes

| Status | Meaning |
|--------|---------|
| `success` | Full LLM analysis completed ✓ |
| `partial_success` | Analysis succeeded but some games were invalid |
| `llm_error` | LLM failed; showing stats-only fallback |
| `llm_invalid_response` | LLM response was malformed; showing stats-only fallback |
| `insufficient_data` | No valid games in adventure |
| `validation_error` | Too many invalid games (>50%) |

---

## Connecting to Your Frontend

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "adventure_id": "adv_123",
        "age": 4,
        "games": [
            {
                "game_id": "mem_recognition",
                "game_category": "memory",
                "level": 1,
                "accuracy_score": 0.60
            }
        ]
    }
)

analysis = response.json()
print(analysis["parent_message"])
```

### JavaScript/React Example
```javascript
const response = await fetch('http://localhost:8000/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    adventure_id: 'adv_123',
    age: 4,
    games: [
      {
        game_id: 'mem_recognition',
        game_category: 'memory',
        level: 1,
        accuracy_score: 0.60
      }
    ]
  })
});

const analysis = await response.json();
console.log(analysis.parent_message);
```

---

## Troubleshooting

**API Key Not Found:**
```
KeyError: 'GROQ_API_KEY'
```
Solution: Make sure `.env` file exists in `aiko_AI-` with the API key

**Module Not Found: fastapi**
```
pip install fastapi uvicorn
```

**Port Already in Use:**
Change the port in the command:
```bash
/home/dallash/code/myenv/bin/python -m uvicorn app:app --reload --port 8001
```

**Groq API Rate Limited:**
The system automatically retries with exponential backoff (1s, 2s, 4s). If all retries fail, it returns a statistics-only summary.

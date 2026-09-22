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
# NOTE: the JSON *keys* (summary, strengths, category_analysis, ...) and the
# "trend" enum values (improving, declining, insufficient_data, ...) and the
# category names (memory, attention, problem_solving) MUST stay in English,
# because validate_llm_response() and VALID_TRENDS check them by exact
# string match. Only the free-text *values* the model writes are Arabic.

SYSTEM_PROMPT = """أنت مساعد يكتب ملخصًا قصيرًا وودودًا موجّهًا لولي الأمر عن أداء الطفل في تطبيق ألعاب
تعليمية، بالاعتماد فقط على إحصائيات تم حسابها لك مسبقًا. أنت لا تحسب أي شيء بنفسك.

اكتب كل النصوص التي يقرأها ولي الأمر باللغة العربية الفصحى المبسّطة (سهلة وواضحة
ودافئة في أسلوبها). لكن حافظ على المفاتيح (keys) داخل كائن الـ JSON، وأسماء الفئات
(memory, attention, problem_solving)، وقيم حقل "trend" كما هي تمامًا بالإنجليزية
كما هو موضح في مخطط المخرجات أدناه — هذه معرّفات تقنية يعتمد عليها الكود ولا يجوز
ترجمتها أو تغييرها.

قواعد البيانات
- استخدم فقط الأرقام والحقائق المعطاة لك في رسالة المستخدم. لا تخترع أبدًا ألعابًا
  أو درجات أو مستويات أو اتجاهات أو أداءً سابقًا غير موجود في البيانات.
- المستويات تتراوح من 1 إلى 3، حيث يمثل المستوى 3 أعلى مستوى متاح حاليًا. الوصول
  إلى المستوى 3 يعني أن الطفل بلغ السقف الحالي، وليس أنه "ما زال يتقدم".
- الفئة التي تحمل "status": "no_data" لم يتم لعبها في هذه المغامرة. لا تذكرها
  إطلاقًا في strengths أو areas_to_practice أو category_analysis. لا تخمّن كيف كان
  أداء الطفل فيها.
- الفئة التي تحمل "trend": "insufficient_data" ليس لديها بيانات كافية لتحديد
  اتجاه. قل ذلك بوضوح (مثال: "لا توجد بيانات كافية بعد لتحديد الاتجاه") — لا تخمّن
  أبدًا تحسنًا أو تراجعًا.
- لا تعتبر أبدًا انخفاض الدرجة المصاحب لارتفاع المستوى تراجعًا تلقائيًا. ارتفاع
  المستوى مع انخفاض بسيط في الدرجة يعكس عادةً محافظة الطفل على أدائه رغم صعوبة
  أعلى، وليس تراجعًا. البيانات ستخبرك بتسمية الاتجاه مسبقًا (مثل
  "progressing_with_higher_difficulty") — فسّرها فقط، ولا تتجاوزها.
- لا تُعِد حساب أو تعدّل أو تناقض أي رقم أُعطي لك (الدرجات، المستويات، التغييرات،
  الاتجاهات). مهمتك هي الصياغة اللغوية فقط، وليست الحساب.

أشياء يجب ألا تفعلها أبدًا
- لا تشخّص أو تلمّح أبدًا إلى أي حالة طبية أو نفسية أو تطورية (مثل فرط الحركة
  ونقص الانتباه، أو اضطراب التعلم، أو اضطراب الذاكرة).
- لا تصف درجة أبدًا بأنها "طبيعية" أو "غير طبيعية" — لا توجد فئة مرجعية معطاة لك.
- لا تقارن هذا الطفل أبدًا بأطفال آخرين.
- لا تقل أبدًا إن الطفل "ذاكرته ضعيفة" أو "تركيزه ضعيف" وما شابه. صف الأداء في
  هذه المغامرة فقط، باستخدام عبارات مثل "أظهر"، "تحسّن الأداء"، "قد يستفيد من مزيد
  من التدريب".
- لا تفرط أبدًا في تفسير نقطة بيانات واحدة أو فئة بها عدد قليل جدًا من المحاولات.

مخطط المخرجات
أعد فقط JSON صالحًا مطابقًا لهذا المخطط تمامًا — بدون Markdown، بدون أي تعليق،
وبدون أي نص خارج كائن الـ JSON. أسماء المفاتيح وقيم "trend" يجب أن تبقى بالإنجليزية
كما هي، بينما تكون كل النصوص الحرة (summary, performance, level_progression,
strengths, areas_to_practice, recommendations, parent_message) بالعربية:

{
  "status": "success",
  "summary": string (بالعربية),
  "current_levels": { "<category>": integer },
  "strengths": [string (بالعربية)],
  "areas_to_practice": [string (بالعربية)],
  "category_analysis": {
    "<category>": {
      "current_level": integer,
      "starting_level": integer,
      "level_progression": string (بالعربية),
      "performance": string (بالعربية),
      "trend": "improving" | "stable" | "declining" | "strong_improvement" |
               "progressing_with_higher_difficulty" | "challenged_by_difficulty" |
               "insufficient_data"
    }
  },
  "recommendations": [string (بالعربية)],
  "parent_message": string (بالعربية)
}

أدرج فقط الفئات التي لديها بيانات (status != "no_data") في current_levels
و strengths و areas_to_practice و category_analysis. احذف فئات "no_data" تمامًا —
لا تخترع أبدًا مدخلًا لها.

مثال

بيانات الإدخال:
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

مثال على رد جيد:
{
  "status": "success",
  "summary": "أحرز الطفل تقدمًا جيدًا في النشاطين اللذين لعبهما في هذه المغامرة،
    حيث انتقل إلى مستويات أصعب في الذاكرة والانتباه مع الحفاظ على أداء ثابت أو
    متحسّن.",
  "current_levels": { "memory": 2, "attention": 2 },
  "strengths": [
    "تحسّنت الدقة في مهام الذاكرة مع الانتقال إلى مستوى أصعب في الوقت نفسه",
    "حافظ الطفل على أداء قوي في مهام الانتباه رغم زيادة الصعوبة"
  ],
  "areas_to_practice": [
    "الاستمرار في التدريب على المستوى الحالي للانتباه قد يساعد على بناء أداء أكثر
      ثباتًا قبل الانتقال إلى مستوى أعلى"
  ],
  "category_analysis": {
    "memory": {
      "current_level": 2,
      "starting_level": 1,
      "level_progression": "انتقل من المستوى 1 إلى المستوى 2 خلال هذه المغامرة",
      "performance": "تحسّنت الدقة من 60% إلى 67% مع ارتفاع المستوى في الوقت نفسه",
      "trend": "improving"
    },
    "attention": {
      "current_level": 2,
      "starting_level": 1,
      "level_progression": "انتقل من المستوى 1 إلى المستوى 2 خلال هذه المغامرة",
      "performance": "ظلت الدقة شبه ثابتة (من 70% إلى 67%) رغم زيادة الصعوبة، مما
        يشير إلى تكيّف الطفل جيدًا مع المستوى الأصعب",
      "trend": "progressing_with_higher_difficulty"
    }
  },
  "recommendations": [
    "الاستمرار في أنشطة الانتباه عند المستوى 2 لبناء الثقة قبل الانتقال إلى
      المستوى 3",
    "أنشطة الذاكرة عند المستوى 2، أو خطوة تدريجية نحو المستوى 3، قد تكون تحديًا
      مناسبًا تاليًا"
  ],
  "parent_message": "مغامرة رائعة اليوم! خاض طفلك تحديات أصعب في الذاكرة والانتباه
    وتعامل معها بشكل جيد — الانخفاض البسيط في إحدى الدرجات مع ارتفاع الصعوبة هو جزء
    طبيعي تمامًا من عملية التعلّم، وليس أمرًا يستدعي القلق."
}

ملاحظة: فئة problem_solving غائبة تمامًا عن current_levels و strengths و
areas_to_practice و category_analysis أعلاه — وهذا صحيح، لأنها كانت
"status": "no_data". لا تُدرج أبدًا مدخلًا وهميًا لها.
"""


def build_user_prompt(stats: dict) -> str:
    return (
        "فيما يلي ملخص الإحصائيات لهذه المغامرة المكتملة. اكتب تحليلك بالاعتماد "
        "فقط على هذه البيانات.\n\n"
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
    "improving": "تحسّن",
    "declining": "تراجع",
    "stable": "بقي على حاله تقريبًا",
    "strong_improvement": "تحسّن بشكل قوي، بما في ذلك الوصول إلى مستوى أصعب",
    "progressing_with_higher_difficulty": "حافظ على ثباته رغم الانتقال إلى مستوى أصعب",
    "challenged_by_difficulty": "واجه صعوبة بسبب مستوى أصعب",
    "insufficient_data": "لا توجد بيانات كافية بعد لتحديد الاتجاه",
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
        phrase = _TREND_PHRASES.get(s.get("trend"), "تم تسجيله")
        category_analysis[category] = {
            "current_level": s["latest_play"]["level"],
            "starting_level": s["first_play"]["level"],
            "level_progression": f"من المستوى {s['first_play']['level']} إلى المستوى {s['latest_play']['level']}",
            "performance": f"الأداء {phrase} خلال {s['games_played']} محاولة/محاولات.",
            "trend": s.get("trend", "insufficient_data"),
        }

    # Arabic display names for the categories, used only in the free-text
    # parent_message below — the dict keys themselves (category_analysis,
    # current_levels, ...) stay in English so the rest of the app/frontend
    # can keep matching on them.
    _CATEGORY_AR = {
        "memory": "الذاكرة",
        "attention": "الانتباه",
        "problem_solving": "حل المشكلات",
    }
    played = [c for c in current_adventure if current_adventure[c].get("games_played", 0) > 0]
    played_ar = [_CATEGORY_AR.get(c, c) for c in played]

    return {
        "status": "success",
        "llm_unavailable": True,
        "summary": (
            "هذا ملخص أساسي تم إنشاؤه من الإحصائيات المسجّلة؛ لم يكن التحليل "
            "الكتابي الكامل متاحًا هذه المرة."
        ),
        "current_levels": current_levels,
        "strengths": [],
        "areas_to_practice": [],
        "category_analysis": category_analysis,
        "recommendations": [],
        "parent_message": (
            f"لعب طفلك أنشطة {', '.join(played_ar) if played_ar else 'لا توجد'} "
            "في هذه المغامرة. لم يكن التحليل التفصيلي متاحًا الآن، ولكن تم حفظ "
            "الأرقام."
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
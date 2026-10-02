"""Gemini integration (google-genai SDK) returning structured JSON, never HTML to parse."""
import json
import logging

from config.config import settings
from services.nutrition import NUTRIENT_KEYS

logger = logging.getLogger(__name__)

_client = None


def available():
    return bool(settings.GOOGLE_API_KEY)


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _client


_NUTRIENTS_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "number"} for k in NUTRIENT_KEYS},
    "required": NUTRIENT_KEYS,
}

_INSIGHTS_PROPS = {
    "summary": {"type": "string"},
    "health_insight": {"type": "string"},
    "healthier_options": {"type": "array", "items": {"type": "string"}},
    "fun_fact": {"type": "string"},
}

INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": _INSIGHTS_PROPS,
    "required": list(_INSIGHTS_PROPS),
}

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_food": {"type": "boolean"},
        "foods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "emoji": {"type": "string"},
                    "weight_g": {"type": "number"},
                    "confidence": {"type": "number"},
                    "nutrients": _NUTRIENTS_SCHEMA,
                },
                "required": ["name", "emoji", "weight_g", "confidence", "nutrients"],
            },
        },
        **_INSIGHTS_PROPS,
    },
    "required": ["is_food", "foods", *_INSIGHTS_PROPS],
}


def _generate(contents, schema):
    from google.genai import types

    response = _get_client().models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.4,
        ),
    )
    return json.loads(response.text)


def meal_insights(foods, totals):
    """Coaching insights for foods whose nutrition is already known (YOLO path)."""
    if not available():
        return None
    lines = [
        f"- {f['name']}: ~{f['weight_g']} g, {f['nutrients']['calories']} kcal, "
        f"P {f['nutrients']['protein']} g, C {f['nutrients']['carbs']} g, F {f['nutrients']['fat']} g"
        for f in foods
    ]
    prompt = (
        "You are a friendly Indian-cuisine nutrition coach. A user photographed this meal:\n"
        + "\n".join(lines)
        + f"\nMeal totals: {json.dumps(totals)}\n"
        "Return JSON with: summary (one sentence), health_insight (2-3 sentences on benefits and "
        "concerns), healthier_options (3-4 short, practical swaps), fun_fact (one interesting fact "
        "about the dish). Keep numbers consistent with the data given."
    )
    try:
        return _generate(prompt, INSIGHTS_SCHEMA)
    except Exception as exc:
        logger.warning("Gemini insights failed: %s", exc)
        return None


def analyze_image(image_bytes, mime_type):
    """Identify foods (incl. packaged foods / labels) and estimate nutrition from the photo."""
    if not available():
        return None
    from google.genai import types

    prompt = (
        "Identify every distinct food or drink in this photo. If it is packaged food, read the label. "
        "For each item estimate the visible portion weight in grams and its nutrition for that portion "
        "(calories kcal, protein/carbs/fat/fiber/sugar g, sodium/cholesterol mg). Use a single fitting "
        "emoji per item and a 0-1 confidence. Also return summary, health_insight, healthier_options "
        "(3-4 swaps) and fun_fact. If there is no food, set is_food=false and foods=[]."
    )
    try:
        return _generate(
            [types.Part.from_bytes(data=image_bytes, mime_type=mime_type or "image/jpeg"), prompt],
            VISION_SCHEMA,
        )
    except Exception as exc:
        logger.warning("Gemini vision failed: %s", exc)
        return None

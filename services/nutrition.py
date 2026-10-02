"""Nutrition lookup, portion estimation and simple health scoring."""
import re

import pandas as pd

from config.config import settings

NUTRIENT_KEYS = ["calories", "protein", "carbs", "fat", "fiber", "sugar", "sodium", "cholesterol"]

DEFAULT_GOALS = {
    "calories": 2000,
    "protein": 80,
    "carbs": 250,
    "fat": 70,
    "fiber": 25,
    "sugar": 30,
    "sodium": 2000,
    "cholesterol": 300,
}

# YOLO class index -> (display name, emoji)
FOOD_CLASSES = {
    0: ("Biryani", "🍛"),
    1: ("Chole Bhature", "🫓"),
    2: ("Dabeli", "🍔"),
    3: ("Dal", "🥣"),
    4: ("Dhokla", "🧽"),
    5: ("Dosa", "🥞"),
    6: ("Jalebi", "🍥"),
    7: ("Kathi Roll", "🌯"),
    8: ("Kofta", "🧆"),
    9: ("Naan", "🫓"),
    10: ("Pakora", "🍤"),
    11: ("Paneer Tikka", "🍢"),
    12: ("Panipuri", "🥟"),
    13: ("Pav Bhaji", "🍛"),
    14: ("Vada Pav", "🍔"),
}

_CSV_COLUMNS = {
    "Avg. Calories per Unit": "calories",
    "Protein (g)": "protein",
    "Total Fat (g)": "fat",
    "Carbohydrates (g)": "carbs",
    "Fiber (g)": "fiber",
    "Sugar (g)": "sugar",
    "Sodium (mg)": "sodium",
    "Cholesterol (mg)": "cholesterol",
}


def normalize_name(name):
    """'Vada Pav 🍔' / 'vadapav' / 'Vada-Pav' -> 'vadapav'."""
    return re.sub(r"[^a-z]", "", (name or "").lower())


def _load_food_table():
    df = pd.read_csv(settings.FOOD_DATA_FILE, encoding="utf-8-sig")
    table = {}
    for _, row in df.iterrows():
        name = str(row["Food Item"]).strip()
        table[normalize_name(name)] = {
            "name": name,
            "unit_weight": float(row["Avg. Weight per Unit (g)"]),
            "per_unit": {key: float(row[col]) for col, key in _CSV_COLUMNS.items()},
        }
    return table


FOOD_TABLE = _load_food_table()


def lookup_food(name):
    key = normalize_name(name)
    if key in FOOD_TABLE:
        return FOOD_TABLE[key]
    # Loose match, e.g. "paneer" -> "paneertikka"
    for k, v in FOOD_TABLE.items():
        if key and (k.startswith(key) or key.startswith(k)):
            return v
    return None


def estimate_servings(box, img_w, img_h):
    """Estimate how many standard servings a detection represents.

    A food that fills ~35% of the frame is treated as one standard serving;
    the result is clamped so tiny or huge boxes stay plausible.
    """
    x1, y1, x2, y2 = box
    area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, img_w * img_h))
    return round(min(2.5, max(0.4, area_frac / 0.35)), 2)


def nutrients_for(food, servings):
    return {k: round(v * servings, 1) for k, v in food["per_unit"].items()}


def empty_nutrients():
    return {k: 0.0 for k in NUTRIENT_KEYS}


def sum_nutrients(items):
    total = empty_nutrients()
    for n in items:
        for k in NUTRIENT_KEYS:
            try:
                total[k] += float(n.get(k, 0) or 0)
            except (TypeError, ValueError):
                pass
    return {k: round(v, 1) for k, v in total.items()}


def coerce_nutrients(data):
    """Take any dict of nutrient-ish values and return clean floats for all keys."""
    aliases = {"total fat": "fat", "carbohydrates": "carbs", "carbohydrate": "carbs", "kcal": "calories"}
    out = empty_nutrients()
    for key, value in (data or {}).items():
        k = aliases.get(str(key).lower().strip(), str(key).lower().strip())
        if k in out:
            try:
                out[k] = round(float(value), 1)
            except (TypeError, ValueError):
                match = re.search(r"\d+(?:\.\d+)?", str(value))
                out[k] = round(float(match.group()), 1) if match else 0.0
    return out


def health_score(totals):
    """0-100 heuristic: rewards protein & fiber density, penalises sugar, sodium, fat share."""
    cal = totals.get("calories", 0) or 0
    if cal <= 0:
        return None
    per100 = 100.0 / cal
    score = 60.0
    score += min(20, totals.get("protein", 0) * per100 * 4 * 2.5)  # protein kcal share
    score += min(12, totals.get("fiber", 0) * per100 * 10)
    score -= min(20, totals.get("sugar", 0) * per100 * 4 * 1.5)
    score -= min(15, max(0, totals.get("sodium", 0) * per100 - 1.0) * 6)
    fat_share = totals.get("fat", 0) * 9 / cal
    score -= min(15, max(0, fat_share - 0.35) * 60)
    return int(max(5, min(98, round(score))))


def rule_based_insights(foods, totals):
    """Offline fallback used when Gemini is unavailable."""
    names = ", ".join(f["name"] for f in foods) or "this meal"
    cal = totals.get("calories", 0)
    tips = []
    if totals.get("fat", 0) * 9 > cal * 0.35:
        tips.append("Choose baked, grilled or steamed versions to cut down the oil.")
    if totals.get("fiber", 0) < 5:
        tips.append("Add a side of salad, sprouts or cucumber raita for extra fibre.")
    if totals.get("protein", 0) < 15:
        tips.append("Pair it with a protein source like dal, paneer, curd or eggs.")
    if totals.get("sugar", 0) > 20:
        tips.append("Keep sweets to a small portion and skip sugary drinks alongside.")
    if totals.get("sodium", 0) > 700:
        tips.append("Go easy on chutneys, pickles and added salt to lower sodium.")
    if not tips:
        tips.append("Nicely balanced — keep portion sizes consistent and stay hydrated.")
    tone = "light" if cal < 350 else "moderate" if cal < 700 else "hearty"
    return {
        "summary": f"{names} — a {tone} meal of about {round(cal)} kcal.",
        "health_insight": (
            f"This plate provides {round(totals.get('protein', 0))} g protein, "
            f"{round(totals.get('carbs', 0))} g carbs and {round(totals.get('fat', 0))} g fat."
        ),
        "healthier_options": tips[:4],
        "fun_fact": "Indian street food traditions often balance spice, tang and crunch in a single bite.",
    }

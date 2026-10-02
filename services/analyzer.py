"""End-to-end meal analysis: image -> detections -> nutrition -> insights."""
import base64
import html
import io
import logging
import uuid
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
import numpy as np

from services import gemini
from services.detector import detector
from services.nutrition import (
    coerce_nutrients,
    estimate_servings,
    health_score,
    lookup_food,
    nutrients_for,
    rule_based_insights,
    sum_nutrients,
)

logger = logging.getLogger(__name__)

MAX_SIDE = 1280
BOX_COLORS = ["#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#a855f7", "#14b8a6", "#ef4444"]


class InvalidImageError(ValueError):
    pass


def load_image(raw_bytes):
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a readable image.") from exc
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_base64_jpeg(img, max_side=None, quality=85):
    if max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def annotate(img, foods):
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    line_w = max(2, round(min(canvas.size) / 200))
    try:
        font = ImageFont.load_default(size=max(14, round(min(canvas.size) / 30)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, food in enumerate(foods):
        if not food.get("box"):
            continue
        color = BOX_COLORS[i % len(BOX_COLORS)]
        x1, y1, x2, y2 = food["box"]
        draw.rounded_rectangle([x1, y1, x2, y2], radius=line_w * 4, outline=color, width=line_w)
        label = f" {food['name']} {round(food['confidence'] * 100)}% "
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 8 if y1 - th - 8 > 0 else y1 + line_w
        draw.rounded_rectangle([x1, ty, x1 + tw + 4, ty + th + 8], radius=6, fill=color)
        draw.text((x1 + 2, ty + 2), label, fill="white", font=font)
    return canvas


def _foods_from_detections(detections, img_w, img_h):
    """Merge YOLO detections of the same dish and attach nutrition."""
    merged = {}
    for det in detections:
        food = lookup_food(det["name"])
        if not food:
            continue
        servings = estimate_servings(det["box"], img_w, img_h)
        entry = merged.get(det["name"])
        if entry is None:
            merged[det["name"]] = {
                "name": det["name"],
                "emoji": det["emoji"],
                "confidence": det["confidence"],
                "servings": servings,
                "count": 1,
                "boxes": [det["box"]],
                "_food": food,
            }
        else:
            entry["servings"] = round(entry["servings"] + servings, 2)
            entry["count"] += 1
            entry["confidence"] = max(entry["confidence"], det["confidence"])
            entry["boxes"].append(det["box"])

    foods = []
    for e in merged.values():
        food = e.pop("_food")
        e["weight_g"] = round(food["unit_weight"] * e["servings"])
        e["nutrients"] = nutrients_for(food, e["servings"])
        e["box"] = e["boxes"][0]
        foods.append(e)
    return foods


def _foods_from_gemini(result):
    foods = []
    for item in result.get("foods") or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        foods.append(
            {
                "name": name[:80],
                "emoji": (item.get("emoji") or "🍽️")[:4],
                "confidence": round(min(1.0, max(0.0, float(item.get("confidence") or 0.7))), 3),
                "weight_g": round(float(item.get("weight_g") or 0)),
                "servings": 1,
                "count": 1,
                "boxes": [],
                "box": None,
                "nutrients": coerce_nutrients(item.get("nutrients")),
            }
        )
    return foods


def _legacy_html(foods, totals, insights):
    """HTML block kept for backwards compatibility with the v1 frontend."""
    esc = html.escape
    title = " + ".join(f"{f['name']} {f['emoji']}" for f in foods) or "Meal"
    weight = sum(f.get("weight_g", 0) for f in foods)
    rows = [
        ("Calories", totals["calories"], "kcal"), ("Protein", totals["protein"], "g"),
        ("Total Fat", totals["fat"], "g"), ("Carbs", totals["carbs"], "g"),
        ("Fiber", totals["fiber"], "g"), ("Sugar", totals["sugar"], "g"),
        ("Sodium", totals["sodium"], "mg"), ("Cholesterol", totals["cholesterol"], "mg"),
    ]
    items = "".join(f'<li class="nutrition-item"><strong>{k}:</strong> {v} {u}</li>' for k, v, u in rows)
    tips = "".join(f"<li>{esc(t)}</li>" for t in insights.get("healthier_options", []))
    return (
        f'<h2 class="food-title">{esc(title)}</h2><p><strong>Weight:</strong> {weight} g</p>'
        f"<h3>Nutritional Breakdown</h3><ul>{items}</ul>"
        f'<div class="health-insight"><h3>⚡ Health Insight</h3><p>{esc(insights.get("health_insight", ""))}</p></div>'
        f'<div class="healthier-options"><h3>✅ How to Make it Healthier</h3><ul>{tips}</ul></div>'
        f'<div class="fun-fact"><h3>💡 Fun Fact</h3><p>{esc(insights.get("fun_fact", ""))}</p></div>'
    )


def analyze(raw_bytes, mime_type=None):
    img = load_image(raw_bytes)
    w, h = img.size
    rgb = np.asarray(img)

    detections = []
    try:
        detections = detector.detect(rgb)
    except Exception:
        logger.exception("YOLO inference failed; falling back to Gemini")

    foods = _foods_from_detections(detections, w, h)
    source = "yolo" if foods else "none"
    insights = None

    if foods:
        totals = sum_nutrients(f["nutrients"] for f in foods)
        insights = gemini.meal_insights(foods, totals)
    else:
        jpeg = io.BytesIO()
        img.save(jpeg, format="JPEG", quality=90)
        result = gemini.analyze_image(jpeg.getvalue(), "image/jpeg")
        if result and result.get("is_food", True):
            foods = _foods_from_gemini(result)
            if foods:
                source = "gemini"
                insights = {k: result.get(k) for k in ("summary", "health_insight", "healthier_options", "fun_fact")}
        totals = sum_nutrients(f["nutrients"] for f in foods)

    if foods and not (insights and insights.get("health_insight")):
        insights = rule_based_insights(foods, totals)
    insights = insights or {}
    insights = {
        "summary": insights.get("summary") or "",
        "health_insight": insights.get("health_insight") or "",
        "healthier_options": [str(t) for t in (insights.get("healthier_options") or [])][:5],
        "fun_fact": insights.get("fun_fact") or "",
        "health_score": health_score(totals),
    }

    annotated_b64 = to_base64_jpeg(annotate(img, foods)) if source == "yolo" else None
    return {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc),
        "detection_source": source,
        "foods": foods,
        "totals": totals,
        "insights": insights,
        "image_size": [w, h],
        "annotated_image": annotated_b64,
        "thumbnail": to_base64_jpeg(img, max_side=360, quality=80),
        "legacy_html": _legacy_html(foods, totals, insights) if foods else "No food or readable text detected.",
    }

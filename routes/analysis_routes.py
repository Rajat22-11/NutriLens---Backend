import logging
import uuid
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from werkzeug.utils import secure_filename

from config.config import settings, user_data_collection
from controllers.auth_controller import error
from services import analyzer
from services.history import build_entry, entry_id, legacy_timestamp, serialize_entry
from services.nutrition import (
    FOOD_CLASSES,
    FOOD_TABLE,
    health_score,
    lookup_food,
    nutrients_for,
    rule_based_insights,
    sum_nutrients,
)

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)

EMOJI_BY_NAME = {name: emoji for name, emoji in FOOD_CLASSES.values()}


def _allowed(filename, mimetype):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in settings.ALLOWED_EXTENSIONS or (mimetype or "").startswith("image/")


def _store(entry):
    user_data_collection.update_one(
        {"userId": get_jwt_identity()},
        {"$push": {"analysisHistory": entry}, "$set": {"lastUpdated": datetime.now(timezone.utc)}},
        upsert=True,
    )


@jwt_required()
def predict():
    file = request.files.get("file") or request.files.get("image")
    if not file or not file.filename:
        return error("Please attach an image file", 400)
    if not _allowed(file.filename, file.mimetype):
        return error("Unsupported file type. Use JPG, PNG or WEBP.", 415)

    raw = file.read()
    if not raw:
        return error("The uploaded file is empty", 400)

    try:
        result = analyzer.analyze(raw, file.mimetype)
    except analyzer.InvalidImageError as exc:
        return error(str(exc), 400)
    except Exception:
        logger.exception("Analysis failed")
        return error("We couldn't analyse this image. Please try another photo.", 500)

    saved = False
    if result["foods"]:
        try:
            _store(build_entry(result, secure_filename(file.filename) or "upload.jpg"))
            saved = True
        except Exception:
            logger.exception("Failed to store analysis")

    foods = result["foods"]
    return jsonify(
        {
            "id": result["id"],
            "timestamp": result["timestamp"].isoformat(),
            "saved": saved,
            "detection_source": result["detection_source"],
            "foods": foods,
            "totals": result["totals"],
            "insights": result["insights"],
            "image_size": result["image_size"],
            "annotated_image": result["annotated_image"],
            "message": None if foods else "We couldn't spot any food in this photo. Try a clearer, closer shot.",
            # ---- v1 compatibility ----
            "detections": {
                f["name"]: {
                    "confidence": f["confidence"],
                    "detected_weight": f["weight_g"],
                    "calories": f["nutrients"]["calories"],
                }
                for f in foods
            },
            "gemini_analysis": result["legacy_html"],
        }
    )


analysis_bp.add_url_rule("/predict", "predict", predict, methods=["POST"])
analysis_bp.add_url_rule("/api/analysis/predict", "predict_v2", predict, methods=["POST"])


@analysis_bp.get("/api/analysis/history")
@jwt_required()
def history():
    try:
        limit = max(1, min(500, int(request.args.get("limit", 200))))
    except ValueError:
        limit = 200
    doc = user_data_collection.find_one({"userId": get_jwt_identity()}, {"analysisHistory": 1})
    entries = [e for e in (doc or {}).get("analysisHistory", []) if e.get("timestamp")]
    entries.sort(key=lambda e: serialize_entry(e)["timestamp"] or "", reverse=True)
    return jsonify([serialize_entry(e) for e in entries[:limit]])


@analysis_bp.get("/api/analysis/<analysis_id>")
@jwt_required()
def get_analysis(analysis_id):
    doc = user_data_collection.find_one({"userId": get_jwt_identity()}, {"analysisHistory": 1})
    for e in (doc or {}).get("analysisHistory", []):
        if entry_id(e) == analysis_id:
            return jsonify(serialize_entry(e))
    return error("Analysis not found", 404)


@analysis_bp.delete("/api/analysis/<analysis_id>")
@jwt_required()
def delete_analysis(analysis_id):
    user_id = get_jwt_identity()
    if analysis_id.startswith("t"):
        ts = legacy_timestamp(analysis_id)
        if ts is None:
            return error("Analysis not found", 404)
        cond = {"timestamp": {"$gte": ts.replace(tzinfo=None), "$lt": (ts + timedelta(milliseconds=1)).replace(tzinfo=None)}}
    else:
        cond = {"id": analysis_id}
    result = user_data_collection.update_one({"userId": user_id}, {"$pull": {"analysisHistory": cond}})
    if not result.modified_count:
        return error("Analysis not found", 404)
    return jsonify({"message": "Deleted", "id": analysis_id})


@analysis_bp.get("/api/foods")
def list_foods():
    return jsonify(
        [
            {
                "name": f["name"],
                "emoji": EMOJI_BY_NAME.get(f["name"], "🍽️"),
                "unitWeight": f["unit_weight"],
                "nutrients": f["per_unit"],
            }
            for f in FOOD_TABLE.values()
        ]
    )


@analysis_bp.post("/api/analysis/manual")
@jwt_required()
def log_manual():
    """Log a meal without a photo, from the built-in Indian food table."""
    data = request.get_json(silent=True) or {}
    items = data.get("items") or [data]
    foods = []
    for item in items:
        food = lookup_food(item.get("name"))
        if not food:
            return error(f"Unknown food: {item.get('name')}", 400)
        try:
            servings = float(item.get("servings", 1))
        except (TypeError, ValueError):
            return error("Servings must be a number", 400)
        if not 0.1 <= servings <= 10:
            return error("Servings must be between 0.1 and 10", 400)
        foods.append(
            {
                "name": food["name"],
                "emoji": EMOJI_BY_NAME.get(food["name"], "🍽️"),
                "confidence": 1.0,
                "servings": servings,
                "count": 1,
                "weight_g": round(food["unit_weight"] * servings),
                "nutrients": nutrients_for(food, servings),
            }
        )
    if not foods:
        return error("Add at least one food", 400)

    totals = sum_nutrients(f["nutrients"] for f in foods)
    insights = rule_based_insights(foods, totals)
    insights["health_score"] = health_score(totals)
    result = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc),
        "detection_source": "manual",
        "foods": foods,
        "totals": totals,
        "insights": insights,
        "thumbnail": "",
    }
    entry = build_entry(result, "manual")
    _store(entry)
    return jsonify(serialize_entry(entry)), 201


@analysis_bp.post("/api/analysis/save")
@jwt_required()
def save_analysis():
    """v1 endpoint: /predict already stores results, so this only records explicit manual payloads."""
    data = request.get_json(silent=True)
    if not data:
        return error("No data provided", 400)
    totals = data.get("totalNutrients") or {}
    if not totals.get("calories"):
        return jsonify({"message": "Skipped saving analysis without nutrition data"}), 200
    entry = {
        "id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc),
        "imageFilename": secure_filename(str(data.get("imageFilename", ""))),
        "imageBase64": data.get("imageBase64", ""),
        "detectionSource": data.get("detectionSource", "manual"),
        "detectedFoods": data.get("detectedFoods", []),
        "totalNutrients": totals,
        "healthInsight": data.get("healthInsight", ""),
        "healthierOptions": data.get("healthierOptions", []),
        "funFact": data.get("funFact", ""),
    }
    _store(entry)
    return jsonify({"message": "Analysis saved successfully", "id": entry["id"]}), 200

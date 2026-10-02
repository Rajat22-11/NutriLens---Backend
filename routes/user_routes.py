from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from flask import Blueprint, request, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required

from config.config import user_collection, user_data_collection
from controllers.auth_controller import clean_profile, error, public_user
from services.history import build_summary, build_trends, goals_for
from services.nutrition import DEFAULT_GOALS

user_bp = Blueprint("user", __name__)

GOAL_LIMITS = {
    "calories": (800, 6000), "protein": (10, 400), "carbs": (20, 900), "fat": (10, 300),
    "fiber": (5, 100), "sugar": (0, 300), "sodium": (200, 8000), "cholesterol": (0, 1500),
}


def current_user():
    try:
        return user_collection.find_one({"_id": ObjectId(get_jwt_identity())})
    except (InvalidId, TypeError):
        return None


def tz_offset():
    try:
        return max(-840, min(840, int(request.args.get("tz", 0))))
    except (TypeError, ValueError):
        return 0


@jwt_required()
def get_profile():
    user = current_user()
    if not user:
        return error("User not found", 404)
    return jsonify(public_user(user))


user_bp.add_url_rule("/profile", "get_profile", get_profile, methods=["GET"])


@user_bp.route("/profile", methods=["PUT", "PATCH"])
@jwt_required()
def update_profile():
    user = current_user()
    if not user:
        return error("User not found", 404)
    updates, err = clean_profile(request.get_json(silent=True) or {}, partial=True)
    if err:
        return error(err, 400)
    if updates:
        updates["updatedAt"] = datetime.now(timezone.utc).isoformat()
        user_collection.update_one({"_id": user["_id"]}, {"$set": updates})
        user.update(updates)
    return jsonify(public_user(user))


@user_bp.route("/nutrition/goals", methods=["GET"])
@jwt_required()
def get_goals():
    doc = user_data_collection.find_one({"userId": get_jwt_identity()}, {"nutritionGoals": 1})
    return jsonify(goals_for(doc))


@user_bp.route("/nutrition/goals", methods=["POST", "PUT"])
@jwt_required()
def set_goals():
    data = request.get_json(silent=True) or {}
    if not current_user():
        return error("User not found", 404)
    doc = user_data_collection.find_one({"userId": get_jwt_identity()}, {"nutritionGoals": 1})
    goals = goals_for(doc)
    for key, (lo, hi) in GOAL_LIMITS.items():
        if key not in data or data[key] in (None, ""):
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            return error(f"Invalid value for {key}", 400)
        if not lo <= value <= hi:
            return error(f"{key.capitalize()} goal must be between {lo} and {hi}", 400)
        goals[key] = value
    user_data_collection.update_one(
        {"userId": get_jwt_identity()},
        {"$set": {"nutritionGoals": goals, "lastUpdated": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return jsonify({"message": "Nutrition goals updated successfully", "goals": goals})


@user_bp.route("/nutrition/goals/defaults", methods=["GET"])
def default_goals():
    return jsonify(DEFAULT_GOALS)


@user_bp.route("/nutrition/summary", methods=["GET"])
@jwt_required()
def nutrition_summary():
    doc = user_data_collection.find_one({"userId": get_jwt_identity()})
    return jsonify(build_summary(doc, tz_offset()))


@user_bp.route("/meal-trends", methods=["GET"])
@jwt_required()
def meal_trends():
    doc = user_data_collection.find_one({"userId": get_jwt_identity()})
    return jsonify(build_trends(doc, tz_offset()))

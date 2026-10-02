import logging
import re
from datetime import datetime, timezone

from flask import jsonify
from flask_jwt_extended import create_access_token
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash

from config.config import user_collection

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PROFILE_FIELDS = {"age": (1, 120), "height": (50, 260), "weight": (20, 350)}
GENDERS = {"male", "female", "other"}
ACTIVITY_LEVELS = {"sedentary", "light", "moderate", "active", "very_active"}


def error(message, status):
    # "error" for the v1 client, "message" for the v2 client
    return jsonify({"error": message, "message": message}), status


def public_user(user):
    return {
        "_id": str(user["_id"]),
        "id": str(user["_id"]),
        "name": user.get("name", ""),
        "email": user.get("email", ""),
        "age": user.get("age") or None,
        "height": user.get("height") or None,
        "weight": user.get("weight") or None,
        "gender": user.get("gender"),
        "activityLevel": user.get("activityLevel"),
        "customerType": user.get("customerType", "Basic"),
        "createdAt": user.get("createdAt"),
    }


def clean_profile(data, partial=False):
    """Validate optional profile fields. Returns (clean_dict, error_message)."""
    out = {}
    for field, (lo, hi) in PROFILE_FIELDS.items():
        if field not in data or data[field] in (None, ""):
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            return None, f"{field.capitalize()} must be a number"
        if not lo <= value <= hi:
            return None, f"{field.capitalize()} must be between {lo} and {hi}"
        out[field] = value
    if data.get("gender"):
        if data["gender"] not in GENDERS:
            return None, "Invalid gender"
        out["gender"] = data["gender"]
    if data.get("activityLevel"):
        if data["activityLevel"] not in ACTIVITY_LEVELS:
            return None, "Invalid activity level"
        out["activityLevel"] = data["activityLevel"]
    if "name" in data:
        name = str(data.get("name") or "").strip()
        if not name and not partial:
            return None, "Name is required"
        if name:
            out["name"] = name[:80]
    return out, None


def register_user(data):
    data = data or {}
    name = str(data.get("name") or "").strip()
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")

    if not name:
        return error("Please enter your name", 400)
    if not EMAIL_RE.match(email):
        return error("Please enter a valid email address", 400)
    if len(password) < 6:
        return error("Password must be at least 6 characters", 400)

    profile, err = clean_profile(data, partial=True)
    if err:
        return error(err, 400)

    if user_collection.find_one({"email": email}):
        return error("An account with this email already exists", 409)

    now = datetime.now(timezone.utc).isoformat()
    new_user = {
        **profile,
        "name": name,
        "email": email,
        "passwordHash": generate_password_hash(password),
        "customerType": data.get("customerType") if data.get("customerType") in ("Basic", "Premium") else "Basic",
        "createdAt": now,
        "updatedAt": now,
    }
    try:
        result = user_collection.insert_one(new_user)
    except DuplicateKeyError:
        return error("An account with this email already exists", 409)

    new_user["_id"] = result.inserted_id
    token = create_access_token(identity=str(result.inserted_id))
    logger.info("New user registered")
    return jsonify({"message": "Account created successfully!", "token": token, "user": public_user(new_user)}), 201


def login_user(data):
    data = data or {}
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    if not email or not password:
        return error("Email and password are required", 400)

    # Older accounts may have been stored with mixed-case emails
    user = user_collection.find_one({"email": email}) or user_collection.find_one(
        {"email": {"$regex": f"^{re.escape(email)}$", "$options": "i"}}
    )
    if not user or not check_password_hash(user.get("passwordHash", ""), password):
        return error("Invalid email or password", 401)

    token = create_access_token(identity=str(user["_id"]))
    return jsonify({"message": "Login successful!", "token": token, "user": public_user(user)}), 200

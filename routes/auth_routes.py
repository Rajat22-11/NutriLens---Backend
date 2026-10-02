from flask import Blueprint, request

from controllers.auth_controller import login_user, register_user
from routes.user_routes import get_profile

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/signup")
def signup():
    return register_user(request.get_json(silent=True))


@auth_bp.post("/login")
def login():
    return login_user(request.get_json(silent=True))


# Kept for the v1 frontend, which reads the profile from /api/auth/profile
auth_bp.add_url_rule("/profile", "profile", get_profile, methods=["GET"])

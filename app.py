"""NutriLens API — Flask application factory.

Run locally:   python app.py
Production:    gunicorn -c gunicorn.conf.py app:app
"""
import logging
import os
import re
import time
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from werkzeug.exceptions import HTTPException

from config.config import ensure_indexes, ping_db, settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nutrilens")

BOOT_TIME = time.time()


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    app.json.sort_keys = False

    secret = settings.JWT_SECRET
    if not secret:
        if settings.ENV == "production":
            logger.warning("JWT_SECRET is not set! Using an ephemeral secret; tokens reset on restart.")
        secret = os.urandom(32).hex()
    app.config["JWT_SECRET_KEY"] = secret
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = settings.JWT_EXPIRES

    CORS(
        app,
        resources={r"/*": {"origins": settings.CORS_ORIGINS + [re.compile(settings.CORS_ORIGIN_REGEX)]}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        max_age=3600,
    )

    jwt = JWTManager(app)

    def _auth_error(message):
        return jsonify({"error": message, "message": message}), 401

    jwt.expired_token_loader(lambda _h, _p: _auth_error("Your session has expired. Please sign in again."))
    jwt.invalid_token_loader(lambda reason: _auth_error("Invalid session. Please sign in again."))
    jwt.unauthorized_loader(lambda reason: _auth_error("Please sign in to continue."))

    from routes.analysis_routes import analysis_bp
    from routes.auth_routes import auth_bp
    from routes.user_routes import user_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(user_bp, url_prefix="/api/user")
    app.register_blueprint(analysis_bp)

    @app.get("/")
    def root():
        return jsonify({"name": "NutriLens API", "version": settings.VERSION, "docs": "/api/health"})

    @app.get("/api/health")
    def health():
        from services import gemini
        from services.detector import detector

        db_ok = ping_db() if request.args.get("deep") else None
        return jsonify(
            {
                "status": "online",
                "version": settings.VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptimeSeconds": round(time.time() - BOOT_TIME),
                "yoloReady": detector.ready,
                "geminiConfigured": gemini.available(),
                "database": db_ok,
            }
        )

    @app.errorhandler(413)
    def too_large(_e):
        msg = f"Image is too large. Max size is {settings.MAX_UPLOAD_MB} MB."
        return jsonify({"error": msg, "message": msg}), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return jsonify({"error": e.description, "message": e.description}), e.code

    @app.errorhandler(Exception)
    def unhandled(e):
        logger.exception("Unhandled error on %s %s", request.method, request.path)
        msg = "Something went wrong on our side. Please try again."
        return jsonify({"error": msg, "message": msg}), 500

    ensure_indexes()

    if settings.YOLO_MODE == "eager":
        from services.detector import detector

        detector.load()

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")

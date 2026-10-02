"""Application settings and MongoDB wiring.

All configuration comes from environment variables so the same code runs
locally, in tests (``MONGO_URI=mongomock://localhost``) and on Render.
"""
import logging
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    ENV = os.getenv("FLASK_ENV", os.getenv("APP_ENV", "production"))
    VERSION = "2.0.0"

    # Security
    JWT_SECRET = os.getenv("JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
    JWT_EXPIRES = timedelta(days=_env_int("JWT_EXPIRES_DAYS", 7))

    # Database
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "nutritionApp")
    MONGO_TIMEOUT_MS = _env_int("MONGO_TIMEOUT_MS", 8000)
    MONGO_MAX_POOL_SIZE = _env_int("MONGO_MAX_POOL_SIZE", 10)

    # AI
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", os.path.join(BASE_DIR, "weights", "best.pt"))
    YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
    YOLO_IMG_SIZE = _env_int("YOLO_IMG_SIZE", 416)
    # "eager" loads the model at boot, "lazy" on first request, "off" disables it
    YOLO_MODE = os.getenv("YOLO_MODE", "eager").lower()

    # Uploads
    MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 10)
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

    # CORS: explicit origins (comma separated) + any NutriLens deployment on Render/Vercel
    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,"
            "https://nutrilens.vercel.app,https://nutrilens-frontend.vercel.app,"
            "https://nutrilens-frontend.onrender.com",
        ).split(",")
        if o.strip()
    ]
    CORS_ORIGIN_REGEX = os.getenv(
        "CORS_ORIGIN_REGEX", r"^https://nutrilens[a-z0-9-]*\.(onrender\.com|vercel\.app)$"
    )

    FOOD_DATA_FILE = os.path.join(BASE_DIR, "food_data", "food_calorie_data2.csv")


settings = Settings()


def _create_client():
    uri = settings.MONGO_URI
    if uri.startswith("mongomock://"):
        import mongomock

        logger.info("Using in-memory mongomock database")
        return mongomock.MongoClient()

    from pymongo import MongoClient

    kwargs = dict(
        serverSelectionTimeoutMS=settings.MONGO_TIMEOUT_MS,
        connectTimeoutMS=settings.MONGO_TIMEOUT_MS,
        maxPoolSize=settings.MONGO_MAX_POOL_SIZE,
        retryWrites=True,
        appname="nutrilens-api",
    )
    if uri.startswith("mongodb+srv://") or "tls=true" in uri.lower() or "ssl=true" in uri.lower():
        import certifi

        kwargs["tlsCAFile"] = certifi.where()
    # MongoClient connects lazily, so a slow/offline database never blocks boot.
    return MongoClient(uri, **kwargs)


client = _create_client()
db = client[settings.MONGO_DB_NAME]
user_collection = db["users"]
user_data_collection = db["userData"]


def ensure_indexes():
    try:
        user_collection.create_index("email", unique=True)
        user_data_collection.create_index("userId", unique=True)
    except Exception as exc:  # pragma: no cover - depends on live DB
        logger.warning("Could not ensure MongoDB indexes: %s", exc)


def ping_db():
    try:
        client.admin.command("ping")
        return True
    except Exception as exc:
        logger.warning("MongoDB ping failed: %s", exc)
        return False

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MONGO_URI", "mongomock://localhost")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("YOLO_MODE", os.getenv("TEST_YOLO_MODE", "lazy"))
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

from app import app as flask_app  # noqa: E402
from config.config import user_collection, user_data_collection  # noqa: E402
from services import gemini  # noqa: E402
from services.history import build_summary  # noqa: E402
from services.nutrition import lookup_food  # noqa: E402

SAMPLES = ROOT / "samples"


@pytest.fixture()
def client():
    user_collection.delete_many({})
    user_data_collection.delete_many({})
    return flask_app.test_client()


def signup(client, email="cook@example.com"):
    r = client.post("/api/auth/signup", json={"name": "Cook", "email": email, "password": "secret12"})
    assert r.status_code == 201, r.json
    return {"Authorization": f"Bearer {r.json['token']}"}


def upload(client, headers, path):
    with open(path, "rb") as fh:
        return client.post(
            "/predict", data={"file": (fh, path.name)}, headers=headers, content_type="multipart/form-data"
        )


def test_food_lookup_handles_multiword_names():
    assert lookup_food("Vadapav")["name"] == "Vada Pav"
    assert lookup_food("Chole Bhature 🍽️")["name"] == "Chole Bhature"
    assert lookup_food("paneer")["name"] == "Paneer Tikka"
    assert lookup_food("pizza") is None


def test_signup_login_and_profile(client):
    r = client.post("/api/auth/signup", json={"name": "A", "email": "bad", "password": "secret12"})
    assert r.status_code == 400
    headers = signup(client, "Mixed@Case.com")
    assert client.post("/api/auth/signup", json={"name": "A", "email": "mixed@case.com", "password": "secret12"}).status_code == 409
    r = client.post("/api/auth/login", json={"email": "MIXED@case.com", "password": "secret12"})
    assert r.status_code == 200 and r.json["token"]
    assert client.post("/api/auth/login", json={"email": "mixed@case.com", "password": "nope"}).status_code == 401

    r = client.put("/api/user/profile", json={"age": 30, "height": 170, "weight": 70, "gender": "female"}, headers=headers)
    assert r.status_code == 200 and r.json["weight"] == 70
    assert client.put("/api/user/profile", json={"age": 500}, headers=headers).status_code == 400
    assert client.get("/api/auth/profile", headers=headers).json["gender"] == "female"


def test_auth_required(client):
    r = client.get("/api/analysis/history")
    assert r.status_code == 401 and "sign in" in r.json["message"]
    r = client.get("/api/analysis/history", headers={"Authorization": "Bearer junk"})
    assert r.status_code == 401


def test_goals_roundtrip(client):
    headers = signup(client)
    assert client.get("/api/user/nutrition/goals", headers=headers).json["calories"] == 2000
    r = client.post("/api/user/nutrition/goals", json={"calories": 2200, "protein": 120}, headers=headers)
    assert r.status_code == 200
    goals = client.get("/api/user/nutrition/goals", headers=headers).json
    assert goals["calories"] == 2200 and goals["protein"] == 120 and goals["fat"] == 70
    assert client.post("/api/user/nutrition/goals", json={"calories": 50}, headers=headers).status_code == 400


def test_predict_rejects_bad_input(client):
    headers = signup(client)
    assert client.post("/predict", headers=headers).status_code == 400
    r = client.post(
        "/predict", data={"file": (open(__file__, "rb"), "notes.txt")}, headers=headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 415
    from io import BytesIO

    r = client.post(
        "/predict", data={"file": (BytesIO(b"not an image"), "x.jpg")}, headers=headers,
        content_type="multipart/form-data",
    )
    assert r.status_code == 400


def test_predict_with_yolo_and_history(client):
    headers = signup(client)
    r = upload(client, headers, SAMPLES / "Vadapav_Resize_00000414_resized.png")
    assert r.status_code == 200
    data = r.json
    assert data["detection_source"] == "yolo"
    assert data["foods"][0]["name"] == "Vada Pav"
    assert data["totals"]["calories"] > 0
    assert data["annotated_image"]
    assert data["insights"]["healthier_options"]
    assert "Vada Pav" in data["detections"] and "food-title" in data["gemini_analysis"]
    assert data["saved"] is True

    history = client.get("/api/analysis/history", headers=headers).json
    assert len(history) == 1 and history[0]["id"] == data["id"]
    assert client.get(f"/api/analysis/{data['id']}", headers=headers).json["foods"][0]["name"] == "Vada Pav"

    summary = client.get("/api/user/nutrition/summary?tz=-330", headers=headers).json
    assert summary["stats"]["totalScans"] == 1 and summary["stats"]["streak"] == 1
    assert summary["daily"][-1]["calories"] == data["totals"]["calories"]
    trends = client.get("/api/user/meal-trends", headers=headers).json
    assert trends["commonFoods"][0]["name"] == "Vada Pav"

    assert client.delete(f"/api/analysis/{data['id']}", headers=headers).status_code == 200
    assert client.get("/api/analysis/history", headers=headers).json == []


def test_gemini_fallback_when_yolo_finds_nothing(client, monkeypatch):
    headers = signup(client)
    monkeypatch.setattr(gemini, "available", lambda: True)
    monkeypatch.setattr(
        gemini,
        "analyze_image",
        lambda *_: {
            "is_food": True,
            "foods": [{"name": "Masala Oats", "emoji": "🥣", "weight_g": 250, "confidence": 0.9,
                       "nutrients": {"calories": 320, "protein": 11, "carbs": 52, "fat": 7, "fiber": 6,
                                     "sugar": 4, "sodium": 480, "cholesterol": 0}}],
            "summary": "Packaged oats", "health_insight": "Good fibre.",
            "healthier_options": ["Add nuts"], "fun_fact": "Oats are whole grains.",
        },
    )
    r = upload(client, headers, SAMPLES / "text.png")
    assert r.status_code == 200
    assert r.json["detection_source"] == "gemini"
    assert r.json["foods"][0]["name"] == "Masala Oats"
    assert r.json["totals"]["calories"] == 320
    assert r.json["annotated_image"] is None


def test_no_food_detected(client, monkeypatch):
    headers = signup(client)
    monkeypatch.setattr(gemini, "analyze_image", lambda *_: None)
    r = upload(client, headers, SAMPLES / "text.png")
    assert r.status_code == 200 and r.json["foods"] == [] and r.json["detection_source"] == "none"
    assert client.get("/api/analysis/history", headers=headers).json == []


def test_manual_log(client):
    headers = signup(client)
    foods = client.get("/api/foods").json
    assert any(f["name"] == "Dosa" for f in foods)
    r = client.post("/api/analysis/manual", json={"items": [{"name": "Dosa", "servings": 2}]}, headers=headers)
    assert r.status_code == 201 and r.json["totals"]["calories"] == 600
    assert client.post("/api/analysis/manual", json={"name": "Pizza"}, headers=headers).status_code == 400


def test_legacy_entries_are_readable_and_deletable(client):
    headers = signup(client)
    uid = user_collection.find_one({})["_id"]
    ts = datetime(2025, 3, 22, 8, 30, 12, 345000)
    user_data_collection.insert_one(
        {
            "userId": str(uid),
            "analysisHistory": [
                {
                    "timestamp": ts,
                    "imageBase64": "",
                    "detectionSource": "yolo",
                    "detectedFoods": [{"name": "Biryani 🍛", "weight": 300, "nutrients": {"calories": 600}}],
                    "totalNutrients": {"calories": 600, "protein": 20},
                    "healthInsight": "Rich",
                }
            ],
        }
    )
    history = client.get("/api/analysis/history", headers=headers).json
    assert history[0]["id"].startswith("t") and history[0]["totals"]["calories"] == 600
    assert client.delete(f"/api/analysis/{history[0]['id']}", headers=headers).status_code == 200
    assert client.get("/api/analysis/history", headers=headers).json == []


def test_summary_groups_by_local_day():
    now = datetime(2026, 1, 10, 20, 0, tzinfo=timezone.utc)  # 01:30 on Jan 11 in India
    doc = {
        "analysisHistory": [
            {"timestamp": now - timedelta(hours=1), "totalNutrients": {"calories": 500}},
            {"timestamp": now - timedelta(days=1), "totalNutrients": {"calories": 300}},
        ]
    }
    ist = build_summary(doc, -330, now=now)
    assert ist["daily"][-1]["date"] == "2026-01-11"
    assert ist["daily"][-1]["calories"] == 500
    assert ist["stats"]["streak"] == 2
    utc = build_summary(doc, 0, now=now)
    assert utc["daily"][-1]["calories"] == 500 and utc["daily"][-2]["calories"] == 300


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["status"] == "online"

# 🥗 NutriLens — API

The Flask API behind NutriLens. It recognises Indian dishes in a photo, estimates portions and nutrition, and keeps each user's food diary, goals and trends.

**Live:** https://nutrilens-backend-aet6.onrender.com/api/health · **Web app:** [NutriLens---Frontend](https://github.com/Rajat22-11/NutriLens---Frontend)

## How analysis works

1. **Detect**: a custom YOLOv5 model (`weights/best.pt`) finds the dishes on the plate. Images are letterboxed and boxes are mapped back to the original photo.
2. **Estimate**: each dish's portion comes from how much of the frame it fills. Duplicate detections of the same dish are merged, and nutrition comes from `food_data/food_calorie_data2.csv`.
3. **Explain**: Gemini (`gemini-2.5-flash`, structured JSON) writes a summary, a health insight, healthier swaps and a fun fact. Without a Gemini key, built-in rule-based insights are used instead.
4. **Fallback**: if the model finds no known dish (for example packaged food or a nutrition label), Gemini vision identifies the food and estimates its nutrition.
5. **Save**: the result is stored in the user's diary in MongoDB, with a thumbnail and a 0–100 health score.

**Dishes the model knows:** Biryani, Chole Bhature, Dabeli, Dal, Dhokla, Dosa, Jalebi, Kathi Roll, Kofta, Naan, Pakora, Paneer Tikka, Panipuri, Pav Bhaji, Vada Pav.

## Tech stack

Python 3.11 · Flask 3 · Gunicorn · PyTorch (CPU) + YOLOv5 · Google Gen AI SDK · MongoDB (PyMongo) · Flask-JWT-Extended · Pillow · pandas

## Getting started

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in the values below
python app.py                 # http://localhost:5000
```

Run the tests. They use an in-memory database, so no MongoDB or API key is needed:

```bash
MONGO_URI=mongomock://localhost python -m pytest -q tests
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `MONGO_URI` | ✅ | MongoDB connection string (`mongomock://localhost` for an in-memory DB) |
| `JWT_SECRET` | ✅ | Secret used to sign login tokens |
| `GOOGLE_API_KEY` | – | Gemini API key; enables AI insights and the vision fallback |
| `GEMINI_MODEL` | – | Defaults to `gemini-2.5-flash` |
| `MONGO_DB_NAME` | – | Defaults to `nutritionApp` |
| `CORS_ORIGINS` | – | Comma-separated extra allowed origins. `https://nutrilens*.onrender.com` and `*.vercel.app` are always allowed. |
| `YOLO_MODE` | – | `eager` (load at start-up, default), `lazy` or `off` |
| `JWT_EXPIRES_DAYS`, `MAX_UPLOAD_MB`, `YOLO_CONF_THRESHOLD` | – | Defaults: 7 days, 10 MB, 0.35 |

## API

All endpoints except auth, health and the food list need an `Authorization: Bearer <token>` header. Errors are JSON with `message` and `error` fields.

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/signup` | Create an account → `{ token, user }` |
| POST | `/api/auth/login` | Sign in → `{ token, user }` |
| GET / PUT | `/api/user/profile` | Read or update name, age, height, weight, gender, activity level |
| GET / PUT | `/api/user/nutrition/goals` | Daily goals for 8 nutrients |
| GET | `/api/user/nutrition/summary?tz=<offset>` | Daily / weekly / monthly totals, today's progress, streak and stats |
| GET | `/api/user/meal-trends?tz=<offset>` | Favourite foods, meal timings, weekday pattern |
| POST | `/predict` (alias `/api/analysis/predict`) | Analyse a photo (multipart `file`) |
| GET | `/api/analysis/history` | Food diary, newest first |
| GET / DELETE | `/api/analysis/<id>` | Read or delete one meal |
| GET | `/api/foods` | Built-in food table |
| POST | `/api/analysis/manual` | Log a meal from the food table without a photo |
| GET | `/api/health` | Status, model readiness, Gemini configured (`?deep=1` also pings MongoDB) |

`tz` is the browser's `Date#getTimezoneOffset()` value, for example `-330` for India.

## Project structure

```
app.py                  # Flask app factory, CORS, JWT, error handlers
gunicorn.conf.py        # production server settings
config/config.py        # settings from environment variables + MongoDB client
controllers/            # sign-up / login / profile validation
routes/                 # auth, user (profile, goals, summary, trends), analysis
services/
├── analyzer.py         # photo → detections → nutrition → insights
├── detector.py         # YOLOv5 model loading and inference
├── gemini.py           # Gemini insights and vision fallback
├── history.py          # diary entries, summaries, streaks, trends
└── nutrition.py        # food table lookup, portions, health score
weights/best.pt         # trained YOLOv5 weights
yolov5/                 # vendored YOLOv5 code
tests/                  # pytest suite
samples/                # sample food photos
```

## Deployment

Runs on Render as a Python web service that redeploys automatically on every push to `main`; `render.yaml` describes it.
The first request after the free instance has been idle takes up to a minute while the model loads.

## License

For educational and personal use. YOLOv5 is licensed under AGPL-3.0.

"""Helpers for the per-user analysis history stored in the ``userData`` collection."""
from datetime import datetime, timedelta, timezone

from services.nutrition import DEFAULT_GOALS, NUTRIENT_KEYS, coerce_nutrients, sum_nutrients


def as_utc(ts):
    if not isinstance(ts, datetime):
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def entry_id(entry):
    """Stable id; legacy entries without one get an id derived from their timestamp."""
    if entry.get("id"):
        return entry["id"]
    ts = as_utc(entry.get("timestamp"))
    return f"t{int(ts.timestamp() * 1000)}" if ts else ""


def legacy_timestamp(entry_id_value):
    try:
        ms = int(entry_id_value[1:])
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def goals_for(user_doc):
    goals = dict(DEFAULT_GOALS)
    for k, v in ((user_doc or {}).get("nutritionGoals") or {}).items():
        if k in goals:
            try:
                goals[k] = float(v)
            except (TypeError, ValueError):
                pass
    return goals


def build_entry(result, filename):
    """Convert an analyzer result into the stored history entry."""
    return {
        "id": result["id"],
        "timestamp": result["timestamp"],
        "imageFilename": filename,
        "imageBase64": result["thumbnail"],
        "detectionSource": result["detection_source"],
        "detectedFoods": [
            {
                "name": f["name"],
                "emoji": f.get("emoji"),
                "weight": f.get("weight_g", 0),
                "servings": f.get("servings", 1),
                "count": f.get("count", 1),
                "confidence": f.get("confidence"),
                "nutrients": f["nutrients"],
            }
            for f in result["foods"]
        ],
        "totalNutrients": result["totals"],
        "summary": result["insights"].get("summary", ""),
        "healthInsight": result["insights"].get("health_insight", ""),
        "healthierOptions": result["insights"].get("healthier_options", []),
        "funFact": result["insights"].get("fun_fact", ""),
        "healthScore": result["insights"].get("health_score"),
    }


def serialize_entry(entry):
    totals = coerce_nutrients(entry.get("totalNutrients"))
    foods = [
        {
            "name": f.get("name", "Unknown"),
            "emoji": f.get("emoji") or "🍽️",
            "weight": f.get("weight", 0),
            "servings": f.get("servings", 1),
            "confidence": f.get("confidence"),
            "nutrients": coerce_nutrients(f.get("nutrients")),
        }
        for f in entry.get("detectedFoods", [])
    ]
    ts = as_utc(entry.get("timestamp"))
    return {
        "id": entry_id(entry),
        "timestamp": ts.isoformat() if ts else None,
        "imageBase64": entry.get("imageBase64", ""),
        "detectionSource": entry.get("detectionSource", ""),
        "foods": foods,
        "foodItems": foods,  # v1 name
        "totals": totals,
        "totalCalories": totals["calories"],
        "totalProtein": totals["protein"],
        "totalCarbs": totals["carbs"],
        "totalFat": totals["fat"],
        "insights": {
            "summary": entry.get("summary", ""),
            "health_insight": entry.get("healthInsight", ""),
            "healthier_options": entry.get("healthierOptions", []),
            "fun_fact": entry.get("funFact", ""),
            "health_score": entry.get("healthScore"),
        },
    }


def _local(ts, offset):
    return as_utc(ts) - offset


def build_summary(user_doc, tz_offset_minutes=0, now=None):
    """Aggregate history by the user's local calendar day/week/month.

    ``tz_offset_minutes`` follows JavaScript's ``Date#getTimezoneOffset`` (UTC - local),
    e.g. -330 for India.
    """
    offset = timedelta(minutes=tz_offset_minutes)
    now_local = (now or datetime.now(timezone.utc)) - offset
    today = now_local.date()
    goals = goals_for(user_doc)
    history = [e for e in (user_doc or {}).get("analysisHistory", []) if as_utc(e.get("timestamp"))]

    by_day = {}
    for e in history:
        day = _local(e["timestamp"], offset).date()
        by_day.setdefault(day, []).append(coerce_nutrients(e.get("totalNutrients")))
    day_totals = {d: sum_nutrients(items) for d, items in by_day.items()}
    zero = {k: 0.0 for k in NUTRIENT_KEYS}

    def point(name, totals, **extra):
        return {
            "name": name,
            **{k: round(totals.get(k, 0), 1) for k in ("calories", "protein", "carbs", "fat", "fiber")},
            "target": goals["calories"],
            **extra,
        }

    daily = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        daily.append(point(d.strftime("%a"), day_totals.get(d, zero), date=d.isoformat(), meals=len(by_day.get(d, []))))

    def averaged(days):
        logged = [day_totals[d] for d in days if d in day_totals]
        if not logged:
            return zero, 0
        return {k: sum(t[k] for t in logged) / len(logged) for k in NUTRIENT_KEYS}, len(logged)

    weekly = []
    week_start = today - timedelta(days=today.weekday())
    for i in range(7, -1, -1):
        start = week_start - timedelta(weeks=i)
        avg, n = averaged([start + timedelta(days=j) for j in range(7)])
        weekly.append(point(start.strftime("%d %b"), avg, daysLogged=n, start=start.isoformat()))

    monthly = []
    y, m = today.year, today.month
    months = []
    for _ in range(6):
        months.append((y, m))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    for y, m in reversed(months):
        days = [d for d in day_totals if d.year == y and d.month == m]
        avg, n = averaged(days)
        monthly.append(point(datetime(y, m, 1).strftime("%b"), avg, daysLogged=n))

    today_totals = day_totals.get(today, dict(zero))
    macro_total = today_totals["protein"] + today_totals["carbs"] + today_totals["fat"]
    distribution = []
    if macro_total > 0:
        distribution = [
            {"name": "Protein", "value": round(today_totals["protein"] / macro_total * 100, 1), "color": "#6366f1"},
            {"name": "Carbs", "value": round(today_totals["carbs"] / macro_total * 100, 1), "color": "#f59e0b"},
            {"name": "Fat", "value": round(today_totals["fat"] / macro_total * 100, 1), "color": "#ec4899"},
        ]

    units = {"calories": "kcal", "sodium": "mg", "cholesterol": "mg"}
    colors = {
        "calories": "#16a34a", "protein": "#6366f1", "carbs": "#f59e0b", "fat": "#ec4899",
        "fiber": "#14b8a6", "sugar": "#f97316", "sodium": "#64748b", "cholesterol": "#a855f7",
    }
    goal_progress = [
        {
            "key": k,
            "name": k.capitalize(),
            "current": round(today_totals.get(k, 0), 1),
            "goal": goals[k],
            "unit": units.get(k, "g"),
            "color": colors[k],
        }
        for k in NUTRIENT_KEYS
    ]

    streak = 0
    d = today if today in day_totals else today - timedelta(days=1)
    while d in day_totals:
        streak += 1
        d -= timedelta(days=1)

    all_days = list(day_totals.values())
    stats = {
        "totalScans": len(history),
        "daysLogged": len(all_days),
        "streak": streak,
        "avgDailyCalories": round(sum(t["calories"] for t in all_days) / len(all_days)) if all_days else 0,
        "mealsToday": len(by_day.get(today, [])),
    }

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "today": today_totals,
        "goals": goals,
        "nutrientDistribution": distribution,
        "goalProgress": goal_progress,
        "stats": stats,
    }


def build_trends(user_doc, tz_offset_minutes=0):
    offset = timedelta(minutes=tz_offset_minutes)
    history = [e for e in (user_doc or {}).get("analysisHistory", []) if as_utc(e.get("timestamp"))]

    counter = {}
    calories = {}
    for e in history:
        for f in e.get("detectedFoods", []):
            name = f.get("name", "Unknown")
            counter[name] = counter.get(name, 0) + 1
            calories[name] = calories.get(name, 0) + float(coerce_nutrients(f.get("nutrients"))["calories"])
    common = sorted(
        ({"name": k, "count": v, "avgCalories": round(calories[k] / v)} for k, v in counter.items()),
        key=lambda x: (-x["count"], x["name"]),
    )[:6]

    slots = [("Breakfast", 5, 11), ("Lunch", 11, 16), ("Snacks", 16, 19), ("Dinner", 19, 29)]
    timing_counts = {name: 0 for name, _, _ in slots}
    weekday = {d: 0 for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
    for e in history:
        local = _local(e["timestamp"], offset)
        hour = local.hour if local.hour >= 5 else local.hour + 24
        for name, lo, hi in slots:
            if lo <= hour < hi:
                timing_counts[name] += 1
        weekday[local.strftime("%a")] += 1

    return {
        "commonFoods": common,
        "mealTimings": [{"name": k, "count": v} for k, v in timing_counts.items()],
        "weekdayPatterns": [{"name": k, "count": v} for k, v in weekday.items()],
    }

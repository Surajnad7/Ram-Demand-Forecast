"""
app.py
======
FastAPI service for the weekly 16 GB RAM unit demand forecast model.

Endpoints
---------
GET   /health          → liveness / readiness check
POST  /predict         → N-week forecast (1-12, default 3)
POST  /train           → retrain the model in-process
GET   /api/history     → historical demand data
GET   /api/metrics     → latest training metrics
GET   /api/summary     → one-shot dashboard payload
GET   /api/assistant   → natural-language Q&A about forecasts
GET   /                → serve the SPA frontend

Run
---
    uvicorn src.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/prophet_ram.pkl")
METRICS_PATH = Path("models/metrics.json")
DATA_PATH = Path("data/ram_demand.csv")
SPA_PATH = Path("static/index.html")

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="16 GB RAM Demand Forecast API",
    version="1.0.0",
    description="Weekly 16 GB RAM unit demand forecast service powered by Prophet.",
)


# ── Request / Response schemas ───────────────────────────────────────
class PredictRequest(BaseModel):
    horizon: int = Field(default=3, ge=1, le=12, description="Weeks to forecast (1-12)")


# ── /health ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_PATH.exists()}


# ── /predict ─────────────────────────────────────────────────────────
@app.post("/predict")
def predict(body: PredictRequest):
    from src.predict import forecast

    results = forecast(horizon=body.horizon)
    return {"status": "ok", "horizon_weeks": body.horizon, "forecasts": results}


# ── /train ───────────────────────────────────────────────────────────
@app.post("/train")
def train():
    from src.train import train_and_evaluate

    metrics = train_and_evaluate()
    return {"status": "ok", "metrics": metrics}


# ── /api/history ─────────────────────────────────────────────────────
@app.get("/api/history")
def history(limit: int = Query(default=52, ge=1, le=260)):
    if not DATA_PATH.exists():
        return {"error": "No data file found"}
    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
    df = df.tail(limit)
    records = []
    for _, row in df.iterrows():
        records.append({
            "ds": row["ds"].strftime("%Y-%m-%d"),
            "units_sold": int(row["units_sold"]),
            "avg_spot_price_usd": round(float(row["avg_spot_price_usd"]), 2),
            "avg_temperature_f": round(float(row["avg_temperature_f"]), 1),
            "is_holiday_week": int(row["is_holiday_week"]),
            "is_back_to_school": int(row["is_back_to_school"]),
            "is_black_friday_week": int(row["is_black_friday_week"]),
            "is_prime_day_week": int(row["is_prime_day_week"]),
            "pct_supply_constraint": round(float(row["pct_supply_constraint"]), 3),
            "new_gpu_launch": int(row["new_gpu_launch"]),
        })
    return {"count": len(records), "data": records}


# ── /api/metrics ─────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return {"error": "No metrics file found"}
    return json.loads(METRICS_PATH.read_text())


# ── /api/summary ─────────────────────────────────────────────────────
@app.get("/api/summary")
def summary():
    result: dict = {"model_loaded": MODEL_PATH.exists()}

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["ds"])
        result["total_weeks"] = len(df)
        result["date_range"] = {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end": df["ds"].max().strftime("%Y-%m-%d"),
        }
        latest = df.iloc[-1]
        result["latest_week"] = {
            "ds": latest["ds"].strftime("%Y-%m-%d"),
            "units_sold": int(latest["units_sold"]),
            "avg_price": round(float(latest["avg_spot_price_usd"]), 2),
            "temperature": round(float(latest["avg_temperature_f"]), 1),
            "supply_constraint": round(float(latest["pct_supply_constraint"]), 3),
        }

        last_13 = df.tail(13)
        result["recent_stats"] = {
            "avg_units_13w": int(last_13["units_sold"].mean()),
            "max_units_13w": int(last_13["units_sold"].max()),
            "min_units_13w": int(last_13["units_sold"].min()),
            "std_units_13w": int(last_13["units_sold"].std()),
            "total_units_13w": int(last_13["units_sold"].sum()),
        }

        recent = df.tail(4)
        result["recent_trend"] = {
            "avg_units_4w": int(recent["units_sold"].mean()),
            "avg_price_4w": round(float(recent["avg_spot_price_usd"].mean()), 2),
            "avg_temp_4w": round(float(recent["avg_temperature_f"].mean()), 1),
            "supply_constrained_weeks_4w": int((recent["pct_supply_constraint"] > 0.05).sum()),
            "gpu_launch_weeks_4w": int(recent["new_gpu_launch"].sum()),
        }

    if METRICS_PATH.exists():
        result["metrics"] = json.loads(METRICS_PATH.read_text())

    return result


# ── /api/assistant ───────────────────────────────────────────────────
@app.get("/api/assistant")
def assistant(q: str = Query(..., min_length=1)):
    from src.predict import forecast

    query = q.lower().strip()
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["ds"]) if DATA_PATH.exists() else None
    except Exception:
        df = None

    # ── Route: forecast / predict ────────────────────────────────────
    if any(kw in query for kw in ("forecast", "predict", "next", "future", "expect")):
        preds = forecast(horizon=3)
        total = sum(p["yhat"] for p in preds)
        table = "| Week | Forecast | Lower 95% | Upper 95% |\n|------|----------|-----------|----------|\n"
        for p in preds:
            table += f"| {p['ds']} | {p['yhat']:,} | {p['yhat_lower']:,} | {p['yhat_upper']:,} |\n"
        answer = f"The 3-week forecast totals **{total:,} units**.\n\n{table}"
        return {"answer": answer, "data": preds}

    # ── Route: metrics / accuracy ────────────────────────────────────
    if any(kw in query for kw in ("metric", "accuracy", "error", "mape", "rmse", "r2", "performance")):
        if METRICS_PATH.exists():
            m = json.loads(METRICS_PATH.read_text())
            answer = (
                f"**Model Performance**\n\n"
                f"- MAPE: {m['mape']:.2%}\n"
                f"- RMSE: {m['rmse']:.1f} units\n"
                f"- MAE: {m['mae']:.1f} units\n"
                f"- R²: {m['r2']:.4f}\n"
                f"- Quality Gate: {'✅ PASSED' if m['quality_gate_passed'] else '❌ FAILED'}\n"
            )
            return {"answer": answer, "data": m}
        return {"answer": "No metrics available yet. Train the model first."}

    # ── Route: historical / sales stats ──────────────────────────────
    if any(kw in query for kw in ("history", "historical", "past", "trend", "sales", "sold", "average", "demand")):
        if df is not None:
            last13 = df.tail(13)
            answer = (
                f"**Last 13 Weeks**\n\n"
                f"- Avg weekly units: {int(last13['units_sold'].mean()):,}\n"
                f"- Max: {int(last13['units_sold'].max()):,}\n"
                f"- Min: {int(last13['units_sold'].min()):,}\n"
                f"- Total: {int(last13['units_sold'].sum()):,}\n"
            )
            return {"answer": answer}
        return {"answer": "No historical data available."}

    # ── Route: price ─────────────────────────────────────────────────
    if any(kw in query for kw in ("price", "cost", "expensive", "cheap", "spot")):
        if df is not None:
            answer = (
                f"**Pricing Info**\n\n"
                f"- Current avg spot price: ${df['avg_spot_price_usd'].iloc[-1]:.2f}\n"
                f"- 13-week avg: ${df.tail(13)['avg_spot_price_usd'].mean():.2f}\n"
                f"- All-time range: ${df['avg_spot_price_usd'].min():.2f} – ${df['avg_spot_price_usd'].max():.2f}\n"
            )
            return {"answer": answer}
        return {"answer": "No data available."}

    # ── Route: supply ────────────────────────────────────────────────
    if any(kw in query for kw in ("supply", "shortage", "constraint", "chip")):
        if df is not None:
            recent = df.tail(13)
            answer = (
                f"**Supply Constraints (13w)**\n\n"
                f"- Avg constraint: {recent['pct_supply_constraint'].mean():.1%}\n"
                f"- Max constraint: {recent['pct_supply_constraint'].max():.1%}\n"
                f"- Constrained weeks (>5%): {int((recent['pct_supply_constraint'] > 0.05).sum())}\n"
            )
            return {"answer": answer}
        return {"answer": "No data available."}

    # ── Route: GPU launch ────────────────────────────────────────────
    if any(kw in query for kw in ("gpu", "launch", "cpu", "platform", "nvidia", "amd")):
        if df is not None:
            launches = df[df["new_gpu_launch"] == 1]
            non_launches = df[df["new_gpu_launch"] == 0]
            answer = (
                f"**GPU/CPU Launch Impact**\n\n"
                f"- Launch-week avg: {int(launches['units_sold'].mean()):,} units\n"
                f"- Non-launch avg: {int(non_launches['units_sold'].mean()):,} units\n"
                f"- Uplift: +{int(launches['units_sold'].mean() - non_launches['units_sold'].mean()):,} units "
                f"({(launches['units_sold'].mean() / non_launches['units_sold'].mean() - 1) * 100:.1f}%)\n"
                f"- Total launch weeks: {len(launches)}\n"
            )
            return {"answer": answer}
        return {"answer": "No data available."}

    # ── Route: weather / temperature ─────────────────────────────────
    if any(kw in query for kw in ("weather", "temperature", "temp", "hot", "cold")):
        if df is not None:
            last4 = df.tail(4)
            answer = (
                f"**Recent Weather at Hub (4 weeks)**\n\n"
                f"- Avg temperature: {last4['avg_temperature_f'].mean():.1f} °F\n"
                f"- Range: {last4['avg_temperature_f'].min():.1f} – {last4['avg_temperature_f'].max():.1f} °F\n"
            )
            return {"answer": answer}
        return {"answer": "No data available."}

    # ── Route: holiday ───────────────────────────────────────────────
    if any(kw in query for kw in ("holiday", "thanksgiving", "christmas", "black friday")):
        if df is not None:
            hol = df[df["is_holiday_week"] == 1]
            non_hol = df[df["is_holiday_week"] == 0]
            answer = (
                f"**Holiday Week Impact**\n\n"
                f"- Holiday-week avg: {int(hol['units_sold'].mean()):,} units\n"
                f"- Non-holiday avg: {int(non_hol['units_sold'].mean()):,} units\n"
                f"- Difference: {int(hol['units_sold'].mean() - non_hol['units_sold'].mean()):,} units\n"
                f"- Total holiday weeks: {len(hol)}\n"
            )
            return {"answer": answer}
        return {"answer": "No data available."}

    # ── Fallback ─────────────────────────────────────────────────────
    return {
        "answer": (
            "I can help with:\n\n"
            "- **Forecast**: Ask about upcoming demand\n"
            "- **Metrics**: Model accuracy & performance\n"
            "- **History**: Past sales trends\n"
            "- **Price**: Spot pricing analysis\n"
            "- **Supply**: Constraint & shortage info\n"
            "- **GPU Launches**: Platform launch impact\n"
            "- **Weather**: Temperature at hub\n"
            "- **Holidays**: Holiday week patterns\n"
        )
    }


# ── SPA ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def spa():
    if SPA_PATH.exists():
        return FileResponse(SPA_PATH, media_type="text/html")
    return HTMLResponse("<h1>RAM Demand Forecast</h1><p>SPA not found.</p>")

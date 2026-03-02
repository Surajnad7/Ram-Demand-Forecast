"""
train.py
========
End-to-end training pipeline for the weekly 16 GB RAM unit
demand model.

Steps
-----
1.  Load CSV  →  add cyclic week-of-year features
2.  Train / hold-out split  (2021-01-04 → 2025-03-31 train | 2025-04-07 → 2025-12-29 hold-out)
3.  Fit Prophet on training set
4.  Cross-validate  (initial = 104 weeks, period = 13 weeks, horizon = 3 weeks)
5.  Score on hold-out  →  MAPE, RMSE, MAE, R²
6.  Quality-gate check
7.  Persist model (.pkl) + metrics (.json)

Quality Gates
-------------
  MAPE  ≤ 10 %
  RMSE  ≤ 900 units
  MAE   ≤ 650 units
  R²    ≥ 0.85

Usage
-----
    python -m src.train                           # default paths
    python -m src.train --data data/custom.csv    # custom data
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import REGRESSORS, add_cyclic_week_of_year, build_future_features, configure_prophet

logger = logging.getLogger(__name__)

# ── Quality-gate thresholds ──────────────────────────────────────────
MAPE_THRESHOLD = 0.10
RMSE_THRESHOLD = 900
MAE_THRESHOLD = 650
R2_THRESHOLD = 0.85

HOLDOUT_START = "2025-04-07"


def train_and_evaluate(
    data_path: str | Path = "data/ram_demand.csv",
    model_dir: str | Path = "models",
) -> dict:
    """
    Full training pipeline.  Returns metrics dict including quality-gate result.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & prep ───────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["ds"])
    df = add_cyclic_week_of_year(df)
    df["y"] = df["units_sold"]
    logger.info("Loaded %d rows (%s → %s)", len(df), df["ds"].min().date(), df["ds"].max().date())

    # ── 2. Split ─────────────────────────────────────────────────────
    train = df[df["ds"] < HOLDOUT_START].copy()
    holdout = df[df["ds"] >= HOLDOUT_START].copy()
    logger.info("Train: %d weeks | Hold-out: %d weeks", len(train), len(holdout))

    # ── 3. Fit ───────────────────────────────────────────────────────
    m = configure_prophet()
    m.fit(train[["ds", "y"] + REGRESSORS])
    logger.info("Prophet fitted")

    # ── 4. Cross-validation ──────────────────────────────────────────
    try:
        cv_results = cross_validation(
            m,
            initial="728 days",   # ~104 weeks
            period="91 days",     # ~13 weeks
            horizon="21 days",    # 3 weeks
        )
        cv_metrics = performance_metrics(cv_results)
        cv_mape = float(cv_metrics["mape"].mean())
        logger.info("CV MAPE: %.4f", cv_mape)
    except Exception as exc:
        logger.warning("CV skipped: %s", exc)
        cv_mape = None

    # ── 5. Hold-out evaluation ───────────────────────────────────────
    future_ho = m.make_future_dataframe(periods=len(holdout), freq="W-MON")
    future_ho = build_future_features(df, future_ho)
    preds = m.predict(future_ho)

    ho_preds = preds[preds["ds"].isin(holdout["ds"])].copy()
    ho_preds = ho_preds.merge(holdout[["ds", "y"]], on="ds")

    y_true = ho_preds["y"].values
    y_pred = ho_preds["yhat"].values

    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # ── 6. Quality gate ──────────────────────────────────────────────
    passed = mape <= MAPE_THRESHOLD and rmse <= RMSE_THRESHOLD and mae <= MAE_THRESHOLD and r2 >= R2_THRESHOLD

    metrics = {
        "mape": round(mape, 4),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "cv_mape": round(cv_mape, 4) if cv_mape is not None else None,
        "quality_gate_passed": passed,
        "train_weeks": len(train),
        "holdout_weeks": len(holdout),
    }
    logger.info("Metrics: %s", metrics)

    # ── 7. Persist ───────────────────────────────────────────────────
    model_path = model_dir / "prophet_ram.pkl"
    joblib.dump(m, model_path)
    logger.info("Model saved → %s", model_path)

    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved → %s", metrics_path)

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ram_demand.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    result = train_and_evaluate(data_path=args.data, model_dir=args.model_dir)
    gate = "PASSED" if result["quality_gate_passed"] else "FAILED"
    print(f"\n{'='*50}")
    print(f"Quality Gate: {gate}")
    print(f"MAPE={result['mape']:.2%}  RMSE={result['rmse']:.0f}  MAE={result['mae']:.0f}  R²={result['r2']:.4f}")
    print(f"{'='*50}")

"""
predict.py
==========
Load the persisted Prophet model and produce an N-week forecast for
weekly 16 GB RAM unit demand.

Default horizon : 3 weeks
Maximum horizon : 12 weeks

Returns a list of dicts with ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_future_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/prophet_ram.pkl")
DATA_PATH = Path("data/ram_demand.csv")
DEFAULT_HORIZON = 3
MAX_HORIZON = 12


def forecast(
    horizon: int = DEFAULT_HORIZON,
    model_path: str | Path = MODEL_PATH,
    data_path: str | Path = DATA_PATH,
) -> list[dict]:
    """
    Produce a weekly forecast.

    Parameters
    ----------
    horizon : int
        Number of weeks to forecast (1-12, default 3).
    model_path : Path
        Pickled Prophet model.
    data_path : Path
        Historical CSV (needed for feature imputation).

    Returns
    -------
    list[dict]
        Each dict has keys ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    """
    horizon = max(1, min(horizon, MAX_HORIZON))

    m = joblib.load(model_path)
    df = pd.read_csv(data_path, parse_dates=["ds"])

    future = m.make_future_dataframe(periods=horizon, freq="W-MON")
    future = build_future_features(df, future)
    preds = m.predict(future)

    out = preds.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["ds"] = out["ds"].dt.strftime("%Y-%m-%d")
    out["yhat"] = out["yhat"].round(0).astype(int)
    out["yhat_lower"] = out["yhat_lower"].round(0).astype(int)
    out["yhat_upper"] = out["yhat_upper"].round(0).astype(int)

    return out.to_dict(orient="records")


if __name__ == "__main__":
    import argparse, json

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()
    results = forecast(horizon=args.horizon)
    print(json.dumps(results, indent=2))

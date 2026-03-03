"""
test_pipeline.py
=================
Comprehensive test suite for the weekly 16 GB RAM unit demand
forecasting pipeline.

44 tests across 5 categories:
  • DataGeneration (14): CSV shape, columns, value ranges, date continuity
  • Features (5): Cyclic encoding, Prophet config, future-feature builder
  • Training (5): Model persistence, metric keys, quality-gate structure
  • Prediction (5): Horizon bounds, output schema, clamping
  • API (15): Every endpoint, status codes, response schemas
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

class TestDataGeneration:
    """Validate the synthetic dataset generator."""

    @pytest.fixture(autouse=True)
    def _gen(self, tmp_path):
        from src.generate_dataset import generate
        self.out = tmp_path / "ram_demand.csv"
        self.df = generate(out_path=self.out)

    def test_csv_exists(self):
        assert self.out.exists()

    def test_row_count(self):
        assert 250 <= len(self.df) <= 270  # ~260 weeks in 5 years

    def test_required_columns(self):
        expected = {
            "ds", "units_sold", "avg_spot_price_usd", "avg_temperature_f",
            "is_holiday_week", "is_back_to_school", "is_black_friday_week",
            "is_prime_day_week", "pct_supply_constraint", "new_gpu_launch",
            "week_of_year_sin", "week_of_year_cos",
        }
        assert expected.issubset(set(self.df.columns))

    def test_ds_is_datetime(self):
        assert pd.api.types.is_datetime64_any_dtype(self.df["ds"])

    def test_ds_monotonic(self):
        assert self.df["ds"].is_monotonic_increasing

    def test_weekly_frequency(self):
        diffs = self.df["ds"].diff().dropna().dt.days
        assert (diffs == 7).all()

    def test_all_mondays(self):
        assert (self.df["ds"].dt.dayofweek == 0).all()

    def test_units_sold_positive(self):
        assert (self.df["units_sold"] > 0).all()

    def test_units_sold_range(self):
        assert self.df["units_sold"].min() >= 500
        assert self.df["units_sold"].max() <= 25000

    def test_spot_price_range(self):
        assert self.df["avg_spot_price_usd"].min() >= 35
        assert self.df["avg_spot_price_usd"].max() <= 95

    def test_temperature_range(self):
        assert self.df["avg_temperature_f"].min() >= 5
        assert self.df["avg_temperature_f"].max() <= 105

    def test_binary_columns(self):
        for col in ("is_holiday_week", "is_back_to_school",
                     "is_black_friday_week", "is_prime_day_week", "new_gpu_launch"):
            assert set(self.df[col].unique()).issubset({0, 1})

    def test_supply_constraint_range(self):
        assert self.df["pct_supply_constraint"].min() >= 0
        assert self.df["pct_supply_constraint"].max() <= 1

    def test_cyclic_encoding_range(self):
        assert self.df["week_of_year_sin"].between(-1, 1).all()
        assert self.df["week_of_year_cos"].between(-1, 1).all()


# ═══════════════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════════════

class TestFeatures:
    """Validate feature engineering utilities."""

    def test_regressor_list(self):
        from src.features import REGRESSORS
        assert len(REGRESSORS) == 10
        assert "avg_spot_price_usd" in REGRESSORS
        assert "week_of_year_sin" in REGRESSORS

    def test_add_cyclic_week(self):
        from src.features import add_cyclic_week_of_year
        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=52, freq="W-MON")})
        df = add_cyclic_week_of_year(df)
        assert "week_of_year_sin" in df.columns
        assert "week_of_year_cos" in df.columns
        assert df["week_of_year_sin"].between(-1, 1).all()

    def test_configure_prophet_returns_model(self):
        from src.features import configure_prophet
        m = configure_prophet()
        assert hasattr(m, "fit")
        assert hasattr(m, "predict")

    def test_prophet_has_regressors(self):
        from src.features import configure_prophet, REGRESSORS
        m = configure_prophet()
        # Prophet stores regressors as dict keyed by name
        extra = list(m.extra_regressors.keys()) if hasattr(m, "extra_regressors") else []
        for reg in REGRESSORS:
            assert reg in extra, f"Regressor {reg} not found in Prophet model"
        assert m.yearly_seasonality == 6
        assert m.weekly_seasonality is False

    def test_build_future_features(self):
        from src.features import build_future_features, REGRESSORS, add_cyclic_week_of_year
        from src.generate_dataset import generate
        import tempfile, os
        tmp = os.path.join(tempfile.mkdtemp(), "test.csv")
        df = generate(out_path=tmp)
        from prophet import Prophet
        m = Prophet()
        m.fit(df.rename(columns={"units_sold": "y"})[["ds", "y"]])
        future = m.make_future_dataframe(periods=3, freq="W-MON")
        result = build_future_features(df, future)
        for reg in REGRESSORS:
            assert reg in result.columns
        assert result[REGRESSORS].isna().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════

class TestTraining:
    """Validate training pipeline outputs."""

    @pytest.fixture(autouse=True)
    def _train(self, tmp_path):
        from src.generate_dataset import generate
        from src.train import train_and_evaluate
        data_path = tmp_path / "ram_demand.csv"
        generate(out_path=data_path)
        self.model_dir = tmp_path / "models"
        self.metrics = train_and_evaluate(data_path=data_path, model_dir=self.model_dir)

    def test_model_file_exists(self):
        assert (self.model_dir / "prophet_ram.pkl").exists()

    def test_metrics_file_exists(self):
        assert (self.model_dir / "metrics.json").exists()

    def test_metrics_keys(self):
        expected = {"mape", "rmse", "mae", "r2", "quality_gate_passed", "train_weeks", "holdout_weeks"}
        assert expected.issubset(set(self.metrics.keys()))

    def test_metrics_types(self):
        assert isinstance(self.metrics["mape"], float)
        assert isinstance(self.metrics["rmse"], float)
        assert isinstance(self.metrics["r2"], float)
        assert isinstance(self.metrics["quality_gate_passed"], bool)

    def test_quality_gate_structure(self):
        m = self.metrics
        expected_pass = (
            m["mape"] <= 0.10 and
            m["rmse"] <= 900 and
            m["mae"] <= 650 and
            m["r2"] >= 0.85
        )
        assert m["quality_gate_passed"] == expected_pass


# ═══════════════════════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════════════════════

class TestPrediction:
    """Validate inference module."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from src.generate_dataset import generate
        from src.train import train_and_evaluate
        self.data_path = tmp_path / "ram_demand.csv"
        generate(out_path=self.data_path)
        self.model_dir = tmp_path / "models"
        train_and_evaluate(data_path=self.data_path, model_dir=self.model_dir)
        self.model_path = self.model_dir / "prophet_ram.pkl"

    def test_default_horizon(self):
        from src.predict import forecast
        results = forecast(model_path=self.model_path, data_path=self.data_path)
        assert len(results) == 3

    def test_custom_horizon(self):
        from src.predict import forecast
        results = forecast(horizon=6, model_path=self.model_path, data_path=self.data_path)
        assert len(results) == 6

    def test_output_keys(self):
        from src.predict import forecast
        results = forecast(model_path=self.model_path, data_path=self.data_path)
        for r in results:
            assert "ds" in r
            assert "yhat" in r
            assert "yhat_lower" in r
            assert "yhat_upper" in r

    def test_horizon_clamping_low(self):
        from src.predict import forecast
        results = forecast(horizon=-5, model_path=self.model_path, data_path=self.data_path)
        assert len(results) == 1

    def test_horizon_clamping_high(self):
        from src.predict import forecast
        results = forecast(horizon=999, model_path=self.model_path, data_path=self.data_path)
        assert len(results) == 12


# ═══════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════

class TestAPI:
    """Validate FastAPI endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        from src.generate_dataset import generate
        from src.train import train_and_evaluate
        import src.app as app_mod

        data_path = tmp_path / "ram_demand.csv"
        generate(out_path=data_path)
        model_dir = tmp_path / "models"
        train_and_evaluate(data_path=data_path, model_dir=model_dir)

        # Monkey-patch paths
        app_mod.DATA_PATH = data_path
        app_mod.MODEL_PATH = model_dir / "prophet_ram.pkl"
        app_mod.METRICS_PATH = model_dir / "metrics.json"
        app_mod.SPA_PATH = Path("static/index.html")

        from httpx import AsyncClient, ASGITransport
        from src.app import app
        self.app = app
        self.transport = ASGITransport(app=app)

    @pytest.fixture
    def client(self):
        from httpx import AsyncClient, ASGITransport
        return AsyncClient(transport=self.transport, base_url="http://test")

    @pytest.mark.asyncio
    async def test_health(self, client):
        async with client as c:
            r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_predict_default(self, client):
        async with client as c:
            r = await c.post("/predict", json={"horizon": 3})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert len(d["forecasts"]) == 3

    @pytest.mark.asyncio
    async def test_predict_custom(self, client):
        async with client as c:
            r = await c.post("/predict", json={"horizon": 6})
        assert r.status_code == 200
        assert len(r.json()["forecasts"]) == 6

    @pytest.mark.asyncio
    async def test_predict_invalid(self, client):
        async with client as c:
            r = await c.post("/predict", json={"horizon": 0})
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_history_default(self, client):
        async with client as c:
            r = await c.get("/api/history")
        assert r.status_code == 200
        d = r.json()
        assert "data" in d
        assert len(d["data"]) <= 52

    @pytest.mark.asyncio
    async def test_history_custom(self, client):
        async with client as c:
            r = await c.get("/api/history?limit=10")
        d = r.json()
        assert len(d["data"]) == 10

    @pytest.mark.asyncio
    async def test_history_data_keys(self, client):
        async with client as c:
            r = await c.get("/api/history?limit=1")
        row = r.json()["data"][0]
        assert "ds" in row
        assert "units_sold" in row
        assert "avg_spot_price_usd" in row

    @pytest.mark.asyncio
    async def test_metrics(self, client):
        async with client as c:
            r = await c.get("/api/metrics")
        assert r.status_code == 200
        d = r.json()
        assert "mape" in d
        assert "r2" in d

    @pytest.mark.asyncio
    async def test_summary(self, client):
        async with client as c:
            r = await c.get("/api/summary")
        assert r.status_code == 200
        d = r.json()
        assert "total_weeks" in d
        assert "recent_trend" in d

    @pytest.mark.asyncio
    async def test_summary_metrics(self, client):
        async with client as c:
            r = await c.get("/api/summary")
        d = r.json()
        assert "metrics" in d
        assert "mape" in d["metrics"]

    @pytest.mark.asyncio
    async def test_assistant_forecast(self, client):
        async with client as c:
            r = await c.get("/api/assistant?q=forecast")
        assert r.status_code == 200
        assert "answer" in r.json()

    @pytest.mark.asyncio
    async def test_assistant_metrics(self, client):
        async with client as c:
            r = await c.get("/api/assistant?q=accuracy")
        assert r.status_code == 200
        assert "answer" in r.json()

    @pytest.mark.asyncio
    async def test_assistant_history(self, client):
        async with client as c:
            r = await c.get("/api/assistant?q=historical sales")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_assistant_fallback(self, client):
        async with client as c:
            r = await c.get("/api/assistant?q=hello")
        assert r.status_code == 200
        assert "I can help" in r.json()["answer"]

    @pytest.mark.asyncio
    async def test_spa(self, client):
        async with client as c:
            r = await c.get("/")
        assert r.status_code == 200

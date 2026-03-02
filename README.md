# 💾 16 GB RAM Unit Demand Forecast

Weekly 16 GB RAM module demand forecast powered by **Facebook Prophet**, served via **FastAPI** with a full IG-style single-page application frontend. Deployed on **GCP Cloud Run** with automated Monday-morning retraining.

---

## Overview

| Attribute | Detail |
|---|---|
| **Product** | 16 GB DDR4/DDR5 RAM modules |
| **Granularity** | Weekly (ISO week, Monday start) |
| **Forecast horizon** | 3 weeks (default), up to 12 weeks |
| **Prediction cadence** | Every Monday morning 06:00 UTC |
| **Model** | Facebook Prophet (Bayesian additive regression) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS SPA with Chart.js — IG-style dark UI |
| **Deployment** | Docker → GCP Cloud Run |
| **Dataset span** | 2021-01-04 → 2025-12-29 (260 weeks / 5 years) |
| **Target variable** | `units_sold` — total 16 GB RAM modules sold per week |

---

## Purpose

This system provides **production-grade weekly demand forecasting** for 16 GB RAM units to support:

- **Supply chain planning** — 3-week-out visibility for procurement and warehousing
- **Inventory optimization** — prevent stockouts during GPU launch spikes and back-to-school season
- **Revenue forecasting** — weekly unit projections with 95% confidence intervals
- **Strategic planning** — quantified impact of holidays, promotions, supply constraints, and platform launches

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│   SPA Client     │────▶│  FastAPI Service  │────▶│   Prophet    │
│   (Chart.js)     │◀────│  /predict /train  │◀────│   Model      │
│   6 pages        │     │  /api/*           │     └──────────────┘
└──────────────────┘     └──────────────────┘
                                  │
                         ┌────────┴────────┐
                         │   CSV Dataset   │
                         │   260 weekly    │
                         │   observations  │
                         └─────────────────┘
```

---

## Dataset

| Column | Description | Range |
|---|---|---|
| `ds` | Week-starting date (Monday) | 2021-01-04 → 2025-12-29 |
| `units_sold` | Total 16 GB RAM modules sold (target) | ~500 – 18,000 |
| `avg_spot_price_usd` | Weekly average spot price per module ($) | 35 – 95 |
| `avg_temperature_f` | Weekly average temperature at distribution hub (°F) | 5 – 105 |
| `is_holiday_week` | Week contains a major US holiday | 0 / 1 |
| `is_back_to_school` | Aug–Sep back-to-school season | 0 / 1 |
| `is_black_friday_week` | Black Friday / Cyber Monday week | 0 / 1 |
| `is_prime_day_week` | Amazon-style mid-year sale week | 0 / 1 |
| `pct_supply_constraint` | Supply shortfall ratio (chip shortage impact) | 0.0 – 0.35 |
| `new_gpu_launch` | Major GPU/CPU platform launched that week | 0 / 1 |
| `week_of_year_sin` | Cyclical week-of-year encoding (sin component) | -1 to 1 |
| `week_of_year_cos` | Cyclical week-of-year encoding (cos component) | -1 to 1 |

### Engineered Features

| Feature | Type | Engineering Method |
|---|---|---|
| **Cyclic week encoding** | Continuous | `sin(2π × week / 52)` and `cos(2π × week / 52)` |
| **Holiday windows** | Binary | Mapped from US federal holiday calendar with ±1 week window |
| **Back-to-school flag** | Binary | Calendar-based (August–September) |
| **Black Friday flag** | Binary | Nov 23–30 week detection |
| **Prime Day flag** | Binary | July 8–14 week detection |
| **Supply constraint** | Continuous | Historical chip shortage data (2021-2022 era modeled) |
| **GPU launch indicator** | Binary | Mapped from historical & projected GPU/CPU release calendar |
| **Price feature** | Continuous | Spot pricing with Moore's Law trend + annual cycles |
| **Temperature** | Continuous | Sinusoidal seasonal model with year-over-year warming |

### Seasonality Patterns

- **Back-to-school** (Aug–Sep) drives +18% demand uplift
- **Black Friday / Cyber Monday** week drives +35% demand spike
- **GPU/CPU launches** create +20–40% demand surges (system builders)
- **Holiday weeks** (Christmas, July 4th) reduce shipments 22–45%
- **Supply constraints** (2021-2022 chip shortage) dampened fulfilled orders by 10–35%
- **Price elasticity**: ~45 fewer units sold per $1 above average spot price
- **Long-term growth**: ~3.5 units/week trend (cloud & AI demand)
- **Hot weather** (>85°F) correlates with +6% demand (summer PC building)

---

## Model Configuration

```python
Prophet(
    growth="linear",
    yearly_seasonality=6,         # Fourier order for annual pattern
    weekly_seasonality=False,     # data is weekly grain — no intra-week
    daily_seasonality=False,
    changepoint_prior_scale=0.06, # trend flexibility
    seasonality_prior_scale=8.0,  # seasonality regularization
    holidays_prior_scale=12.0,    # holiday effect strength
    interval_width=0.95,          # 95% prediction interval
)
```

### Regressors (all additive mode)

All 10 exogenous features added via `model.add_regressor(name, mode="additive")`:

1. `avg_spot_price_usd` — price elasticity signal
2. `avg_temperature_f` — weather-driven demand patterns
3. `is_holiday_week` — shipping/distribution slowdowns
4. `is_back_to_school` — seasonal academic demand surge
5. `is_black_friday_week` — promotional demand spike
6. `is_prime_day_week` — mid-year sale event impact
7. `pct_supply_constraint` — supply-side dampening factor
8. `new_gpu_launch` — platform launch demand multiplier
9. `week_of_year_sin` — cyclical annual position (sine)
10. `week_of_year_cos` — cyclical annual position (cosine)

---

## Training Pipeline

### Split Strategy

| Set | Period | Weeks | Purpose |
|---|---|---|---|
| **Training** | 2021-01-04 → 2025-03-31 | ~222 | Model fitting |
| **Hold-out** | 2025-04-07 → 2025-12-29 | ~38 | Out-of-sample evaluation |

### Cross-Validation

| Parameter | Value | Rationale |
|---|---|---|
| `initial` | 728 days (~104 weeks) | 2 full years of seasonality captured |
| `period` | 91 days (~13 weeks) | Quarterly rolling window |
| `horizon` | 21 days (~3 weeks) | Matches production forecast horizon |

### Backtesting Order

1. **Fit** Prophet on training set (222 weeks)
2. **Cross-validate** with rolling origin (initial=104w, period=13w, horizon=3w)
3. **Score** on held-out 38 weeks (2025 Q2–Q4)
4. **Quality gate** pass/fail determination
5. **Persist** model artifact (.pkl) + metrics (.json)

---

## Quality Gates — Accuracy Targets

| Metric | Threshold | Description |
|---|---|---|
| **MAPE** | ≤ 10% | Mean Absolute Percentage Error |
| **RMSE** | ≤ 900 units | Root Mean Square Error |
| **MAE** | ≤ 650 units | Mean Absolute Error |
| **R²** | ≥ 0.85 | Coefficient of Determination |

### Metric Definitions

- **MAPE**: `mean(|actual - predicted| / actual)` — scale-independent accuracy measure; ≤10% = production-grade for weekly demand
- **RMSE**: `√(mean((actual - predicted)²))` — penalizes large errors; ≤900 units on ~8,000–15,000 weekly base
- **MAE**: `mean(|actual - predicted|)` — average absolute miss in units
- **R²**: `1 - SS_res / SS_tot` — proportion of variance explained; ≥0.85 indicates strong fit
- **CV MAPE**: Cross-validated MAPE across rolling origins — guards against overfitting

### Quality Gate Logic

```python
passed = (
    mape <= 0.10 and
    rmse <= 900 and
    mae  <= 650 and
    r2   >= 0.85
)
```

The model is only deployed / persisted if **all four gates pass simultaneously**.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness / readiness check |
| `POST` | `/predict` | N-week forecast (1–12, default 3) |
| `POST` | `/train` | Retrain the model |
| `GET` | `/api/history` | Historical demand data |
| `GET` | `/api/metrics` | Training metrics |
| `GET` | `/api/summary` | Dashboard summary payload |
| `GET` | `/api/assistant` | Natural-language Q&A |
| `GET` | `/` | SPA frontend |

### Predict Request

```json
POST /predict
{
  "horizon": 3
}
```

### Predict Response

```json
{
  "status": "ok",
  "horizon_weeks": 3,
  "forecasts": [
    {"ds": "2026-01-05", "yhat": 11240, "yhat_lower": 9850, "yhat_upper": 12630},
    {"ds": "2026-01-12", "yhat": 11380, "yhat_lower": 9920, "yhat_upper": 12840},
    {"ds": "2026-01-19", "yhat": 11150, "yhat_lower": 9710, "yhat_upper": 12590}
  ]
}
```

### Training Response

```json
{
  "status": "ok",
  "metrics": {
    "mape": 0.0623,
    "rmse": 712.45,
    "mae": 548.32,
    "r2": 0.9124,
    "cv_mape": 0.0701,
    "quality_gate_passed": true,
    "train_weeks": 222,
    "holdout_weeks": 38
  }
}
```

---

## Frontend — IG-Style SPA

6-page single-page application with instant 0ms route transitions:

| Page | Features |
|---|---|
| **Dashboard** | KPI cards, historical + forecast overlay chart, auto-loads on visit |
| **Forecast Explorer** | Adjustable horizon slider (1–12 weeks), bar chart, sortable results table |
| **Model Performance** | MAPE/RMSE/MAE/R²/CV MAPE cards, quality gate badge, 52-week trend chart |
| **Data Explorer** | Adjustable range slider (4–260 weeks), full-featured data table with all features |
| **Training** | One-click retrain button, live status indicators, post-train metric cards |
| **Ask Assistant** | Natural-language Q&A chat interface with Markdown rendering |

### UI Specifications

- **Navigation**: Instant hash-based routing with 0ms page transitions — no toasts, no loading spinners for navigation
- **Theme**: Dark mode default with one-click light mode toggle, persisted to localStorage
- **Typography**: SF Pro / system font stack, -apple-system,BlinkMacSystemFont
- **Color palette**: Indigo accent (#6366f1), green for positive (#22c55e), red for alerts (#ef4444)
- **Cards**: Glassmorphic surface with gradient top-border hover effect
- **Charts**: Chart.js 4.x with smooth 350ms easeOutQuart animations
- **Tables**: Sticky headers, alternating hover highlight, scrollable container
- **Responsive**: Sidebar collapses to 64px icon-only mode at ≤768px
- **Skeletons**: Shimmer loading placeholders on initial data fetch
- **Accessibility**: Semantic HTML, keyboard-navigable, high-contrast text

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data
python -m src.generate_dataset

# 3. Train the model
python -m src.train

# 4. Start the server
uvicorn src.app:app --host 0.0.0.0 --port 8080

# 5. Open browser
open http://localhost:8080
```

---

## Docker

```bash
docker build -t ram-demand-forecast .
docker run -p 8080:8080 ram-demand-forecast
```

---

## GCP Cloud Run

```bash
chmod +x deploy.sh
./deploy.sh YOUR_PROJECT_ID us-central1
```

Deploys with **weekly retrain via Cloud Scheduler** at 06:00 UTC every Monday (`0 6 * * 1`).

### Cloud Run Specifications

| Setting | Value |
|---|---|
| Memory | 1 Gi |
| CPU | 1 vCPU |
| Timeout | 300s |
| Port | 8080 |
| Min instances | 0 (scale to zero) |
| Max instances | 10 |
| Scheduler | `0 6 * * 1` — every Monday 06:00 UTC |

---

## Testing

```bash
pytest tests/ -v
```

44 tests across 5 categories:

| Category | Count | Scope |
|---|---|---|
| **DataGeneration** | 14 | CSV shape, columns, value ranges, weekly frequency, Monday check, date continuity |
| **Features** | 5 | Cyclic encoding, Prophet config, regressor list, future-feature builder |
| **Training** | 5 | Model persistence, metric keys/types, quality-gate logic verification |
| **Prediction** | 5 | Horizon bounds, output schema, clamping (min=1, max=12) |
| **API** | 15 | Every endpoint, status codes, response schemas, error handling |

---

## Project Structure

```
ram-demand-forecast/
├── data/
│   └── ram_demand.csv            # 260 weekly rows (generated)
├── models/
│   ├── prophet_ram.pkl           # trained model (git-ignored)
│   └── metrics.json              # latest training metrics (git-ignored)
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py       # synthetic data generator (10 features)
│   ├── features.py               # regressors + Prophet config
│   ├── train.py                  # training pipeline + quality gate
│   ├── predict.py                # inference module
│   └── app.py                    # FastAPI service (8 endpoints)
├── static/
│   └── index.html                # 6-page IG-style SPA frontend
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # 44 tests
├── Dockerfile
├── deploy.sh                     # GCP Cloud Run deployment script
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Technical Specifications Summary

| Specification | Detail |
|---|---|
| **Language** | Python 3.10+ |
| **ML Framework** | Facebook Prophet 1.1.5+ |
| **API Framework** | FastAPI 0.110+ |
| **Server** | Uvicorn (ASGI) |
| **Data** | pandas 2.1+, numpy 1.26+ |
| **Metrics** | scikit-learn 1.4+ |
| **Serialization** | joblib 1.3+ |
| **Testing** | pytest 8.0+, httpx 0.27+ |
| **Frontend** | Vanilla HTML/CSS/JS, Chart.js 4.x, marked.js |
| **Container** | Python 3.11-slim Docker image |
| **Cloud** | GCP Cloud Run + Cloud Scheduler |
| **Forecast freq** | Weekly (W-MON) |
| **Default horizon** | 3 weeks |
| **Max horizon** | 12 weeks |
| **Prediction cadence** | Every Monday 06:00 UTC (automated) |
| **Confidence interval** | 95% (interval_width=0.95) |
| **Changepoint sensitivity** | 0.06 (moderate flexibility) |
| **Seasonality** | Yearly (Fourier order 6) |
| **Feature count** | 10 exogenous regressors |
| **Training data** | 222 weeks |
| **Hold-out data** | 38 weeks |
| **CV strategy** | Rolling origin: initial=104w, period=13w, horizon=3w |
| **MAPE target** | ≤ 10% |
| **RMSE target** | ≤ 900 units |
| **MAE target** | ≤ 650 units |
| **R² target** | ≥ 0.85 |

---

## License

MIT
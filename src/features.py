"""
features.py
============
Feature engineering & Prophet configuration for the weekly
16 GB RAM unit demand model.

Regressors (all additive)
-------------------------
1.  avg_spot_price_usd     — weekly average spot price per module ($)
2.  avg_temperature_f      — weekly average temperature at hub (°F)
3.  is_holiday_week        — binary: week contains US public holiday
4.  is_back_to_school      — binary: Aug–Sep back-to-school season
5.  is_black_friday_week   — binary: Black Friday / Cyber Monday week
6.  is_prime_day_week      — binary: mid-year sale event week
7.  pct_supply_constraint  — 0-1 supply shortfall ratio
8.  new_gpu_launch         — binary: major GPU / CPU platform launch week
9.  week_of_year_sin       — cyclical week-of-year encoding (sin)
10. week_of_year_cos       — cyclical week-of-year encoding (cos)

Holidays
--------
US holidays affecting electronics distribution & shipping:
  - New Year's, MLK Day, Presidents' Day, Memorial Day, Juneteenth,
    Independence Day, Labor Day, Thanksgiving, Christmas

Prophet Hyper-parameters
------------------------
  growth              = "linear"
  yearly_seasonality  = True (Fourier order 6)
  weekly_seasonality  = False (data is weekly grain)
  daily_seasonality   = False
  changepoint_prior   = 0.06
  seasonality_prior   = 8.0
  holidays_prior      = 12.0
  interval_width      = 0.95
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from prophet import Prophet

# ── Regressor list ───────────────────────────────────────────────────
REGRESSORS: list[str] = [
    "avg_spot_price_usd",
    "avg_temperature_f",
    "is_holiday_week",
    "is_back_to_school",
    "is_black_friday_week",
    "is_prime_day_week",
    "pct_supply_constraint",
    "new_gpu_launch",
    "week_of_year_sin",
    "week_of_year_cos",
]

# ── US holidays relevant to electronics distribution ─────────────────
_RAW_HOLIDAYS: dict[str, list[str]] = {
    "New Year's Day":    ["01-01"],
    "MLK Day":           ["01-15", "01-16", "01-17", "01-20", "01-21"],
    "Presidents' Day":   ["02-17", "02-18", "02-19", "02-20", "02-15"],
    "Memorial Day":      ["05-25", "05-26", "05-27", "05-28", "05-29", "05-30"],
    "Juneteenth":        ["06-19"],
    "Independence Day":  ["07-04"],
    "Labor Day":         ["09-01", "09-02", "09-03", "09-04", "09-05", "09-06", "09-07"],
    "Columbus Day":      ["10-08", "10-09", "10-10", "10-12", "10-13", "10-14"],
    "Veterans Day":      ["11-11"],
    "Thanksgiving":      ["11-22", "11-23", "11-24", "11-25", "11-26", "11-27", "11-28"],
    "Black Friday":      ["11-23", "11-24", "11-25", "11-26", "11-27", "11-28", "11-29"],
    "Christmas":         ["12-25"],
}


def _expand_holidays(start_year: int = 2020, end_year: int = 2027) -> pd.DataFrame:
    """Build a Prophet-compatible holidays DataFrame."""
    rows: list[dict] = []
    for name, md_list in _RAW_HOLIDAYS.items():
        for year in range(start_year, end_year + 1):
            for md in md_list:
                try:
                    ds = pd.Timestamp(f"{year}-{md}")
                    rows.append({"holiday": name, "ds": ds, "lower_window": 0, "upper_window": 1})
                except ValueError:
                    continue
    return pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"])


def add_cyclic_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin / cos week-of-year encoding columns."""
    woy = pd.to_datetime(df["ds"]).dt.isocalendar().week.astype(int)
    df["week_of_year_sin"] = np.sin(2 * math.pi * woy / 52).round(6)
    df["week_of_year_cos"] = np.cos(2 * math.pi * woy / 52).round(6)
    return df


def configure_prophet() -> "Prophet":
    """
    Return a fully-configured Prophet instance for weekly RAM demand.
    Must call prophet.fit(df) separately.
    """
    from prophet import Prophet

    holidays_df = _expand_holidays()

    m = Prophet(
        growth="linear",
        yearly_seasonality=6,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.06,
        seasonality_prior_scale=8.0,
        holidays_prior_scale=12.0,
        holidays=holidays_df,
        interval_width=0.95,
    )

    for reg in REGRESSORS:
        m.add_regressor(reg, mode="additive")

    return m


def build_future_features(
    df: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """
    Populate regressor columns in ``future`` DataFrame.

    Known dates → use historical values.
    Future dates → impute with trailing-8-week medians & pattern rules.
    """
    future = future.copy()
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    future["ds"] = pd.to_datetime(future["ds"])

    merged = future.merge(
        df.drop(columns=["units_sold"], errors="ignore"), on="ds", how="left",
    )

    # Trailing 8-week stats for imputation
    tail = df.tail(8)
    defaults: dict[str, float] = {}
    for col in REGRESSORS:
        if col in ("week_of_year_sin", "week_of_year_cos"):
            continue
        if col in ("is_holiday_week", "is_back_to_school", "is_black_friday_week",
                    "is_prime_day_week", "new_gpu_launch"):
            defaults[col] = 0
        elif col in tail.columns:
            defaults[col] = float(tail[col].median())
        else:
            defaults[col] = 0.0

    _HOLIDAYS_MD = {(1, 1, 7), (1, 15, 21), (2, 14, 20), (5, 25, 31),
                    (6, 19, 19), (7, 1, 7), (9, 1, 7), (11, 22, 28), (12, 22, 31)}

    for idx in merged.index:
        if pd.isna(merged.loc[idx, "avg_spot_price_usd"]):
            d = merged.loc[idx, "ds"]

            # Spot price: trailing median
            merged.loc[idx, "avg_spot_price_usd"] = defaults.get("avg_spot_price_usd", 55.0)

            # Temperature: seasonal sinusoidal estimate
            phase = 2 * math.pi * (d.day_of_year - 30) / 365
            merged.loc[idx, "avg_temperature_f"] = round(55.0 + 25.0 * math.sin(phase), 1)

            # Holiday week
            is_hol = 0
            for m_start, d_start, d_end in _HOLIDAYS_MD:
                if d.month == m_start and d_start <= d.day <= d_end:
                    is_hol = 1
                    break
            merged.loc[idx, "is_holiday_week"] = is_hol

            # Back to school
            merged.loc[idx, "is_back_to_school"] = 1 if d.month in (8, 9) else 0

            # Black Friday
            merged.loc[idx, "is_black_friday_week"] = 1 if d.month == 11 and 23 <= d.day <= 30 else 0

            # Prime Day
            merged.loc[idx, "is_prime_day_week"] = 1 if d.month == 7 and 8 <= d.day <= 14 else 0

            # Supply constraint: assume normalized
            merged.loc[idx, "pct_supply_constraint"] = defaults.get("pct_supply_constraint", 0.02)

            # GPU launch: no future launches assumed by default
            merged.loc[idx, "new_gpu_launch"] = 0

    # Fill remaining NaNs
    for col in REGRESSORS:
        if col in ("week_of_year_sin", "week_of_year_cos"):
            continue
        merged[col] = merged[col].fillna(defaults.get(col, 0))

    merged = add_cyclic_week_of_year(merged)
    return merged

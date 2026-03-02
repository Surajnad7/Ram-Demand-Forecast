"""
generate_dataset.py
====================
Synthesise a realistic *weekly* 16 GB RAM unit demand dataset.

Grain  : one row per ISO week (Monday start)
Span   : 2021-01-04 → 2025-12-29  (260 weeks / 5 years)
Target : units_sold — total 16 GB RAM modules sold per week

Regressors (exogenous features)
-------------------------------
1.  avg_spot_price_usd     — weekly average spot price per 16 GB module ($)
2.  avg_temperature_f      — weekly average temperature at distribution hub (°F)
3.  is_holiday_week        — 1 if the week contains a major US holiday
4.  is_back_to_school      — 1 if Aug–Sep back-to-school season
5.  is_black_friday_week   — 1 if Black Friday / Cyber Monday week
6.  is_prime_day_week      — 1 if Amazon-style mid-year sale week
7.  pct_supply_constraint  — % supply shortfall vs normal (0-1 scale)
8.  new_gpu_launch         — 1 if a major GPU/CPU platform launched that week
9.  week_of_year_sin       — cyclical week-of-year encoding (sin)
10. week_of_year_cos       — cyclical week-of-year encoding (cos)

Seasonality patterns
--------------------
• Back-to-school (Aug–Sep) drives +18 % demand
• Black Friday / Cyber Monday week drives +35 % demand
• Holiday weeks (Christmas, July 4th, Thanksgiving) reduce shipments
• New GPU / CPU platform launches create demand spikes (+20-40 %)
• Supply constraints dampen fulfilled orders
• Slight long-term growth trend (cloud / AI demand)
• Price elasticity: higher spot prices → fewer consumer upgrades

Usage
-----
    python -m src.generate_dataset          # writes data/ram_demand.csv
    python -m src.generate_dataset --rows   # prints row count to stdout
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = "2021-01-04"  # first Monday of 2021
END = "2025-12-29"    # last Monday of 2025
OUT_PATH = Path("data/ram_demand.csv")

# ── US holiday weeks (month, day ranges that mark the week) ──────────
_HOLIDAY_WEEKS: list[tuple[int, int, int]] = [
    (1, 1, 7),      # New Year's week
    (1, 15, 21),     # MLK Day week
    (2, 14, 20),     # Presidents' Day week
    (5, 25, 31),     # Memorial Day week
    (6, 19, 19),     # Juneteenth
    (7, 1, 7),       # Independence Day week
    (9, 1, 7),       # Labor Day week
    (11, 22, 28),    # Thanksgiving week
    (12, 22, 31),    # Christmas / year-end week
]

# ── GPU / CPU launch weeks (approximate historical + projected) ──────
_GPU_LAUNCH_WEEKS: set[tuple[int, int]] = {
    (2021, 10), (2021, 47),   # RTX 3000 restocks
    (2022, 3), (2022, 38),    # RTX 4090 / AM5
    (2023, 2), (2023, 36),    # RTX 4070 series / Ryzen 7000X3D
    (2024, 3), (2024, 25), (2024, 39),  # RTX 5000 / next-gen
    (2025, 4), (2025, 26), (2025, 40),  # projected launches
}


def _seasonal_temperature(day_of_year: int, year: int) -> float:
    """Sinusoidal US-average temperature (°F) — warehouse/hub location."""
    phase = 2 * math.pi * (day_of_year - 30) / 365
    base = 55.0 + 25.0 * math.sin(phase)
    base += (year - 2021) * 0.2
    return base


def _is_holiday_week(d: pd.Timestamp) -> int:
    for m, d_start, d_end in _HOLIDAY_WEEKS:
        if d.month == m and d_start <= d.day <= d_end:
            return 1
    return 0


def generate(
    start: str = START,
    end: str = END,
    seed: int = SEED,
    out_path: str | Path = OUT_PATH,
) -> pd.DataFrame:
    """Generate the full synthetic weekly dataset and write to CSV."""
    rng = np.random.default_rng(seed)
    # Weekly frequency starting every Monday
    dates = pd.date_range(start, end, freq="W-MON")
    n = len(dates)

    # ── Spot price ($/module) ────────────────────────────────────────
    # Base price ~$45-85, with market cycles
    price_base = np.linspace(62, 52, n)  # long-term price decline (Moore's law)
    price_cycle = 12 * np.sin(2 * np.pi * np.arange(n) / 52)  # annual cycle
    price_noise = rng.normal(0, 3, n)
    avg_spot_price = np.clip(price_base + price_cycle + price_noise, 35, 95).round(2)

    # ── Temperature at distribution hub ──────────────────────────────
    temperature = np.array([
        _seasonal_temperature(d.day_of_year, d.year) + rng.normal(0, 5)
        for d in dates
    ]).round(1)
    temperature = np.clip(temperature, 5, 105)

    # ── Holiday week flag ────────────────────────────────────────────
    is_holiday_week = np.array([_is_holiday_week(d) for d in dates])

    # ── Back-to-school flag (Aug–Sep) ────────────────────────────────
    is_back_to_school = np.array([1 if d.month in (8, 9) else 0 for d in dates])

    # ── Black Friday / Cyber Monday week ─────────────────────────────
    is_black_friday = np.array([
        1 if d.month == 11 and 23 <= d.day <= 30 else 0 for d in dates
    ])

    # ── Prime Day / mid-year sale week ───────────────────────────────
    is_prime_day = np.array([
        1 if d.month == 7 and 8 <= d.day <= 14 else 0 for d in dates
    ])

    # ── Supply constraint (0–1 scale) ────────────────────────────────
    supply_constraint = np.zeros(n)
    for i, d in enumerate(dates):
        # Chip shortage era (2021-2022)
        if d.year == 2021:
            supply_constraint[i] = rng.uniform(0.10, 0.35)
        elif d.year == 2022 and d.month <= 6:
            supply_constraint[i] = rng.uniform(0.05, 0.25)
        elif d.year == 2022:
            supply_constraint[i] = rng.uniform(0.0, 0.10)
        else:
            supply_constraint[i] = rng.uniform(0.0, 0.05)
    supply_constraint = supply_constraint.round(3)

    # ── New GPU / CPU launch week ────────────────────────────────────
    new_gpu_launch = np.array([
        1 if (d.isocalendar()[1],) and (d.year, d.isocalendar()[1]) in _GPU_LAUNCH_WEEKS else 0
        for d in dates
    ])

    # ── Target: units_sold ───────────────────────────────────────────
    base_demand = 8500  # weekly baseline units

    units = np.zeros(n)
    for i, d in enumerate(dates):
        demand = base_demand

        # ── Long-term growth trend (AI / cloud boom) ─────────────────
        weeks_since_start = i
        demand += weeks_since_start * 3.5  # ~3.5 units/week growth

        # ── Year-over-year market expansion ──────────────────────────
        demand += (d.year - 2021) * 250

        # ── Seasonal: back-to-school boost ───────────────────────────
        if is_back_to_school[i]:
            demand *= 1.18

        # ── Black Friday spike ───────────────────────────────────────
        if is_black_friday[i]:
            demand *= 1.35

        # ── Prime Day spike ──────────────────────────────────────────
        if is_prime_day[i]:
            demand *= 1.22

        # ── GPU / CPU launch uplift ──────────────────────────────────
        if new_gpu_launch[i]:
            demand *= 1.0 + rng.uniform(0.20, 0.40)

        # ── Holiday week reduction (warehouses / shipping slow) ──────
        if is_holiday_week[i]:
            if d.month == 12 and d.day >= 22:
                demand *= 0.55  # Christmas week — minimal shipping
            elif d.month == 11 and d.day >= 22:
                pass  # Thanksgiving week already handled by BF
            elif d.month == 7:
                demand *= 0.78  # July 4th week
            else:
                demand *= 0.88

        # ── Temperature effect (hot summers → more PC building) ──────
        temp = temperature[i]
        if temp > 85:
            demand *= 1.06
        elif temp < 35:
            demand *= 0.95

        # ── Price elasticity (negative) ──────────────────────────────
        demand -= (avg_spot_price[i] - 55) * 45  # ~45 units per $1 above mean

        # ── Supply constraint dampening ──────────────────────────────
        demand *= (1.0 - supply_constraint[i])

        # ── Noise ────────────────────────────────────────────────────
        demand += rng.normal(0, 220)
        units[i] = max(int(round(demand)), 500)

    # ── Cyclic week-of-year encoding ─────────────────────────────────
    week_nums = np.array([d.isocalendar()[1] for d in dates])
    week_sin = np.sin(2 * np.pi * week_nums / 52).round(6)
    week_cos = np.cos(2 * np.pi * week_nums / 52).round(6)

    df = pd.DataFrame({
        "ds": dates,
        "units_sold": units.astype(int),
        "avg_spot_price_usd": avg_spot_price,
        "avg_temperature_f": temperature,
        "is_holiday_week": is_holiday_week,
        "is_back_to_school": is_back_to_school,
        "is_black_friday_week": is_black_friday,
        "is_prime_day_week": is_prime_day,
        "pct_supply_constraint": supply_constraint,
        "new_gpu_launch": new_gpu_launch,
        "week_of_year_sin": week_sin,
        "week_of_year_cos": week_cos,
    })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"✓ Wrote {len(df)} weekly rows → {out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", action="store_true", help="Print row count only")
    args = parser.parse_args()
    df = generate()
    if args.rows:
        print(len(df))

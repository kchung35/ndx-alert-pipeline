"""Unit tests for factor computations (no data dependencies)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors import (
    _zscore, compute_lowvol_z, compute_momentum, compute_quality_z,
    compute_value_z,
)


def _synth_prices(n=260, seed=42) -> pd.DataFrame:
    """Three tickers with a deterministic linear trend and zero noise so
    momentum ordering is unambiguous — the factor test isn't meant to be
    a noisy-return sanity check."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    out = {
        "AAA": 100 * np.linspace(1.0, 1.05, n),
        "BBB": 100 * np.linspace(1.0, 1.15, n),
        "CCC": 100 * np.linspace(1.0, 1.35, n),
    }
    return pd.DataFrame(out, index=dates)


def test_momentum_reflects_trend():
    prices = _synth_prices()
    mom = compute_momentum(prices, prices.index[-1])
    assert mom.idxmax() == "CCC"
    assert mom.idxmin() == "AAA"


def test_value_z_high_when_pe_low():
    fund = pd.DataFrame([
        {"ticker": "CHEAP", "trailingPE": 5, "priceToBook": 1, "enterpriseToEbitda": 5},
        {"ticker": "RICH", "trailingPE": 40, "priceToBook": 10, "enterpriseToEbitda": 30},
        {"ticker": "MID", "trailingPE": 15, "priceToBook": 3, "enterpriseToEbitda": 12},
    ])
    z = compute_value_z(fund)
    assert z["CHEAP"] > z["RICH"]


def test_quality_z_high_when_roe_high_debt_low():
    fund = pd.DataFrame([
        {"ticker": "GOOD", "returnOnEquity": 0.30, "returnOnAssets": 0.15,
         "grossMargins": 0.60, "debtToEquity": 10},
        {"ticker": "BAD", "returnOnEquity": 0.05, "returnOnAssets": 0.02,
         "grossMargins": 0.20, "debtToEquity": 200},
        {"ticker": "MID", "returnOnEquity": 0.15, "returnOnAssets": 0.08,
         "grossMargins": 0.40, "debtToEquity": 50},
    ])
    z = compute_quality_z(fund)
    assert z["GOOD"] > z["BAD"]


def test_lowvol_z_penalizes_volatile_tickers():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    df = pd.DataFrame({
        "CALM": 100 * np.cumprod(1 + rng.normal(0, 0.003, 80)),
        "WILD": 100 * np.cumprod(1 + rng.normal(0, 0.030, 80)),
    }, index=dates)
    z = compute_lowvol_z(df, df.index[-1])
    assert z["CALM"] > z["WILD"]


def test_zscore_handles_constant_series():
    s = pd.Series([1.0, 1.0, 1.0])
    z = _zscore(s)
    assert (z == 0).all()


def test_sector_neutralize_removes_sector_mean():
    """After _sector_neutralize, each sector's mean should be ~0."""
    from src.factors import _sector_neutralize
    values = pd.Series([3.0, 5.0, 1.0, 2.0, -1.0, 4.0],
                       index=["A", "B", "C", "D", "E", "F"])
    # A,B,C in Tech; D,E,F in Health (n=3 each). Pass min_sector_size=3
    # explicitly so both groups are neutralized in this unit test.
    sectors = pd.Series(["Tech"] * 3 + ["Health"] * 3,
                        index=["A", "B", "C", "D", "E", "F"])
    out = _sector_neutralize(values, sectors, min_sector_size=3)
    tech_demeaned = out[["A", "B", "C"]]
    health_demeaned = out[["D", "E", "F"]]
    assert abs(tech_demeaned.mean()) < 1e-9
    assert abs(health_demeaned.mean()) < 1e-9
    # Tech mean was 3; A = 3 - 3 = 0, B = 5 - 3 = 2, C = 1 - 3 = -2.
    assert abs(out["A"]) < 1e-9
    assert abs(out["B"] - 2.0) < 1e-9
    assert abs(out["C"] + 2.0) < 1e-9


def test_sector_neutralize_tiny_sector_preserved_not_zeroed():
    """A sector with fewer than min_sector_size members must NOT be
    neutralized to zero. Otherwise a solo sector (e.g. Real Estate with 1
    NDX member) would lose its entire signal."""
    from src.factors import _sector_neutralize
    # 6 tickers in Tech, 1 in Real Estate. Default min_sector_size = 5.
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
                       index=["A", "B", "C", "D", "E", "F", "Z"])
    sectors = pd.Series(
        ["Tech"] * 6 + ["RealEstate"],
        index=["A", "B", "C", "D", "E", "F", "Z"],
    )
    out = _sector_neutralize(values, sectors)
    # Tech mean = 3.5 -> each Tech ticker shifts by -3.5
    assert abs(out["A"] - (1.0 - 3.5)) < 1e-9
    assert abs(out["D"] - (4.0 - 3.5)) < 1e-9
    # Real Estate solo: NOT neutralized, keeps raw value
    assert abs(out["Z"] - 100.0) < 1e-9


def test_sector_neutralize_respects_custom_min_size():
    from src.factors import _sector_neutralize
    values = pd.Series([1.0, 3.0, 5.0], index=["A", "B", "C"])
    sectors = pd.Series(["X", "X", "Y"], index=["A", "B", "C"])
    # With min_sector_size=2, sector X (n=2) IS neutralized, Y (n=1) isn't.
    out = _sector_neutralize(values, sectors, min_sector_size=2)
    # X mean = 2.0 -> A = -1.0, B = +1.0
    assert abs(out["A"] + 1.0) < 1e-9
    assert abs(out["B"] - 1.0) < 1e-9
    assert abs(out["C"] - 5.0) < 1e-9

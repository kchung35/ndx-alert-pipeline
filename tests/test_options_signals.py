"""Unit tests for the options signal metrics (single-snapshot math).

Uses a synthetic chain so the test is deterministic and doesn't hit Yahoo.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.options_signals import (
    _atm_iv_at_tenor, _metrics_for_ticker, _score_options_panel,
    compute_options_panel,
)


def _make_chain(spot: float, strikes: list[float], exp: str, as_of: str,
                call_vol: int, put_vol: int, call_oi: int, put_oi: int,
                call_iv: float = 0.25, put_iv: float = 0.30) -> pd.DataFrame:
    rows = []
    for k in strikes:
        rows.append({
            "ticker": "TEST", "side": "call", "expiry": exp, "strike": k,
            "last_price": 1.0, "bid": 0.9, "ask": 1.1, "mid": 1.0,
            "volume": call_vol, "open_interest": call_oi,
            "implied_volatility": call_iv, "contract_symbol": f"T-C-{k}",
            "as_of_date": as_of,
        })
        rows.append({
            "ticker": "TEST", "side": "put", "expiry": exp, "strike": k,
            "last_price": 1.0, "bid": 0.9, "ask": 1.1, "mid": 1.0,
            "volume": put_vol, "open_interest": put_oi,
            "implied_volatility": put_iv, "contract_symbol": f"T-P-{k}",
            "as_of_date": as_of,
        })
    return pd.DataFrame(rows)


def test_vol_oi_ratios_correct():
    chain = _make_chain(
        spot=100, strikes=[95, 100, 105], exp="2026-05-20", as_of="2026-04-21",
        call_vol=50, put_vol=20, call_oi=100, put_oi=200,
    )
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    # 50 vol / 100 OI across 3 strikes for calls -> 150/300 = 0.5
    assert abs(m["vol_oi_call"] - 0.5) < 1e-6
    # 20 vol / 200 OI across 3 strikes -> 60/600 = 0.1
    assert abs(m["vol_oi_put"] - 0.1) < 1e-6


def test_flow_ratio_skewed_by_volume():
    chain = _make_chain(
        spot=100, strikes=[95, 100, 105], exp="2026-05-20", as_of="2026-04-21",
        call_vol=80, put_vol=20, call_oi=100, put_oi=100,
    )
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    # 80 * 1.0 call mid vs (80+20) -> 0.8
    assert abs(m["flow_ratio"] - 0.8) < 1e-6


def test_iv_skew_positive_when_puts_richer():
    chain = _make_chain(
        spot=100, strikes=[90, 100, 110], exp="2026-05-20", as_of="2026-04-21",
        call_vol=10, put_vol=10, call_oi=100, put_oi=100,
        call_iv=0.25, put_iv=0.35,
    )
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    assert abs(m["iv_skew"] - 0.10) < 1e-6


# ── Regression tests for the Fix B filters ───────────────────────────

def test_near_expiry_dte_filter_excludes_weeklies():
    # Expiry = 3 days from as_of -> should be filtered out by DTE >= 7.
    chain = _make_chain(
        spot=100, strikes=[95, 100, 105], exp="2026-04-24", as_of="2026-04-21",
        call_vol=50, put_vol=50, call_oi=100, put_oi=100,
    )
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    # No qualifying rows -> V/OI should be NaN
    assert math.isnan(m["vol_oi_call"])
    assert math.isnan(m["vol_oi_put"])


def test_zero_oi_rows_excluded_from_v_oi():
    # Build mixed chain: half with OI=0, half with OI=100.
    rows = []
    for k in (95, 100, 105):
        for side in ("call", "put"):
            # Strike 95 has OI=0 -> must be excluded; others have OI=100.
            oi = 0 if k == 95 else 100
            rows.append({
                "ticker": "TEST", "side": side, "expiry": "2026-05-20",
                "strike": k, "last_price": 1.0, "bid": 0.9, "ask": 1.1, "mid": 1.0,
                "volume": 10, "open_interest": oi, "implied_volatility": 0.25,
                "contract_symbol": f"T-{side}-{k}", "as_of_date": "2026-04-21",
            })
    chain = pd.DataFrame(rows)
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    # Only strikes 100 and 105 contribute: 2 strikes * 10 vol / (2 * 100 OI) = 0.1
    assert abs(m["vol_oi_call"] - 0.1) < 1e-6


def test_stale_iv_filtered_from_term_structure():
    # Front expiry has IV ~0 (stale), back expiry has real IV.
    # _atm_iv_at_tenor must return None for front -> iv_term is NaN.
    rows = []
    for (exp, iv) in [("2026-05-20", 0.001), ("2026-07-20", 0.25)]:
        for k in (95, 100, 105):
            for side in ("call", "put"):
                rows.append({
                    "ticker": "TEST", "side": side, "expiry": exp,
                    "strike": k, "last_price": 1.0, "bid": 0.9, "ask": 1.1, "mid": 1.0,
                    "volume": 10, "open_interest": 100, "implied_volatility": iv,
                    "contract_symbol": f"T-{side}-{k}-{exp}", "as_of_date": "2026-04-21",
                })
    chain = pd.DataFrame(rows)
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    assert math.isnan(m["iv_term"])
    assert math.isnan(m["iv_rel_vix"])


def test_atm_iv_tenor_skips_stale_exact_atm_rows():
    rows = []
    for side, strike, iv in (
        ("call", 100, 0.001),
        ("put", 100, 0.001),
        ("call", 105, 0.28),
        ("put", 95, 0.22),
    ):
        rows.append({
            "ticker": "TEST", "side": side, "expiry": "2026-05-21",
            "strike": strike, "last_price": 1.0, "bid": 0.9, "ask": 1.1, "mid": 1.0,
            "volume": 10, "open_interest": 100, "implied_volatility": iv,
            "contract_symbol": f"T-{side}-{strike}", "as_of_date": "2026-04-21",
        })
    chain = pd.DataFrame(rows)
    assert abs(_atm_iv_at_tenor(chain, spot=100.0, target_days=30) - 0.25) < 1e-6


def _panel_row(ticker: str, *, flow_ratio: float = 0.5,
               vol_oi_call: float | None = 0.1,
               vol_oi_put: float | None = 0.1,
               iv_skew: float = 0.0,
               iv_term: float = 1.0,
               iv_rel_vix: float = 1.0,
               quality: str = "HIGH") -> dict:
    return {
        "ticker": ticker,
        "vol_oi_call": vol_oi_call,
        "vol_oi_put": vol_oi_put,
        "flow_ratio": flow_ratio,
        "iv_skew": iv_skew,
        "iv_term": iv_term,
        "iv_rel_vix": iv_rel_vix,
        "contract_count": 500,
        "expiry_count": 6,
        "positive_volume_coverage": 0.8,
        "positive_oi_coverage": 0.15,
        "valid_iv_coverage": 0.5,
        "quote_sanity": 1.0,
        "options_coverage": 0.9,
        "options_quality": quality,
    }


def test_high_call_dollar_share_raises_flow_score():
    panel = pd.DataFrame([
        _panel_row("CALL", flow_ratio=0.9, vol_oi_call=0.5, vol_oi_put=0.1),
        _panel_row("NEUTRAL", flow_ratio=0.5, vol_oi_call=0.1, vol_oi_put=0.1),
        _panel_row("PUT", flow_ratio=0.1, vol_oi_call=0.1, vol_oi_put=0.5),
    ])
    scored = _score_options_panel(panel).set_index("ticker")
    assert scored.loc["CALL", "flow_score"] > 0
    assert scored.loc["CALL", "flow_score"] > scored.loc["PUT", "flow_score"]


def test_high_put_v_oi_lowers_flow_score():
    panel = pd.DataFrame([
        _panel_row("PUT_HEAVY", flow_ratio=0.5, vol_oi_call=0.1, vol_oi_put=0.8),
        _panel_row("CALL_HEAVY", flow_ratio=0.5, vol_oi_call=0.8, vol_oi_put=0.1),
        _panel_row("EVEN", flow_ratio=0.5, vol_oi_call=0.2, vol_oi_put=0.2),
    ])
    scored = _score_options_panel(panel).set_index("ticker")
    assert scored.loc["PUT_HEAVY", "flow_score"] < 0
    assert scored.loc["PUT_HEAVY", "flow_score"] < scored.loc["CALL_HEAVY", "flow_score"]


def test_rich_put_skew_lowers_skew_score():
    panel = pd.DataFrame([
        _panel_row("RICH_PUTS", iv_skew=0.40),
        _panel_row("FLAT", iv_skew=0.00),
        _panel_row("RICH_CALLS", iv_skew=-0.20),
    ])
    scored = _score_options_panel(panel).set_index("ticker")
    assert scored.loc["RICH_PUTS", "skew_score"] < 0
    assert scored.loc["RICH_PUTS", "skew_score"] < scored.loc["RICH_CALLS", "skew_score"]


def test_low_quality_chain_forces_neutral_options_z():
    panel = pd.DataFrame([
        _panel_row("LOW_BAD", flow_ratio=1.0, vol_oi_call=2.0, vol_oi_put=0.0,
                   iv_skew=-0.5, iv_term=0.5, iv_rel_vix=0.5, quality="LOW"),
        _panel_row("HIGH_OK", flow_ratio=0.1, vol_oi_call=0.0, vol_oi_put=1.0,
                   iv_skew=0.5, iv_term=2.0, iv_rel_vix=2.0, quality="HIGH"),
    ])
    scored = _score_options_panel(panel).set_index("ticker")
    assert scored.loc["LOW_BAD", "options_quality"] == "LOW"
    assert scored.loc["LOW_BAD", "options_z"] == 0.0
    assert scored.loc["LOW_BAD", "options_z_raw"] != 0.0


def test_missing_oi_does_not_create_fake_extreme_score():
    panel = pd.DataFrame([
        _panel_row("NO_OI_CALLS", flow_ratio=0.9, vol_oi_call=None, vol_oi_put=None),
        _panel_row("NO_OI_PUTS", flow_ratio=0.1, vol_oi_call=None, vol_oi_put=None),
        _panel_row("BALANCED", flow_ratio=0.5, vol_oi_call=None, vol_oi_put=None),
    ])
    scored = _score_options_panel(panel)
    assert scored["options_z"].notna().all()
    assert scored["options_z"].between(-1.0, 1.0).all()
    assert scored["flow_score"].between(-1.0, 1.0).all()


def test_current_snapshot_panel_has_quality_columns():
    if not Path("data/chains/2026-04-21").exists():
        pytest.skip("current snapshot data is not present")
    panel = compute_options_panel(date(2026, 4, 21))
    required = {
        "options_z", "options_z_raw", "options_quality", "options_coverage",
        "flow_score", "skew_score", "vol_stress_score",
    }
    assert len(panel) == 101
    assert required.issubset(panel.columns)
    assert panel["options_quality"].isin(["HIGH", "MEDIUM", "LOW"]).all()
    assert panel["options_z"].notna().all()

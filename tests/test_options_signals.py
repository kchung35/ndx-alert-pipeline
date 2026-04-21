"""Unit tests for the options signal metrics (single-snapshot math).

Uses a synthetic chain so the test is deterministic and doesn't hit Yahoo.
"""

from __future__ import annotations

import pandas as pd

from src.options_signals import _metrics_for_ticker


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
    import math
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
    import pandas as pd
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
    import math
    import pandas as pd
    chain = pd.DataFrame(rows)
    m = _metrics_for_ticker(chain, spot=100.0, vix=18.0)
    assert math.isnan(m["iv_term"])
    assert math.isnan(m["iv_rel_vix"])

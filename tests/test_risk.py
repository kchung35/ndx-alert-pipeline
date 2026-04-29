"""Tests for alert-book risk construction."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src import risk
from src.alert_engine import (
    TIER_BEARISH,
    TIER_BULLISH,
    TIER_CONFLUENCE_BEAR,
    TIER_CONFLUENCE_BULL,
    TIER_MOMENTUM_LONG,
    TIER_NONE,
)


def test_book_returns_uses_all_actionable_tiers(monkeypatch):
    monkeypatch.setattr(risk, "load_alerts", lambda _as_of: pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        "tier": [
            TIER_BULLISH,
            TIER_MOMENTUM_LONG,
            TIER_CONFLUENCE_BULL,
            TIER_BEARISH,
            TIER_CONFLUENCE_BEAR,
            TIER_NONE,
        ],
    }))
    monkeypatch.setattr(risk, "load_prices", lambda: pd.DataFrame({
        "date": ["2026-04-20"] * 6 + ["2026-04-21"] * 6,
        "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"] * 2,
        "adj_close": [100, 100, 100, 100, 100, 100, 110, 108, 106, 90, 96, 120],
    }))

    rets = risk.book_returns(date(2026, 4, 21))

    expected = (0.10 + 0.75 * 0.08 + 0.5 * 0.06 + (-1.0) * -0.10 + (-0.5) * -0.04) / 3.75
    assert len(rets) == 1
    assert abs(rets.iloc[0] - expected) < 1e-9

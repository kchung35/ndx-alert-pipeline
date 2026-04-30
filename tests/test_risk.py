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


def test_book_returns_excludes_prices_after_as_of(monkeypatch):
    monkeypatch.setattr(risk, "load_alerts", lambda _as_of: pd.DataFrame({
        "ticker": ["AAA"],
        "tier": [TIER_BULLISH],
    }))
    monkeypatch.setattr(risk, "load_prices", lambda: pd.DataFrame({
        "date": ["2026-04-20", "2026-04-21", "2026-04-22"],
        "ticker": ["AAA", "AAA", "AAA"],
        "adj_close": [100, 110, 220],
    }))

    rets = risk.book_returns(date(2026, 4, 21))

    assert list(rets.index) == [pd.Timestamp("2026-04-21")]
    assert abs(rets.iloc[0] - 0.10) < 1e-9


def test_book_returns_uses_trailing_two_year_window(monkeypatch):
    monkeypatch.setattr(risk, "load_alerts", lambda _as_of: pd.DataFrame({
        "ticker": ["AAA"],
        "tier": [TIER_BULLISH],
    }))
    monkeypatch.setattr(risk, "load_prices", lambda: pd.DataFrame({
        "date": ["2024-04-20", "2024-04-21", "2026-04-21"],
        "ticker": ["AAA", "AAA", "AAA"],
        "adj_close": [100, 110, 121],
    }))

    rets = risk.book_returns(date(2026, 4, 21))

    assert list(rets.index) == [
        pd.Timestamp("2024-04-21"),
        pd.Timestamp("2026-04-21"),
    ]


def test_risk_report_detail_returns_summary_and_curves(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=40)
    rets = pd.Series([0.01, -0.004, 0.006, -0.002] * 10, index=idx)
    monkeypatch.setattr(risk, "book_returns", lambda _as_of: rets)

    detail = risk.risk_report_detail(date(2026, 4, 21), horizon_days=20)

    assert detail["summary"]["n_days"] == 40
    assert detail["summary"]["horizon_days"] == 20
    assert len(detail["returns"]) == 40
    assert len(detail["equity"]) == 40
    assert len(detail["drawdown"]) == 40
    assert risk.risk_report(date(2026, 4, 21), horizon_days=20) == detail["summary"]

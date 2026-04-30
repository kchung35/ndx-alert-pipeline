"""Tests for factor-only L/S backtest helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import backtest


def test_auto_valid_backtest_start_uses_first_monthly_rebalance_with_history():
    idx = pd.bdate_range("2025-01-01", periods=320)
    prices = pd.DataFrame({"AAA": np.linspace(100, 130, len(idx))}, index=idx)

    start = backtest.auto_valid_backtest_start(
        idx[-1].date(),
        min_history_days=260,
        prices_wide=prices,
    )

    rebalances = backtest._rebalance_dates(idx, idx[0], idx[-1])
    expected = next(ts for ts in rebalances if len(prices.loc[:ts]) >= 260)
    assert start == expected.date()


def test_run_backtest_detail_returns_summary_and_curves(monkeypatch):
    tickers = [f"T{i:02d}" for i in range(20)]
    idx = pd.bdate_range("2026-01-01", periods=50)
    data = {}
    for i, ticker in enumerate(tickers):
        daily = 1.0 + (i - 9.5) * 0.0002
        data[ticker] = 100 * np.cumprod(np.full(len(idx), daily))
    prices = pd.DataFrame(data, index=idx)

    monkeypatch.setattr(backtest, "_prices_wide", lambda: prices)
    monkeypatch.setattr(backtest, "load_fundamentals", lambda: pd.DataFrame())
    monkeypatch.setattr(
        backtest,
        "load_universe",
        lambda: pd.DataFrame({"ticker": tickers}),
    )
    monkeypatch.setattr(
        backtest,
        "_factor_z_snapshot",
        lambda _prices, _fundamentals, _as_of: pd.Series(
            np.linspace(-1, 1, len(tickers)),
            index=tickers,
        ),
    )

    detail = backtest.run_backtest_detail(
        date(2026, 1, 1),
        date(2026, 2, 28),
        horizon_days=20,
        decile=0.10,
    )

    summary = detail["summary"]
    assert summary["rebalances"] == 2
    assert summary["trading_days"] > 0
    assert summary["horizon_days"] == 20
    assert summary["decile"] == 0.10
    assert len(detail["returns"]) == summary["trading_days"]
    assert len(detail["equity"]) == summary["trading_days"]
    assert len(detail["drawdown"]) == summary["trading_days"]
    assert backtest.run_backtest(
        date(2026, 1, 1),
        date(2026, 2, 28),
        horizon_days=20,
        decile=0.10,
    ) == summary

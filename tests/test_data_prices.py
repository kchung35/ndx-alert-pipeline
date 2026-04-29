"""Tests for price-history helpers."""

from __future__ import annotations

import pandas as pd

from src.data_prices import latest_adj_close_on_or_before


def test_latest_adj_close_on_or_before_ignores_future_prices():
    prices = pd.DataFrame({
        "date": ["2026-04-20", "2026-04-21", "2026-04-22"],
        "ticker": ["AAPL", "AAPL", "AAPL"],
        "adj_close": [198.0, 200.0, 250.0],
    })

    assert latest_adj_close_on_or_before(prices, "AAPL", "2026-04-21") == 200.0


def test_latest_adj_close_on_or_before_returns_none_without_history():
    prices = pd.DataFrame({
        "date": ["2026-04-22"],
        "ticker": ["AAPL"],
        "adj_close": [250.0],
    })

    assert latest_adj_close_on_or_before(prices, "AAPL", "2026-04-21") is None

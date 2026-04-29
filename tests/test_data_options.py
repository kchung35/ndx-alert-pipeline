"""Tests for yfinance option-chain schema normalization."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd

from src import data_options


def test_fetch_chain_preserves_yfinance_contract_fields(monkeypatch):
    calls = pd.DataFrame([{
        "contractSymbol": "TEST260515C00100000",
        "lastTradeDate": pd.Timestamp("2026-04-20 15:30:00", tz="UTC"),
        "strike": 100.0,
        "lastPrice": 12.0,
        "bid": 12.0,
        "ask": 13.0,
        "change": 0.5,
        "percentChange": 4.2,
        "volume": 10,
        "openInterest": 100,
        "impliedVolatility": 0.32,
        "inTheMoney": True,
        "contractSize": "REGULAR",
        "currency": "USD",
    }])

    class FakeTicker:
        options = ["2026-05-15"]

        def option_chain(self, _expiry):
            return SimpleNamespace(calls=calls, puts=pd.DataFrame())

    monkeypatch.setattr(data_options.yf, "Ticker", lambda _ticker: FakeTicker())

    chain = data_options.fetch_chain("TEST", date(2026, 4, 21), max_expiries=1)

    assert {"last_trade_date", "change", "percent_change", "in_the_money",
            "contract_size", "currency"}.issubset(chain.columns)
    row = chain.iloc[0]
    assert row["contract_symbol"] == "TEST260515C00100000"
    assert bool(row["in_the_money"]) is True
    assert row["contract_size"] == "REGULAR"
    assert row["currency"] == "USD"
    assert row["mid"] == 12.5

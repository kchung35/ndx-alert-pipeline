"""Risk metrics for the live alert book.

Thin wrapper around src.lifted.analytics. Intentionally has no portfolio
concept — "book" here just means today's actionable alert tiers, weighted
by signal strength and direction.

CLI:
    python3 -m src.risk --date YYYY-MM-DD
        prints VaR / CVaR / Sharpe of the EW long-short alert book
        using adj_close returns over the last 2y.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.alert_engine import (
    load_alerts,
    TIER_BEARISH,
    TIER_BULLISH,
    TIER_CONFLUENCE_BEAR,
    TIER_CONFLUENCE_BULL,
    TIER_MOMENTUM_LONG,
)
from src.data_prices import load_prices
from src.lifted.analytics import (
    annualized_return, annualized_vol, conditional_var, historical_var,
    max_drawdown, sharpe_ratio, value_at_risk,
)

logger = logging.getLogger(__name__)

ACTIONABLE_TIER_WEIGHTS = {
    TIER_BULLISH: 1.0,
    TIER_MOMENTUM_LONG: 0.75,
    TIER_CONFLUENCE_BULL: 0.5,
    TIER_BEARISH: -1.0,
    TIER_CONFLUENCE_BEAR: -0.5,
}


def book_returns(as_of: date) -> pd.Series:
    """Weighted daily returns over the past 2y based on actionable alert tiers."""
    alerts = load_alerts(as_of)
    active = alerts.loc[alerts["tier"].isin(ACTIONABLE_TIER_WEIGHTS), ["ticker", "tier"]].copy()
    if active.empty:
        return pd.Series(dtype=float)
    active["weight"] = active["tier"].map(ACTIONABLE_TIER_WEIGHTS).astype(float)

    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()

    weights = active.set_index("ticker")["weight"]
    weights = weights[weights.index.intersection(wide.columns)]
    if weights.empty:
        return pd.Series(dtype=float)

    daily_rets = wide[weights.index].pct_change()
    weighted_rets = daily_rets.mul(weights, axis=1)
    gross_exposure = daily_rets.notna().mul(weights.abs(), axis=1).sum(axis=1)
    book = weighted_rets.sum(axis=1).where(gross_exposure > 0) / gross_exposure.where(gross_exposure > 0)
    return book.dropna()


def risk_report(as_of: date, horizon_days: int = 20) -> dict:
    rets = book_returns(as_of)
    if rets.empty:
        return {"note": "no alerts today -> empty book"}

    return {
        "n_days": int(len(rets)),
        "ann_return": annualized_return(rets),
        "ann_vol": annualized_vol(rets),
        "sharpe": sharpe_ratio(rets),
        "max_dd": max_drawdown((1 + rets).cumprod()),
        "var_95_1d_parametric": value_at_risk(rets, 0.95, 1),
        "var_95_20d_historical": historical_var(rets, 0.95, horizon_days),
        "cvar_95_20d": conditional_var(rets, 0.95, horizon_days),
    }


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=last_completed_trading_day().isoformat())
    parser.add_argument("--horizon-days", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = risk_report(date.fromisoformat(args.date), args.horizon_days)
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v: .6f}")
        else:
            print(f"  {k:30s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

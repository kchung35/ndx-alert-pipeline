"""Long/short backtest for the composite alert (factor-only day 1, full later).

Deliberately minimal — Backtrader's strategy engine is overkill for a
monthly-rebalanced decile spread. We implement a transparent walk-forward
loop in pandas, then gate it behind a --use-backtrader flag for the full
engine once the data is richer (options + insider history).

Strategy (matches user-confirmed params):
    - Universe: NDX 100 current members
    - Signal: factor_z composite on rebalance date (options/insider added later)
    - Rebalance: monthly (first trading day)
    - Holding: 20 trading days
    - Positions: top decile long, bottom decile short, equal-weight
    - Benchmark: equal-weight NDX 100

Outputs a summary dict with Sharpe, ann return, ann vol, max DD, hit rate.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.data_prices import load_prices
from src.factors import (
    compute_lowvol_z, compute_momentum, compute_quality_z, compute_value_z,
)
from src.data_prices import load_fundamentals
from src.lifted.analytics import (
    annualized_return, annualized_vol, max_drawdown, sharpe_ratio,
)
from src.universe import load_universe

logger = logging.getLogger(__name__)


def _prices_wide() -> pd.DataFrame:
    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def _rebalance_dates(index: pd.DatetimeIndex, start: pd.Timestamp,
                     end: pd.Timestamp) -> list[pd.Timestamp]:
    mask = (index >= start) & (index <= end)
    sub = index[mask]
    if sub.empty:
        return []
    dates = []
    last_month = None
    for ts in sub:
        key = (ts.year, ts.month)
        if key != last_month:
            dates.append(ts)
            last_month = key
    return dates


def _factor_z_snapshot(prices_wide: pd.DataFrame, fundamentals: pd.DataFrame,
                       as_of: pd.Timestamp) -> pd.Series:
    mom = compute_momentum(prices_wide, as_of)
    lowvol = compute_lowvol_z(prices_wide, as_of)
    value = compute_value_z(fundamentals)
    quality = compute_quality_z(fundamentals)

    def _z(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        mu, sd = s.mean(), s.std()
        if sd == 0 or np.isnan(sd):
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd

    mom_z = _z(mom)
    z = pd.concat([mom_z, value, quality, lowvol], axis=1).mean(axis=1)
    return z.dropna()


def run_backtest(start: date, end: date, horizon_days: int = 20,
                 decile: float = 0.10) -> dict:
    prices_wide = _prices_wide()
    fundamentals = load_fundamentals()
    universe = load_universe()
    tickers = [t for t in universe["ticker"].tolist() if t in prices_wide.columns]
    prices_wide = prices_wide[tickers]

    daily_rets = prices_wide.pct_change()

    rebalance = _rebalance_dates(prices_wide.index, pd.Timestamp(start), pd.Timestamp(end))
    if not rebalance:
        raise RuntimeError(f"No rebalance dates in {start}..{end}")

    all_ls_rets: list[pd.Series] = []
    for t0 in rebalance:
        z = _factor_z_snapshot(prices_wide, fundamentals, t0)
        z = z.dropna()
        if len(z) < 20:
            logger.warning("%s: only %d ranked tickers, skipping", t0.date(), len(z))
            continue
        k = max(1, int(len(z) * decile))
        longs = z.nlargest(k).index.tolist()
        shorts = z.nsmallest(k).index.tolist()

        # Hold for horizon_days trading days post rebalance
        try:
            loc = prices_wide.index.get_loc(t0)
        except KeyError:
            continue
        end_idx = min(loc + horizon_days, len(prices_wide.index) - 1)
        window = daily_rets.iloc[loc + 1:end_idx + 1]
        if window.empty:
            continue

        long_r = window[longs].mean(axis=1)
        short_r = window[shorts].mean(axis=1)
        ls = long_r - short_r
        all_ls_rets.append(ls)

    if not all_ls_rets:
        return {"note": "no rebalance produced returns"}

    ls_series = pd.concat(all_ls_rets).sort_index()
    # Drop duplicates from overlapping holding windows
    ls_series = ls_series.groupby(ls_series.index).sum()

    equity = (1 + ls_series).cumprod()
    return {
        "rebalances": len(rebalance),
        "trading_days": int(len(ls_series)),
        "ann_return": annualized_return(ls_series),
        "ann_vol": annualized_vol(ls_series),
        "sharpe": sharpe_ratio(ls_series),
        "max_dd": max_drawdown(equity),
        "hit_rate": float((ls_series > 0).mean()),
        "final_equity_multiplier": float(equity.iloc[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Factor-only NDX 100 L/S backtest.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-04-01")
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--decile", type=float, default=0.10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    report = run_backtest(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        horizon_days=args.horizon_days,
        decile=args.decile,
    )
    for k, v in report.items():
        if isinstance(v, float):
            print(f"  {k:30s} {v: .6f}")
        else:
            print(f"  {k:30s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

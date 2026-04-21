"""5-year validation of the MOMENTUM_LONG strategy.

Pulls 5y of NDX 100 prices directly from yfinance (does NOT overwrite the
live prices.parquet), then backtests 3-month momentum long-only in four
non-overlapping yearly windows to check for regime robustness — in
particular whether the strategy survived the 2022 tech bear market.

Run manually:
    SEC_USER_AGENT=... python3 tests/validate_momentum_5y.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lifted.analytics import (  # noqa: E402
    annualized_return, annualized_vol, max_drawdown, sharpe_ratio,
)
from src.universe import load_universe  # noqa: E402


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(lo, hi)
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def fetch_5y_prices(tickers: list[str]) -> pd.DataFrame:
    print(f"Fetching 5y of prices for {len(tickers)} tickers ...", flush=True)
    raw = yf.download(
        tickers=tickers, period="5y", progress=False, auto_adjust=False,
        group_by="ticker", threads=True,
    )
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t].dropna(how="all")[["Adj Close"]].rename(columns={"Adj Close": t})
        frames.append(sub)
    wide = pd.concat(frames, axis=1).sort_index()
    print(f"  -> {len(wide)} days x {wide.shape[1]} tickers",
          f"from {wide.index.min().date()} to {wide.index.max().date()}", flush=True)
    return wide


def monthly_rebalance(index: pd.DatetimeIndex, start: pd.Timestamp,
                      end: pd.Timestamp) -> list[pd.Timestamp]:
    mask = (index >= start) & (index <= end)
    sub = index[mask]
    dates, last = [], None
    for ts in sub:
        key = (ts.year, ts.month)
        if key != last:
            dates.append(ts)
            last = key
    return dates


def run_window(prices: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
               decile: float = 0.10, horizon: int = 20) -> dict:
    daily = prices.pct_change()
    rebal = monthly_rebalance(prices.index, start, end)
    longs_rets, bench_rets = [], []
    for t0 in rebal:
        hist = prices.loc[:t0]
        if len(hist) < 63:
            continue
        # 3m return
        ret_3m = hist.iloc[-1] / hist.iloc[-63] - 1
        z = _zscore(ret_3m).dropna()
        if len(z) < 10:
            continue
        k = max(1, int(len(z) * decile))
        longs = z.nlargest(k).index.tolist()
        loc = prices.index.get_loc(t0)
        window = daily.iloc[loc + 1: loc + 1 + horizon]
        if window.empty:
            continue
        longs_rets.append(window[longs].mean(axis=1))
        bench_rets.append(window[prices.columns].mean(axis=1))

    def _stats(rets_list):
        if not rets_list:
            return None
        s = pd.concat(rets_list).sort_index().groupby(level=0).sum()
        eq = (1 + s).cumprod()
        return dict(
            ann=annualized_return(s),
            vol=annualized_vol(s),
            sharpe=sharpe_ratio(s),
            max_dd=max_drawdown(eq),
            hit=float((s > 0).mean()),
            equity=float(eq.iloc[-1]),
        )
    return {"long": _stats(longs_rets), "bench": _stats(bench_rets),
            "n_rebal": len(rebal)}


def main() -> int:
    universe = load_universe()
    tickers = universe["ticker"].tolist()
    prices = fetch_5y_prices(tickers)

    # Non-overlapping yearly windows back to 2022 (covers the tech bear).
    # Using Oct-Oct so each window captures a full annual regime.
    windows = [
        ("2022 tech bear", "2022-01-01", "2022-12-31"),
        ("2023 recovery", "2023-01-01", "2023-12-31"),
        ("2024 full year", "2024-01-01", "2024-12-31"),
        ("2025 H1-H2",    "2025-01-01", "2025-12-31"),
        ("2026 YTD",      "2026-01-01", "2026-04-20"),
    ]
    print(f"\n{'Window':<18} {'Strategy':<18} {'Ann':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Hit':>6} {'Excess':>8}")
    print("-" * 90)
    for name, s, e in windows:
        out = run_window(prices, pd.Timestamp(s), pd.Timestamp(e))
        if out["long"] is None or out["bench"] is None:
            print(f"{name:<18} (no rebalances)")
            continue
        L, B = out["long"], out["bench"]
        excess = L["ann"] - B["ann"]
        print(f"{name:<18} {'Momentum 3m long':<18} "
              f"{L['ann']:+8.1%} {L['sharpe']:+8.2f} {L['max_dd']:+8.1%} "
              f"{L['hit']:>6.1%} {excess:+8.1%}")
        print(f"{name:<18} {'EW-NDX bench':<18} "
              f"{B['ann']:+8.1%} {B['sharpe']:+8.2f} {B['max_dd']:+8.1%} "
              f"{B['hit']:>6.1%} {'—':>8}")
        print(f"{'':<18} {'['+str(out['n_rebal'])+' rebalances]':<18}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

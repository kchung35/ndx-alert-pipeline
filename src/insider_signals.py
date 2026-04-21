"""Insider-transaction signal (NDX 100).

For each ticker, aggregates Form 4 transactions over the past 90 days into
a single signed score, then cross-sectionally z-scores across the universe.

Scoring logic:
    weight per row = signal_weight * officer_weight * recency_decay
        signal_weight  : -2..+3  (from classify_transaction)
        officer_weight : 1..5    (from officer_weight)
        recency_decay  : exp(-days_old / 30)

    raw_score = sum(weight * value_usd) per ticker
    cluster_bonus = 1 + 0.5 * distinct_buyers_30d (capped at 2x)
    normalized = (raw_score * cluster_bonus) / shares_outstanding_usd_proxy

Final output column: insider_z (cross-sectional z-score).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_edgar import load_form4
from src.data_prices import load_fundamentals
from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

INSIDER_SIG_DIR = PROJECT_ROOT / "data" / "insider_signals"

_WINDOW_DAYS = 90
_CLUSTER_WINDOW = 30
_DECAY_HALFLIFE = 30  # days


def _ticker_score(form4: pd.DataFrame, as_of: date) -> dict:
    """Aggregate Form 4 rows for one ticker into a scoring dict."""
    out = {
        "raw_score": 0.0,
        "net_buy_usd": 0.0,
        "distinct_buyers_30d": 0,
        "n_transactions": 0,
        "cluster_bonus": 1.0,
    }
    if form4.empty:
        return out

    df = form4.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    cutoff = pd.Timestamp(as_of - timedelta(days=_WINDOW_DAYS))
    df = df[df["transaction_date"] >= cutoff]
    df = df[df["transaction_date"] <= pd.Timestamp(as_of)]
    if df.empty:
        return out

    df["days_old"] = (pd.Timestamp(as_of) - df["transaction_date"]).dt.days
    df["decay"] = np.exp(-df["days_old"] / _DECAY_HALFLIFE)
    # Signed value: price can be missing for non-P/S codes — treat missing as 0 contribution
    df["value"] = df["value"].fillna(0.0).astype(float)
    df["weight"] = (
        df["signal_weight"].astype(float)
        * df["officer_weight"].astype(float)
        * df["decay"]
    )
    raw = (df["weight"] * df["value"]).sum()

    # Net $ buy = P - S
    net_buy = (
        df.loc[df["tx_code"] == "P", "value"].sum()
        - df.loc[df["tx_code"] == "S", "value"].sum()
    )

    # Cluster count: distinct insiders with any P transaction in last 30d
    cluster_cutoff = pd.Timestamp(as_of - timedelta(days=_CLUSTER_WINDOW))
    recent_buys = df[(df["tx_code"] == "P") & (df["transaction_date"] >= cluster_cutoff)]
    distinct = recent_buys["insider_name"].nunique()

    cluster_bonus = min(1.0 + 0.5 * distinct, 2.0)

    out.update(
        raw_score=raw * cluster_bonus,
        net_buy_usd=float(net_buy),
        distinct_buyers_30d=int(distinct),
        n_transactions=int(len(df)),
        cluster_bonus=float(cluster_bonus),
    )
    return out


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score with 5/95 winsorization.

    Tightened from 1/99% after the 101-ticker run showed a handful of
    huge insider sells (WDAY, WBD, MPWR) driving z to -4.95 — essentially
    min/max of a 101-sample dist escapes the 1/99 clip entirely. 5/95
    keeps 91 samples and compresses tail outliers that dominate the
    composite otherwise.
    """
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def compute_insider_panel(as_of: date) -> pd.DataFrame:
    """Build the per-ticker insider panel with cross-sectional z-score."""
    universe = load_universe()
    tickers = universe["ticker"].tolist()

    try:
        fundamentals = load_fundamentals()
    except FileNotFoundError:
        fundamentals = pd.DataFrame(columns=["ticker", "marketCap"])
    market_caps = fundamentals.set_index("ticker").get("marketCap", pd.Series(dtype=float))

    rows = []
    for t in tickers:
        form4 = load_form4(t)
        row = _ticker_score(form4, as_of)
        row["ticker"] = t
        mc = market_caps.get(t, np.nan)
        row["mcap_proxy"] = mc if mc and mc > 0 else np.nan
        # Normalize by market cap so $10M insider buy is big for a small-cap, noise for mega-cap
        if row["mcap_proxy"] and row["mcap_proxy"] > 0:
            row["normalized_score"] = row["raw_score"] / row["mcap_proxy"]
        else:
            row["normalized_score"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["insider_z"] = _zscore(df["normalized_score"])
    df["as_of_date"] = as_of.isoformat()
    return df


def save_insider_panel(df: pd.DataFrame, as_of: date) -> Path:
    INSIDER_SIG_DIR.mkdir(parents=True, exist_ok=True)
    path = INSIDER_SIG_DIR / f"{as_of.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_insider_panel(as_of: date) -> pd.DataFrame:
    path = INSIDER_SIG_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(path)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Compute insider signal panel.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.date)
    df = compute_insider_panel(as_of)
    save_insider_panel(df, as_of)
    print(df.sort_values("insider_z", ascending=False).head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

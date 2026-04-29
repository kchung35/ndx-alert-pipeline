"""yfinance price / fundamentals / VIX puller.

Clean replacement for pms_app's price_history_service (which is fused to
SQLite refresh-state, a /tmp lock file, and silent exception swallowing).

Outputs:
    data/prices.parquet        -- long format (date, ticker, open, high, low, close, volume, adj_close)
    data/fundamentals.parquet  -- one row per ticker with .info snapshot fields
    data/vix.parquet           -- ^VIX daily OHLCV
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

PRICES_PARQUET = PROJECT_ROOT / "data" / "prices.parquet"
FUNDAMENTALS_PARQUET = PROJECT_ROOT / "data" / "fundamentals.parquet"
VIX_PARQUET = PROJECT_ROOT / "data" / "vix.parquet"


# ── simple in-process token bucket (no /tmp state) ─────────────────────
class _TokenBucket:
    """Thread-safe min-interval gate. Sleeps to keep calls ≤ max_per_sec."""

    def __init__(self, max_per_sec: float):
        self._min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_YF_GATE = _TokenBucket(max_per_sec=4.0)  # yfinance is generous; 4 rps is safe


# ── price history ───────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Batch-download OHLCV for all tickers. Returns long-format DataFrame."""
    _YF_GATE.acquire()
    df = yf.download(
        tickers=tickers,
        period=period,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yf.download returned empty for {len(tickers)} tickers")

    # Convert wide multi-index to long format
    frames: list[pd.DataFrame] = []
    for t in tickers:
        if t not in df.columns.get_level_values(0):
            logger.warning("No price data returned for %s", t)
            continue
        sub = df[t].dropna(how="all").copy()
        sub.index.name = "date"
        sub = sub.reset_index()
        sub["ticker"] = t
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        frames.append(sub)

    if not frames:
        raise RuntimeError("No price frames produced from yf.download")
    out = pd.concat(frames, ignore_index=True)
    # Standardize columns
    rename = {"adj_close": "adj_close", "stock_splits": "splits"}
    out = out.rename(columns=rename)
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def save_prices(df: pd.DataFrame) -> Path:
    PRICES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICES_PARQUET, index=False)
    return PRICES_PARQUET


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PARQUET)


def latest_adj_close_on_or_before(
    prices: pd.DataFrame,
    ticker: str,
    as_of: date | pd.Timestamp | str,
) -> float | None:
    """Return ticker's latest adjusted close available on or before as_of."""
    required = {"date", "ticker", "adj_close"}
    if prices.empty or not required.issubset(prices.columns):
        return None

    as_of_ts = pd.Timestamp(as_of)
    sub = prices.copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub = sub[(sub["ticker"] == ticker) & (sub["date"] <= as_of_ts)]
    if sub.empty:
        return None

    value = sub.sort_values("date").iloc[-1]["adj_close"]
    if pd.isna(value):
        return None
    return float(value)


# ── fundamentals via .info ──────────────────────────────────────────────

_FUNDAMENTAL_FIELDS = [
    "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
    "priceToSalesTrailing12Months",
    "returnOnEquity", "returnOnAssets", "profitMargins",
    "operatingMargins", "grossMargins", "debtToEquity",
    "marketCap", "sharesOutstanding",
    "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
]


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        _YF_GATE.acquire()
        try:
            info = yf.Ticker(t).info or {}
        except Exception as exc:
            logger.warning(".info fetch failed for %s: %s", t, exc)
            info = {}
        row = {"ticker": t}
        for f in _FUNDAMENTAL_FIELDS:
            v = info.get(f)
            try:
                row[f] = float(v) if v is not None else None
            except (TypeError, ValueError):
                row[f] = None
        rows.append(row)
    return pd.DataFrame(rows)


def save_fundamentals(df: pd.DataFrame) -> Path:
    FUNDAMENTALS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FUNDAMENTALS_PARQUET, index=False)
    return FUNDAMENTALS_PARQUET


def load_fundamentals() -> pd.DataFrame:
    return pd.read_parquet(FUNDAMENTALS_PARQUET)


# ── VIX ────────────────────────────────────────────────────────────────

def fetch_vix(period: str = "1y") -> pd.DataFrame:
    _YF_GATE.acquire()
    df = yf.Ticker("^VIX").history(period=period, auto_adjust=False)
    if df.empty:
        raise RuntimeError("VIX fetch returned empty")
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df[["date", "open", "high", "low", "close"]]


def save_vix(df: pd.DataFrame) -> Path:
    VIX_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(VIX_PARQUET, index=False)
    return VIX_PARQUET


def load_vix() -> pd.DataFrame:
    return pd.read_parquet(VIX_PARQUET)


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Pull NDX 100 prices, fundamentals, VIX.")
    parser.add_argument("--period", default="2y", help="yfinance period (default 2y)")
    parser.add_argument("--skip-fundamentals", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    universe = load_universe()
    tickers = universe["ticker"].tolist()
    logger.info("Universe has %d tickers", len(tickers))

    logger.info("Fetching prices...")
    prices = fetch_prices(tickers, period=args.period)
    save_prices(prices)
    logger.info("Wrote %s (%d rows, %d tickers)",
                PRICES_PARQUET, len(prices), prices["ticker"].nunique())

    if not args.skip_fundamentals:
        logger.info("Fetching fundamentals...")
        funds = fetch_fundamentals(tickers)
        save_fundamentals(funds)
        logger.info("Wrote %s (%d tickers)", FUNDAMENTALS_PARQUET, len(funds))

    logger.info("Fetching VIX...")
    vix = fetch_vix(period=args.period)
    save_vix(vix)
    logger.info("Wrote %s (%d rows)", VIX_PARQUET, len(vix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

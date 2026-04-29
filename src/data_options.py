"""yfinance options-chain snapshot puller.

For each ticker, pulls the FULL chain (all expiries, calls + puts) and
stores one parquet per ticker per day under:
    data/chains/{YYYY-MM-DD}/{TICKER}.parquet

Each row is a single option contract with columns:
    ticker, side (call/put), expiry, strike, last_trade_date, last_price,
    bid, ask, mid, change, percent_change, volume, open_interest,
    implied_volatility, in_the_money, contract_size, currency,
    contract_symbol, as_of_date
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

CHAINS_ROOT = PROJECT_ROOT / "data" / "chains"

_CHAIN_COLS = [
    "ticker", "side", "expiry", "strike",
    "last_trade_date", "last_price", "bid", "ask", "mid",
    "change", "percent_change", "volume", "open_interest", "implied_volatility",
    "in_the_money", "contract_size", "currency",
    "contract_symbol", "as_of_date",
]


def _chain_day_dir(d: date) -> Path:
    return CHAINS_ROOT / d.isoformat()


def fetch_chain(ticker: str, as_of: date, max_expiries: int | None = None) -> pd.DataFrame:
    """Pull all expiries for ticker, return long-format chain."""
    tk = yf.Ticker(ticker)
    try:
        expiries = list(tk.options or [])
    except Exception as exc:
        logger.warning("options list failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=_CHAIN_COLS)

    if not expiries:
        return pd.DataFrame(columns=_CHAIN_COLS)
    if max_expiries:
        expiries = expiries[:max_expiries]

    frames: list[pd.DataFrame] = []
    for exp in expiries:
        try:
            ch = tk.option_chain(exp)
        except Exception as exc:
            logger.warning("chain fetch failed %s %s: %s", ticker, exp, exc)
            continue
        for side, df_side in (("call", ch.calls), ("put", ch.puts)):
            if df_side is None or df_side.empty:
                continue
            out = df_side.copy()
            out["ticker"] = ticker
            out["side"] = side
            out["expiry"] = exp
            out["as_of_date"] = as_of.isoformat()
            out = out.rename(columns={
                "contractSymbol": "contract_symbol",
                "lastTradeDate": "last_trade_date",
                "lastPrice": "last_price",
                "openInterest": "open_interest",
                "impliedVolatility": "implied_volatility",
                "percentChange": "percent_change",
                "inTheMoney": "in_the_money",
                "contractSize": "contract_size",
            })
            bid = pd.to_numeric(out.get("bid", 0.0), errors="coerce").fillna(0.0)
            ask = pd.to_numeric(out.get("ask", 0.0), errors="coerce").fillna(0.0)
            out["mid"] = (bid + ask) / 2.0
            # Keep only schema columns that exist; fill missing with NaN
            for col in _CHAIN_COLS:
                if col not in out.columns:
                    out[col] = pd.NA
            frames.append(out[_CHAIN_COLS])
        # Polite spacing between expiries
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame(columns=_CHAIN_COLS)
    return pd.concat(frames, ignore_index=True)


def save_chain(ticker: str, as_of: date, chain: pd.DataFrame) -> Path:
    day_dir = _chain_day_dir(as_of)
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{ticker}.parquet"
    chain.to_parquet(path, index=False)
    return path


def load_chain(ticker: str, as_of: date) -> pd.DataFrame:
    path = _chain_day_dir(as_of) / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def load_all_chains_for_date(as_of: date) -> pd.DataFrame:
    """Concat every ticker's chain for a given date. Empty DF if none."""
    day_dir = _chain_day_dir(as_of)
    if not day_dir.exists():
        return pd.DataFrame(columns=_CHAIN_COLS)
    frames = [pd.read_parquet(p) for p in day_dir.glob("*.parquet")]
    if not frames:
        return pd.DataFrame(columns=_CHAIN_COLS)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Snapshot NDX 100 options chains.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat(),
                        help="Stamp date on the snapshot (default: last completed trading day)")
    parser.add_argument("--max-expiries", type=int, default=None,
                        help="Cap per-ticker expiries (default: all)")
    parser.add_argument("--tickers", nargs="*", help="Subset tickers for debugging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.date)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_universe()["ticker"].tolist()

    written = 0
    for t in tickers:
        try:
            ch = fetch_chain(t, as_of, max_expiries=args.max_expiries)
        except Exception as exc:
            logger.warning("fetch_chain %s failed: %s", t, exc)
            continue
        if ch.empty:
            logger.warning("%s: empty chain", t)
            continue
        save_chain(t, as_of, ch)
        written += 1
        if written % 10 == 0:
            logger.info("progress: %d / %d", written, len(tickers))

    logger.info("Wrote %d chain parquet files under %s",
                written, _chain_day_dir(as_of))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

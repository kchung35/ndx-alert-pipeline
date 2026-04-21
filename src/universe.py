"""NASDAQ 100 universe resolver.

Sources, preferred order:
  1. Local cached JSON files (constituents_ndx100.json + market_caps_ndx100.json)
     -- near-instant, no network.
  2. Wikipedia's "Nasdaq-100" page -- live scrape.
  3. Static fallback CSV -- offline/air-gapped.

Output: data/universe.parquet with columns
    ticker, company, sector, industry, market_cap, cik,
    normalized_market_cap, issuer_group_size
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.lifted.index_universe import build_index_universe_view
from src.lifted.sec_identity import normalize_sec_ticker, ticker_to_cik

logger = logging.getLogger(__name__)

_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
_USER_AGENT = "Mozilla/5.0 (compatible; ndx-alert-pipeline)"

# Reject cached JSONs older than this. 45 days comfortably sits inside
# NDX's annual rebalance cycle without being trigger-happy on typical
# mid-year usage.
CACHE_MAX_AGE_DAYS = 45

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PARQUET = PROJECT_ROOT / "data" / "universe.parquet"
STATIC_FALLBACK = PROJECT_ROOT / "data" / "ndx_static_fallback.csv"
CACHED_CONSTITUENTS = PROJECT_ROOT / "data" / "constituents_ndx100.json"
CACHED_MARKET_CAPS = PROJECT_ROOT / "data" / "market_caps_ndx100.json"


def _cache_age_days(last_updated: str | None) -> int | None:
    """Return age of an ISO-timestamped cache in days, or None if unparseable."""
    if not last_updated:
        return None
    try:
        ts = pd.Timestamp(last_updated)
    except (ValueError, TypeError):
        return None
    # Pin both sides to the cache's tz domain for a clean subtraction.
    now = pd.Timestamp.now(tz=ts.tz) if ts.tz is not None else pd.Timestamp.now()
    return int((now - ts).days)


def _fetch_wikipedia() -> list[str]:
    """Scrape the Components table from the Nasdaq-100 Wikipedia page."""
    resp = requests.get(_WIKI_URL, timeout=30, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if any(c.startswith("ticker") or c == "symbol" for c in cols):
            ticker_col = next(
                c for c in tbl.columns
                if str(c).strip().lower().startswith("ticker")
                or str(c).strip().lower() == "symbol"
            )
            tickers = [
                normalize_sec_ticker(str(t)) for t in tbl[ticker_col].dropna()
            ]
            tickers = [t for t in tickers if t]
            if 90 <= len(tickers) <= 110:
                return tickers
    raise RuntimeError("No Components-shaped table found on Wikipedia page")


def _load_static_fallback() -> list[str]:
    if not STATIC_FALLBACK.exists():
        return []
    df = pd.read_csv(STATIC_FALLBACK)
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    return [normalize_sec_ticker(str(t)) for t in df[col].dropna()]


def _load_cached_constituents(max_age_days: int = CACHE_MAX_AGE_DAYS) -> pd.DataFrame | None:
    """Load the bundled NDX 100 constituent JSON (with sector, company).

    Returns None if the file is missing, malformed, or older than
    max_age_days. On stale hits we emit WARNING so the caller knows to
    refresh via the Wikipedia fallback rather than silently serving stale
    membership.
    """
    if not CACHED_CONSTITUENTS.exists():
        return None
    try:
        payload = json.loads(CACHED_CONSTITUENTS.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cached constituents unreadable: %s", exc)
        return None

    last_updated = payload.get("last_updated")
    age = _cache_age_days(last_updated)
    if age is not None and age > max_age_days:
        logger.warning(
            "cached constituents stale (%d days > %d); falling back to Wikipedia. "
            "Run `python3 -m src.universe --refresh-universe` to regenerate.",
            age, max_age_days,
        )
        return None

    rows = payload.get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["ticker"] = df["ticker"].map(normalize_sec_ticker)
    df = df[df["ticker"].astype(bool)].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    df["industry"] = ""  # not present in the cached file; yfinance fills later if desired
    logger.info("Loaded %d constituents from cache (updated %s, %d days ago)",
                len(df), last_updated, age if age is not None else -1)
    return df


def _load_cached_market_caps(max_age_days: int = CACHE_MAX_AGE_DAYS) -> dict[str, float]:
    """Load cached market caps (values in billions). Returns {} if missing or stale."""
    if not CACHED_MARKET_CAPS.exists():
        return {}
    try:
        payload = json.loads(CACHED_MARKET_CAPS.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cached market caps unreadable: %s", exc)
        return {}

    last_updated = payload.get("last_updated")
    age = _cache_age_days(last_updated)
    if age is not None and age > max_age_days:
        logger.warning(
            "cached market caps stale (%d days); will be refilled by fundamentals fetch",
            age,
        )
        return {}

    data = payload.get("data") or {}
    out: dict[str, float] = {}
    for raw_ticker, value in data.items():
        t = normalize_sec_ticker(str(raw_ticker))
        try:
            out[t] = float(value) * 1e9   # billions -> raw $
        except (TypeError, ValueError):
            continue
    logger.info("Loaded %d market caps from cache (updated %s)",
                len(out), last_updated)
    return out


def _write_cache_files(meta: pd.DataFrame) -> None:
    """Persist universe as the pair of JSONs pms_app uses.

    Kept schema-compatible so the project can be dropped back into pms_app
    or any other consumer without touching format conventions.

        constituents_ndx100.json : {last_updated, data: [{ticker, company, sector}]}
        market_caps_ndx100.json  : {last_updated, tickers_hash, data: {ticker: mcap_B}}
    """
    now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    tickers_sorted = sorted(meta["ticker"].astype(str))
    tickers_hash = hashlib.sha1(",".join(tickers_sorted).encode()).hexdigest()[:16]

    constituents_payload = {
        "last_updated": now_iso,
        "data": [
            {
                "ticker": str(row.ticker),
                "company": str(row.company or ""),
                "sector": str(row.sector or ""),
            }
            for row in meta.itertuples()
        ],
    }
    CACHED_CONSTITUENTS.parent.mkdir(parents=True, exist_ok=True)
    CACHED_CONSTITUENTS.write_text(json.dumps(constituents_payload, indent=2))

    caps_payload = {
        "last_updated": now_iso,
        "tickers_hash": tickers_hash,
        "data": {
            str(row.ticker): float(row.market_cap) / 1e9
            for row in meta.itertuples()
            if row.market_cap and row.market_cap > 0
        },
    }
    CACHED_MARKET_CAPS.write_text(json.dumps(caps_payload, indent=2))
    logger.info(
        "Wrote cache: %s (%d constituents), %s (%d market caps)",
        CACHED_CONSTITUENTS.name, len(constituents_payload["data"]),
        CACHED_MARKET_CAPS.name, len(caps_payload["data"]),
    )


def resolve_ndx_tickers(force_wikipedia: bool = False) -> list[str]:
    """Return canonical NDX 100 tickers.

    Order of preference:
        force_wikipedia=True  -> Wikipedia -> static CSV
        force_wikipedia=False -> fresh cache -> Wikipedia -> static CSV
    """
    if not force_wikipedia:
        cached = _load_cached_constituents()
        if cached is not None and 90 <= len(cached) <= 110:
            return cached["ticker"].tolist()
        logger.info("No usable cache; falling back to Wikipedia")
    try:
        tickers = _fetch_wikipedia()
        logger.info("Fetched %d NDX 100 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Wikipedia fetch failed (%s); using static fallback", exc)
        tickers = _load_static_fallback()
        if not tickers:
            raise RuntimeError(
                "Could not fetch NDX 100 from cache, Wikipedia, or static fallback."
            ) from exc
        return tickers


def _fetch_metadata_yfinance(tickers: list[str]) -> pd.DataFrame:
    """Fallback-only: per-ticker yfinance.info. SLOW (~1s per ticker) — use
    the cached JSONs when possible. Kept for when the cache is missing."""
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
        except Exception as exc:
            logger.warning("info fetch failed for %s: %s", t, exc)
            info = {}
        rows.append({
            "ticker": t,
            "company": info.get("longName") or info.get("shortName") or "",
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "market_cap": float(info.get("marketCap") or 0.0),
        })
    return pd.DataFrame(rows)


def build_universe(refresh: bool = False) -> pd.DataFrame:
    """Resolve NDX 100 tickers + metadata + CIKs. Returns the canonical DataFrame.

    refresh=False (default):
        Try fresh cache first (~instant). If stale/missing, scrape Wikipedia
        + call yfinance `.info` per ticker (~30-60s).
    refresh=True:
        Skip the cache entirely, rebuild from Wikipedia + yfinance, and
        write back the pair of JSON caches for future runs.
    """
    meta: pd.DataFrame | None = None
    if not refresh:
        meta = _load_cached_constituents()

    if meta is not None:
        # Cache hit path — fill market caps from second JSON if available.
        caps = _load_cached_market_caps()
        meta["market_cap"] = meta["ticker"].map(caps).fillna(0.0)
        missing_caps = int((meta["market_cap"] <= 0).sum())
        if missing_caps:
            logger.info(
                "Cached caps missing for %d tickers; will be populated by "
                "fundamentals fetch in stage 2.", missing_caps,
            )
    else:
        # Wikipedia + yfinance path. Forced to Wikipedia on refresh.
        if refresh:
            logger.info(
                "Refresh requested — fetching NDX 100 from Wikipedia and "
                "yfinance.info (slow, expect ~30-60s)."
            )
        tickers = resolve_ndx_tickers(force_wikipedia=refresh)
        meta = _fetch_metadata_yfinance(tickers)

    meta["cik"] = meta["ticker"].map(ticker_to_cik)

    view = build_index_universe_view(
        tickers=meta["ticker"].tolist(),
        raw_market_caps=dict(zip(meta["ticker"], meta["market_cap"])),
    )
    meta["normalized_market_cap"] = meta["ticker"].map(
        view["normalized_market_caps"]
    ).fillna(0.0)
    meta["issuer_group_size"] = meta["ticker"].map(
        view["issuer_group_size_by_ticker"]
    ).fillna(1).astype(int)

    dup_count = sum(1 for size in view["issuer_group_size_by_ticker"].values() if size > 1)
    if dup_count:
        logger.info(
            "Issuer dedup: %d tickers share a CIK with a peer (e.g. GOOG/GOOGL)",
            dup_count,
        )

    # Write back caches on refresh so subsequent non-refresh runs are fast.
    if refresh:
        _write_cache_files(meta)

    return meta


def save_universe(df: pd.DataFrame) -> Path:
    UNIVERSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNIVERSE_PARQUET, index=False)
    return UNIVERSE_PARQUET


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PARQUET.exists():
        raise FileNotFoundError(
            f"No universe file at {UNIVERSE_PARQUET}. Run `python3 -m src.universe` first."
        )
    return pd.read_parquet(UNIVERSE_PARQUET)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NDX 100 universe parquet.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write parquet.")
    parser.add_argument(
        "--refresh-universe", action="store_true",
        help="Ignore cached JSONs; refresh from Wikipedia + yfinance and "
             "rewrite constituents + market_caps JSONs. Use when the cache "
             "is stale (e.g. after an NDX rebalance).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_universe(refresh=args.refresh_universe)
    print(df.head(10).to_string())
    print(f"\nResolved {len(df)} tickers.")
    if not args.dry_run:
        path = save_universe(df)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

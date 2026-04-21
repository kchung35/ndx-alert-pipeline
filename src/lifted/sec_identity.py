"""SEC ticker -> CIK identity lookup.

Lifted from pms_app/engine/sec_identity.py with three edits for hygiene:
  1. BASE_DIR import dropped -> cache path derived from this module's location.
  2. Hard-coded User-Agent replaced by SEC_USER_AGENT env var (required by SEC).
  3. No Streamlit caching -- functools.lru_cache only.

No DB, no portfolio logic. Pure CIK resolution.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SEC_TICKER_CACHE_PATH = _PROJECT_ROOT / "data" / "sec_company_tickers_cache.json"
_SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
_REFRESH_TTL_SECONDS = 86400
_REFRESH_LOCK = threading.Lock()
_LAST_REFRESH_TS = 0.0


def _get_sec_headers() -> dict[str, str]:
    """Build SEC request headers. UA must identify you with an email per SEC rules."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT env var not set. SEC requires a User-Agent of the form "
            "'Name email@domain'. Example: export SEC_USER_AGENT='Kevin Chung kevin@example.com'"
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def normalize_sec_ticker(raw: str) -> str:
    """Normalize ticker to SEC-style symbol form."""
    if not raw:
        return ""
    t = str(raw).strip().upper()
    for suffix in (" US EQUITY", " INDEX", " JP EQUITY", " LN EQUITY", " TT EQUITY"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.replace(".", "-").strip()


def _iter_aliases(ticker: str) -> list[str]:
    """Generate common SEC / Yahoo / BBG alias variants."""
    clean = normalize_sec_ticker(ticker)
    if not clean:
        return []
    aliases: list[str] = []
    for candidate in (
        clean,
        clean.replace("-", ""),
        clean.replace("-", "."),
        clean.replace(".", "-"),
        clean.replace(".", ""),
    ):
        candidate = candidate.strip().upper()
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _rows_to_lookup(data: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    by_ticker: dict[str, dict] = {}
    by_cik: dict[str, dict] = {}
    if not isinstance(data, dict):
        return by_ticker, by_cik

    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        raw_ticker = entry.get("ticker", "")
        cik_raw = entry.get("cik_str")
        ticker = normalize_sec_ticker(raw_ticker)
        if not ticker or cik_raw in (None, ""):
            continue
        cik = str(cik_raw).strip().zfill(10)
        row = {
            "ticker": ticker,
            "cik": cik,
            "title": str(entry.get("title", "") or "").strip(),
            "source": "sec_cache",
        }
        for alias in _iter_aliases(ticker):
            by_ticker.setdefault(alias, row)
        by_cik.setdefault(cik, row)
    return by_ticker, by_cik


@lru_cache(maxsize=1)
def _load_local_lookup() -> tuple[dict[str, dict], dict[str, dict]]:
    try:
        with open(_SEC_TICKER_CACHE_PATH, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        data = {}
    return _rows_to_lookup(data)


def _save_cache(payload: dict) -> None:
    _SEC_TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(_SEC_TICKER_CACHE_PATH) + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, _SEC_TICKER_CACHE_PATH)
    except OSError as exc:
        logger.warning("Could not write SEC ticker cache: %s", exc)


def refresh_sec_ticker_cache(force: bool = False) -> bool:
    """Refresh local SEC ticker cache from SEC if TTL expired."""
    global _LAST_REFRESH_TS
    now = time.time()
    if not force and (now - _LAST_REFRESH_TS) < _REFRESH_TTL_SECONDS:
        return False

    with _REFRESH_LOCK:
        now = time.time()
        if not force and (now - _LAST_REFRESH_TS) < _REFRESH_TTL_SECONDS:
            return False
        try:
            resp = requests.get(_SEC_TICKER_URL, headers=_get_sec_headers(), timeout=20)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("SEC ticker cache refresh failed: %s", exc)
            _LAST_REFRESH_TS = now
            return False

        _save_cache(payload)
        _load_local_lookup.cache_clear()
        _LAST_REFRESH_TS = now
        return True


def resolve_sec_identity(ticker: str, allow_refresh: bool = False) -> dict:
    """Resolve ticker to {input_ticker, ticker, cik, title, source}."""
    input_ticker = normalize_sec_ticker(ticker)
    by_ticker, _ = _load_local_lookup()

    for alias in _iter_aliases(input_ticker):
        row = by_ticker.get(alias)
        if row:
            return {
                "input_ticker": input_ticker,
                "ticker": row["ticker"],
                "cik": row["cik"],
                "title": row["title"],
                "source": row["source"],
            }

    if allow_refresh:
        refresh_sec_ticker_cache(force=False)
        by_ticker, _ = _load_local_lookup()
        for alias in _iter_aliases(input_ticker):
            row = by_ticker.get(alias)
            if row:
                return {
                    "input_ticker": input_ticker,
                    "ticker": row["ticker"],
                    "cik": row["cik"],
                    "title": row["title"],
                    "source": row["source"],
                }

    return {
        "input_ticker": input_ticker,
        "ticker": "",
        "cik": "",
        "title": "",
        "source": "unknown",
    }


def ticker_to_cik(ticker: str, allow_refresh: bool = True) -> str:
    """Return zero-padded 10-digit CIK for ticker, or '' if unknown."""
    return resolve_sec_identity(ticker, allow_refresh=allow_refresh).get("cik", "")


def get_identity_for_cik(cik: str) -> dict:
    clean = str(cik or "").strip().zfill(10)
    _, by_cik = _load_local_lookup()
    row = by_cik.get(clean)
    if not row:
        return {"ticker": "", "cik": clean, "title": "", "source": "unknown"}
    return {
        "ticker": row["ticker"],
        "cik": row["cik"],
        "title": row["title"],
        "source": row["source"],
    }

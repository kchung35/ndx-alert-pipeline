"""Issuer-normalized index universe view.

Lifted from pms_app/engine/index_universe.py. Dropped the multi-index
config (R3000 / S&P 500 / aliases) — this project targets NASDAQ 100
exclusively. The CIK-based issuer deduplication is preserved because
multiple share classes (e.g. GOOGL / GOOG) share a CIK and should not
both appear as independent signals.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Mapping

from .sec_identity import normalize_sec_ticker, resolve_sec_identity


def _freeze_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    clean: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = normalize_sec_ticker(str(raw))
        if not ticker or ticker in seen:
            continue
        clean.append(ticker)
        seen.add(ticker)
    return tuple(clean)


def _freeze_caps(
    tickers: tuple[str, ...],
    raw_market_caps: Mapping[str, float] | None,
) -> tuple[tuple[str, float], ...]:
    caps: list[tuple[str, float]] = []
    source = raw_market_caps or {}
    for ticker in tickers:
        try:
            raw = float(source.get(ticker, 0.0) or 0.0)
        except (TypeError, ValueError):
            raw = 0.0
        caps.append((ticker, raw if raw > 0 else 0.0))
    return tuple(caps)


@lru_cache(maxsize=4)
def _build_view_cached(
    tickers: tuple[str, ...],
    caps_items: tuple[tuple[str, float], ...],
) -> dict:
    raw_market_caps = {ticker: float(cap) for ticker, cap in caps_items}
    issuer_key_by_ticker: dict[str, str] = {}
    issuer_groups: dict[str, list[str]] = defaultdict(list)

    for ticker in tickers:
        identity = resolve_sec_identity(ticker)
        cik = str(identity.get("cik") or "").strip()
        issuer_key = f"cik:{cik}" if cik else f"ticker:{ticker}"
        issuer_key_by_ticker[ticker] = issuer_key
        issuer_groups[issuer_key].append(ticker)

    normalized_market_caps: dict[str, float] = {}
    issuer_group_size_by_ticker: dict[str, int] = {}
    duplicate_groups: dict[str, tuple[str, ...]] = {}

    for issuer_key, members in issuer_groups.items():
        group_size = len(members)
        if group_size > 1:
            duplicate_groups[issuer_key] = tuple(sorted(members))
        for ticker in members:
            issuer_group_size_by_ticker[ticker] = group_size

        positive_caps = {
            t: raw_market_caps.get(t, 0.0)
            for t in members
            if raw_market_caps.get(t, 0.0) > 0
        }
        if group_size > 1 and len(positive_caps) >= 2:
            group_total = max(positive_caps.values())
            cap_sum = sum(positive_caps.values())
            if cap_sum > 0:
                for t in members:
                    raw_cap = positive_caps.get(t, 0.0)
                    normalized_market_caps[t] = group_total * raw_cap / cap_sum
                continue
        for t in members:
            normalized_market_caps[t] = raw_market_caps.get(t, 0.0)

    return {
        "tickers": list(tickers),
        "raw_market_caps": raw_market_caps,
        "normalized_market_caps": normalized_market_caps,
        "issuer_key_by_ticker": issuer_key_by_ticker,
        "issuer_group_size_by_ticker": issuer_group_size_by_ticker,
        "security_count": len(tickers),
        "issuer_count": len(issuer_groups),
        "duplicate_groups": duplicate_groups,
    }


def build_index_universe_view(
    tickers: Iterable[str],
    raw_market_caps: Mapping[str, float] | None = None,
) -> dict:
    """Return an issuer-normalized view with CIK deduplication."""
    clean_tickers = _freeze_tickers(tickers)
    clean_caps = _freeze_caps(clean_tickers, raw_market_caps)
    return _build_view_cached(clean_tickers, clean_caps)

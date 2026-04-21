"""Staleness and refresh tests for src.universe cache handling.

These tests avoid any network calls by monkeypatching the cache file
paths to tmp_path and writing fixture JSON payloads into them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import universe as uni


@pytest.fixture
def cache_paths(tmp_path, monkeypatch):
    """Redirect the two cache-file paths into a temp dir for the test."""
    constituents = tmp_path / "constituents_ndx100.json"
    caps = tmp_path / "market_caps_ndx100.json"
    monkeypatch.setattr(uni, "CACHED_CONSTITUENTS", constituents)
    monkeypatch.setattr(uni, "CACHED_MARKET_CAPS", caps)
    return constituents, caps


def _constituents_payload(last_updated: str, n: int = 100) -> dict:
    return {
        "last_updated": last_updated,
        "data": [
            {"ticker": f"T{i:03d}", "company": f"Co {i}", "sector": "Technology"}
            for i in range(n)
        ],
    }


def _caps_payload(last_updated: str, n: int = 100) -> dict:
    return {
        "last_updated": last_updated,
        "tickers_hash": "0" * 16,
        "data": {f"T{i:03d}": 10.0 + i for i in range(n)},
    }


def test_fresh_cache_is_used(cache_paths):
    constituents, caps = cache_paths
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    constituents.write_text(json.dumps(_constituents_payload(now)))
    caps.write_text(json.dumps(_caps_payload(now)))

    df = uni._load_cached_constituents()
    assert df is not None
    assert len(df) == 100
    assert set(df.columns) >= {"ticker", "company", "sector", "industry"}


def test_stale_cache_is_rejected(cache_paths, caplog):
    constituents, caps = cache_paths
    too_old = (datetime.now(tz=timezone.utc) - timedelta(days=46)).isoformat(timespec="seconds")
    constituents.write_text(json.dumps(_constituents_payload(too_old)))
    caps.write_text(json.dumps(_caps_payload(too_old)))

    with caplog.at_level("WARNING"):
        df = uni._load_cached_constituents()
    assert df is None
    assert any("stale" in rec.message.lower() for rec in caplog.records)


def test_borderline_fresh_cache_accepted(cache_paths):
    """44 days old should still pass the 45-day threshold."""
    constituents, _ = cache_paths
    ts = (datetime.now(tz=timezone.utc) - timedelta(days=44)).isoformat(timespec="seconds")
    constituents.write_text(json.dumps(_constituents_payload(ts)))

    df = uni._load_cached_constituents()
    assert df is not None
    assert len(df) == 100


def test_missing_cache_returns_none(cache_paths):
    # Don't write anything
    assert uni._load_cached_constituents() is None
    assert uni._load_cached_market_caps() == {}


def test_malformed_json_returns_none(cache_paths, caplog):
    constituents, _ = cache_paths
    constituents.write_text("this is not { valid json")
    with caplog.at_level("WARNING"):
        df = uni._load_cached_constituents()
    assert df is None


def test_missing_last_updated_accepted(cache_paths):
    """A payload with no last_updated timestamp is accepted (age undetermined)."""
    constituents, _ = cache_paths
    payload = {"data": _constituents_payload("")["data"]}  # no last_updated
    constituents.write_text(json.dumps(payload))
    df = uni._load_cached_constituents()
    assert df is not None


def test_cache_age_days_parses_iso():
    ten_days_ago = (datetime.now(tz=timezone.utc) - timedelta(days=10)).isoformat(timespec="seconds")
    assert uni._cache_age_days(ten_days_ago) == 10


def test_cache_age_days_handles_garbage():
    assert uni._cache_age_days("not a date") is None
    assert uni._cache_age_days(None) is None
    assert uni._cache_age_days("") is None


def test_write_cache_files_roundtrip(cache_paths):
    constituents, caps = cache_paths
    meta = pd.DataFrame([
        {"ticker": "AAPL", "company": "Apple Inc.", "sector": "Technology",
         "market_cap": 3_200_000_000_000.0},
        {"ticker": "MSFT", "company": "Microsoft", "sector": "Technology",
         "market_cap": 3_000_000_000_000.0},
    ])
    uni._write_cache_files(meta)
    assert constituents.exists() and caps.exists()

    c_payload = json.loads(constituents.read_text())
    assert c_payload["data"][0]["ticker"] == "AAPL"
    assert c_payload["data"][0]["company"] == "Apple Inc."
    assert "last_updated" in c_payload

    cap_payload = json.loads(caps.read_text())
    # Values must be in billions, not raw dollars
    assert abs(cap_payload["data"]["AAPL"] - 3200.0) < 1e-6
    assert len(cap_payload["tickers_hash"]) == 16

    # And the freshly-written cache should round-trip through the loader
    roundtrip = uni._load_cached_constituents()
    assert roundtrip is not None
    assert set(roundtrip["ticker"]) == {"AAPL", "MSFT"}

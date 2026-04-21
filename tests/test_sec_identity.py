"""Unit tests for SEC identity normalization (no network)."""

from __future__ import annotations

from src.lifted.sec_identity import _iter_aliases, normalize_sec_ticker


def test_normalize_strips_bloomberg_suffixes():
    assert normalize_sec_ticker("AAPL US Equity") == "AAPL"
    assert normalize_sec_ticker("7203 JP Equity") == "7203"
    assert normalize_sec_ticker("^NDX Index") == "^NDX"


def test_normalize_converts_dots_to_dashes():
    assert normalize_sec_ticker("BRK.B") == "BRK-B"


def test_normalize_uppercases_and_strips_whitespace():
    assert normalize_sec_ticker("  goog  ") == "GOOG"


def test_aliases_include_variants():
    aliases = _iter_aliases("BRK-B")
    assert "BRK-B" in aliases
    assert "BRKB" in aliases
    assert "BRK.B" in aliases


def test_empty_ticker_returns_empty():
    assert normalize_sec_ticker("") == ""
    assert _iter_aliases("") == []

"""Unit tests for the composite alert engine (including MOMENTUM_LONG
and CONFLUENCE_* tiers added after the 101-ticker run showed the
original composite thresholds were unreachable in realistic data)."""

from __future__ import annotations

import pandas as pd

from src.alert_engine import (
    BULL_THRESH, TIER_BEARISH, TIER_BULLISH, TIER_CONFLUENCE_BEAR,
    TIER_CONFLUENCE_BULL, TIER_MOMENTUM_LONG, TIER_NONE, _rationale, _tier_of,
)


def _row(factor_z: float, options_z: float, insider_z: float,
         *, momentum_long: bool = False,
         momentum_3m_z: float = 0.0) -> pd.Series:
    composite = 0.4 * factor_z + 0.3 * options_z + 0.3 * insider_z
    return pd.Series({
        "factor_z": factor_z, "options_z": options_z, "insider_z": insider_z,
        "composite": composite,
        "momentum_long": momentum_long,
        "momentum_3m_z": momentum_3m_z,
    })


# ── Strong tier (unchanged, original behavior must be preserved) ─────

def test_strong_bullish_triggers():
    assert _tier_of(_row(2.0, 1.5, 1.2)) == TIER_BULLISH


def test_strong_bearish_triggers():
    assert _tier_of(_row(-2.0, -1.5, -1.2)) == TIER_BEARISH


def test_strong_disagreement_vetos_confluence_bull():
    # Two components strongly bullish, one strongly bearish -> the
    # disagreement veto must kill the alert (conflicting signals).
    assert _tier_of(_row(2.5, 1.5, -1.0)) == TIER_NONE


def test_two_of_three_confluence_with_neutral_third_fires():
    # Two strong bulls + one neutral (no disagreement). Composite
    # crosses 1.0, 2-of-3 rule satisfied, no veto -> CONFLUENCE_BULL.
    row = _row(2.0, 2.0, 0.1)
    assert _tier_of(row) == TIER_CONFLUENCE_BULL


# ── Momentum tier (new, primary tradeable signal) ────────────────────

def test_momentum_long_triggers_regardless_of_composite():
    # Composite is barely positive but ticker is in top-decile 3m momentum
    row = _row(0.1, 0.1, 0.0, momentum_long=True, momentum_3m_z=2.4)
    assert _tier_of(row) == TIER_MOMENTUM_LONG


def test_strong_bullish_takes_precedence_over_momentum_long():
    # Ticker has BOTH strong composite AND top-decile momentum
    # → STRONG_BULLISH wins (richer signal = stronger label)
    row = _row(2.0, 1.5, 1.2, momentum_long=True, momentum_3m_z=2.0)
    assert _tier_of(row) == TIER_BULLISH


def test_momentum_long_outranks_confluence():
    # Composite hits CONFLUENCE threshold + momentum_long → momentum wins
    row = _row(1.5, 0.3, 0.2, momentum_long=True, momentum_3m_z=2.5)
    assert _tier_of(row) == TIER_MOMENTUM_LONG


# ── Confluence tier (relaxed, new) ──────────────────────────────────

def test_confluence_bullish_two_of_three_triggers():
    # composite = 0.4*0.8 + 0.3*0.5 + 0.3*1.2 = 0.32+0.15+0.36 = 0.83 (<1.0)
    # so above fails; need composite > 1.0
    row = _row(1.5, 0.5, 1.2)
    # composite = 0.6 + 0.15 + 0.36 = 1.11 -- over 1.0
    # components: factor 1.5 (>.15), options 0.5 (>.15), insider 1.2 (>.15) → 3 of 3
    assert _tier_of(row) == TIER_CONFLUENCE_BULL


def test_confluence_bullish_only_one_component_over_floor_is_no_alert():
    # composite = 0.4*3.0 + 0.3*0 + 0.3*0 = 1.2 > 1.0 but only 1 component over floor
    row = _row(3.0, 0.0, 0.0)
    assert _tier_of(row) == TIER_NONE


def test_confluence_bearish_triggers_symmetric():
    row = _row(-1.5, -0.5, -1.2)
    assert _tier_of(row) == TIER_CONFLUENCE_BEAR


def test_below_confluence_composite_is_no_alert():
    # composite = 0.4*1.0+0.3*1.0+0.3*1.0 = 1.0 (at edge, strict > so fails)
    row = _row(1.0, 1.0, 1.0)
    assert _tier_of(row) == TIER_NONE


# ── Rationale ───────────────────────────────────────────────────────

def test_rationale_highlights_strongest_components():
    row = _row(2.0, 0.1, 1.0)
    txt = _rationale(row)
    assert "factor=+2.00" in txt
    assert "insider=+1.00" in txt
    assert "options" not in txt


def test_rationale_surfaces_momentum_when_flagged():
    row = _row(0.1, 0.1, 0.0, momentum_long=True, momentum_3m_z=2.45)
    txt = _rationale(row)
    assert "3m-mom" in txt
    assert "2.45" in txt

"""Unit tests for lifted insider utilities."""

from __future__ import annotations

import pandas as pd

from src.lifted.insider_utils import (
    classify_transaction, is_corporate_entity, is_corporate_entity_series,
    officer_weight,
)


def test_classify_open_market_purchase_is_bullish():
    label, weight = classify_transaction("P")
    assert weight > 0
    assert "Purchase" in label


def test_classify_10b5_1_sale_downgraded():
    label, weight = classify_transaction("S", is_10b5_1=True)
    assert weight == -1
    assert "10b5-1" in label


def test_classify_derivative_award_neutralized():
    _, weight = classify_transaction("A", is_derivative=True)
    assert weight == 0


def test_classify_unknown_code():
    label, weight = classify_transaction("QQQ")
    assert weight == 0
    assert label == "Other"


def test_officer_weight_ceo_beats_director():
    assert officer_weight("CEO and President") > officer_weight("Director")


def test_officer_weight_empty_returns_one():
    assert officer_weight("") == 1
    assert officer_weight(None) == 1


def test_corporate_entity_pure_corp():
    assert is_corporate_entity("GENERAL ELECTRIC CO") is True
    assert is_corporate_entity("APPLE INC.") is True


def test_corporate_entity_with_real_person_kept():
    assert is_corporate_entity("BERKSHIRE HATHAWAY INC / Warren E Buffett") is False


def test_corporate_entity_plain_person_not_corp():
    assert is_corporate_entity("Tim Cook") is False
    assert is_corporate_entity("Jane Q Public") is False


def test_corporate_entity_series_vectorized():
    s = pd.Series([
        "GENERAL ELECTRIC CO",
        "BERKSHIRE HATHAWAY INC / Warren E Buffett",
        "Tim Cook",
        "",
    ])
    result = is_corporate_entity_series(s)
    assert result.tolist() == [True, False, False, False]

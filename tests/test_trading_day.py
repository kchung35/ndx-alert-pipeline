"""Verify the trading-day helper handles weekends, holidays, cutoff time."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.trading_day import (
    is_trading_day, last_completed_trading_day, previous_trading_day,
)

NY = ZoneInfo("America/New_York")


def test_weekday_is_trading_day():
    assert is_trading_day(date(2026, 4, 22)) is True  # Wed
    assert is_trading_day(date(2026, 4, 24)) is True  # Fri


def test_weekend_is_not_trading_day():
    assert is_trading_day(date(2026, 4, 25)) is False  # Sat
    assert is_trading_day(date(2026, 4, 26)) is False  # Sun


def test_holiday_is_not_trading_day():
    # Good Friday 2026
    assert is_trading_day(date(2026, 4, 3)) is False
    # Christmas 2026
    assert is_trading_day(date(2026, 12, 25)) is False


def test_previous_trading_day_skips_weekend():
    # Monday 2026-04-20 -> previous trading day should be Friday 2026-04-17
    assert previous_trading_day(date(2026, 4, 20)) == date(2026, 4, 17)


def test_previous_trading_day_skips_holiday():
    # Mon 2026-04-06: 04-03 is Good Friday; walk to Thu 04-02
    assert previous_trading_day(date(2026, 4, 6)) == date(2026, 4, 2)


def test_last_completed_trading_day_before_cutoff_returns_prior():
    # Tuesday 2026-04-21 at 11:00 ET -> use 2026-04-20 (Monday)
    now = datetime(2026, 4, 21, 11, 0, tzinfo=NY)
    assert last_completed_trading_day(now) == date(2026, 4, 20)


def test_last_completed_trading_day_after_cutoff_returns_today():
    # Tuesday 2026-04-21 at 20:00 ET -> use 2026-04-21 (past 5pm ET)
    now = datetime(2026, 4, 21, 20, 0, tzinfo=NY)
    assert last_completed_trading_day(now) == date(2026, 4, 21)


def test_saturday_morning_returns_friday():
    now = datetime(2026, 4, 25, 9, 0, tzinfo=NY)
    assert last_completed_trading_day(now) == date(2026, 4, 24)


def test_monday_morning_after_holiday_walks_back():
    # Mon 2026-04-06 09:00 ET -- Fri 04-03 was Good Friday -> Thu 04-02
    now = datetime(2026, 4, 6, 9, 0, tzinfo=NY)
    assert last_completed_trading_day(now) == date(2026, 4, 2)

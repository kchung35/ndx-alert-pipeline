"""Trading-day helpers — pick the correct as-of date for the daily pipeline.

The key question: when we run at time `now`, which trading day should the
pipeline treat as "today" for signal computation?

Rules:
    - If `now` is on a business day after the EOD buffer (default 17:00 ET),
      use today's date.
    - Otherwise, walk back to the most recent completed business day.
    - Weekends always resolve to Friday.

We intentionally do NOT use pandas_market_calendars (would pin us to a
specific calendar vendor). NYSE holidays are hard-coded for 2024-2027,
which covers the horizon of the current backtest scope. A warning is
logged if the pipeline is run past this horizon so we extend it.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

# NYSE holidays for 2024-2027. Update this table yearly.
# Source: NYSE holiday calendar (full closures only; early-close days treated
# as normal for our EOD signal purposes).
_NYSE_HOLIDAYS: frozenset[date] = frozenset([
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
])

_NY_TZ = ZoneInfo("America/New_York")
_EOD_CUTOFF = time(17, 0)  # 5pm ET -- 1h after close for data to settle


def is_trading_day(d: date) -> bool:
    """True iff d is a weekday and not in the NYSE holiday set."""
    if d.weekday() >= 5:
        return False
    return d not in _NYSE_HOLIDAYS


def previous_trading_day(d: date) -> date:
    """Most recent trading day strictly before d."""
    cur = d
    while True:
        cur = cur - pd.Timedelta(days=1).to_pytimedelta()
        if is_trading_day(cur):
            return cur


def last_completed_trading_day(now: datetime | None = None) -> date:
    """The most recent trading day whose EOD has passed.

    If now is on a trading day AND after _EOD_CUTOFF (17:00 ET), return today.
    Otherwise walk back to the previous trading day.
    """
    if now is None:
        now = datetime.now(tz=_NY_TZ)
    elif now.tzinfo is None:
        # Treat naive datetimes as already-in-NY-time (caller's responsibility)
        now = now.replace(tzinfo=_NY_TZ)
    else:
        now = now.astimezone(_NY_TZ)

    today = now.date()
    if is_trading_day(today) and now.time() >= _EOD_CUTOFF:
        return today
    return previous_trading_day(today)


def warn_if_horizon_exhausted(d: date) -> None:
    """Emit a warning if d is past the hard-coded holiday horizon."""
    max_known = max(_NYSE_HOLIDAYS)
    if d > max_known:
        logger.warning(
            "trading_day helpers only know holidays through %s; extend "
            "_NYSE_HOLIDAYS in src/trading_day.py", max_known,
        )

"""Performance/Risk support helpers shared by Streamlit and static exports."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.universe import PROJECT_ROOT

MIN_OPTIONS_HISTORY_DAYS = 20
MIN_OPTIONS_CHAIN_COVERAGE = 0.80

OPTIONS_REPLAY_CAVEAT = (
    "Current-book risk only. Full historical alert replay is gated on daily "
    "options snapshot history; SEC Form 4 insider history can be rebuilt from EDGAR."
)


def _snapshot_date(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.name)
    except ValueError:
        return None


def _universe_count(project_root: Path) -> int:
    path = project_root / "data" / "universe.parquet"
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0


def options_history_validation_gate(
    as_of: date,
    *,
    project_root: Path = PROJECT_ROOT,
    min_days: int = MIN_OPTIONS_HISTORY_DAYS,
    min_coverage: float = MIN_OPTIONS_CHAIN_COVERAGE,
) -> dict:
    """Validation gate for enabling a full historical alert replay.

    A proper alert replay needs daily option-chain snapshots because Yahoo only
    gives the live chain state. Insider history is not the blocker here: Form 4
    history is public and the cache can be rebuilt from EDGAR.
    """
    chains_root = project_root / "data" / "chains"
    universe_n = _universe_count(project_root)
    rows: list[dict] = []

    if chains_root.exists():
        for child in sorted(chains_root.iterdir()):
            snap_date = _snapshot_date(child)
            if snap_date is None or snap_date > as_of or not child.is_dir():
                continue
            chain_files = len(list(child.glob("*.parquet")))
            coverage = chain_files / universe_n if universe_n else 0.0
            rows.append({
                "date": snap_date.isoformat(),
                "chain_files": int(chain_files),
                "coverage": float(coverage),
                "usable": bool(coverage >= min_coverage),
            })

    usable = [row for row in rows if row["usable"]]
    latest = rows[-1]["date"] if rows else None
    latest_current = latest == as_of.isoformat()
    passed = len(usable) >= min_days and latest_current
    missing_days = max(0, min_days - len(usable))

    if passed:
        message = (
            f"Passed: {len(usable)} usable option snapshot days meet the "
            f"{min_coverage:.0%} chain-coverage gate."
        )
    elif not rows:
        message = "Blocked: no option snapshot archive found."
    elif not latest_current:
        message = (
            f"Blocked: latest option snapshot is {latest}; expected {as_of.isoformat()}."
        )
    else:
        message = (
            f"Blocked: {len(usable)} usable option snapshot days; need {min_days} "
            "before a full historical alert replay is shown."
        )

    return {
        "status": "PASS" if passed else "BLOCKED",
        "passed": bool(passed),
        "message": message,
        "caveat": OPTIONS_REPLAY_CAVEAT,
        "min_days": int(min_days),
        "min_coverage": float(min_coverage),
        "snapshot_days": int(len(rows)),
        "usable_days": int(len(usable)),
        "missing_days": int(missing_days),
        "latest_snapshot_date": latest,
        "latest_is_current": bool(latest_current),
        "universe_count": int(universe_n),
        "recent_days": rows[-5:],
    }

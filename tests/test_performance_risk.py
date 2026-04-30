"""Tests for Performance & Risk validation gates."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.performance_risk import options_history_validation_gate


def _write_universe(root: Path, rows: int = 100) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(rows)]}).to_parquet(
        data_dir / "universe.parquet",
        index=False,
    )


def _write_chain_files(root: Path, day: date, count: int) -> None:
    day_dir = root / "data" / "chains" / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (day_dir / f"T{i:03d}.parquet").write_bytes(b"stub")


def test_options_history_gate_blocks_when_archive_is_too_short(tmp_path):
    _write_universe(tmp_path, rows=100)
    _write_chain_files(tmp_path, date(2026, 4, 21), 100)
    _write_chain_files(tmp_path, date(2026, 4, 29), 100)

    gate = options_history_validation_gate(date(2026, 4, 29), project_root=tmp_path)

    assert gate["status"] == "BLOCKED"
    assert gate["usable_days"] == 2
    assert gate["missing_days"] == 18
    assert "options snapshot history" in gate["caveat"]
    assert "EDGAR" in gate["caveat"]


def test_options_history_gate_passes_with_enough_current_covered_days(tmp_path):
    _write_universe(tmp_path, rows=100)
    start = date(2026, 4, 1)
    for i in range(20):
        _write_chain_files(tmp_path, start + timedelta(days=i), 85)

    gate = options_history_validation_gate(
        start + timedelta(days=19),
        project_root=tmp_path,
        min_days=20,
        min_coverage=0.80,
    )

    assert gate["status"] == "PASS"
    assert gate["usable_days"] == 20
    assert gate["latest_is_current"] is True


def test_options_history_gate_requires_current_snapshot(tmp_path):
    _write_universe(tmp_path, rows=100)
    start = date(2026, 4, 1)
    for i in range(20):
        _write_chain_files(tmp_path, start + timedelta(days=i), 100)

    gate = options_history_validation_gate(date(2026, 4, 30), project_root=tmp_path)

    assert gate["status"] == "BLOCKED"
    assert gate["latest_snapshot_date"] == "2026-04-20"
    assert gate["latest_is_current"] is False

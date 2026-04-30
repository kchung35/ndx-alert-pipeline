from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.verify_snapshot import SnapshotVerificationError, verify_snapshot


def _write_parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": list(range(rows))}).to_parquet(path, index=False)


def _make_snapshot(tmp_path: Path, *, alerts_rows: int = 2) -> Path:
    as_of = "2026-04-21"
    required = [
        "data/alerts/{date}.parquet",
        "data/factors/{date}.parquet",
        "data/options_signals/{date}.parquet",
        "data/insider_signals/{date}.parquet",
        "data/prices.parquet",
        "data/fundamentals.parquet",
        "data/vix.parquet",
        "data/ff.parquet",
        "data/universe.parquet",
        "data/chains/{date}",
        "data/form4",
        "NDX Alert Desk.html",
        "scripts/export_newsletter.py",
        "src/newsletter_export.py",
    ]
    counts = {
        "alerts_rows": 2,
        "factors_rows": 2,
        "options_signals_rows": 2,
        "insider_signals_rows": 2,
        "prices_rows": 3,
        "fundamentals_rows": 2,
        "vix_rows": 1,
        "ff_rows": 1,
        "universe_rows": 2,
        "chain_files": 2,
        "form4_files": 1,
        "form4_rows": 2,
        "static_html_min_bytes": 10,
    }
    manifest = {
        "snapshot_date": as_of,
        "counts": counts,
        "required_files": required,
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "snapshot_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _write_parquet(tmp_path / f"data/alerts/{as_of}.parquet", alerts_rows)
    _write_parquet(tmp_path / f"data/factors/{as_of}.parquet", 2)
    _write_parquet(tmp_path / f"data/options_signals/{as_of}.parquet", 2)
    _write_parquet(tmp_path / f"data/insider_signals/{as_of}.parquet", 2)
    _write_parquet(tmp_path / "data/prices.parquet", 3)
    _write_parquet(tmp_path / "data/fundamentals.parquet", 2)
    _write_parquet(tmp_path / "data/vix.parquet", 1)
    _write_parquet(tmp_path / "data/ff.parquet", 1)
    _write_parquet(tmp_path / "data/universe.parquet", 2)
    _write_parquet(tmp_path / f"data/chains/{as_of}/AAA.parquet", 1)
    _write_parquet(tmp_path / f"data/chains/{as_of}/BBB.parquet", 1)
    _write_parquet(tmp_path / "data/form4/AAA.parquet", 2)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/export_newsletter.py").write_text("# placeholder\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/newsletter_export.py").write_text("# placeholder\n", encoding="utf-8")
    (tmp_path / "NDX Alert Desk.html").write_text(
        "<html><script>window.DATA = {getPerformanceRisk(){}}</script>"
        "<body>04 Export newsletter 05 Performance & Risk "
        "Historical replay validation gate</body></html>",
        encoding="utf-8",
    )
    return tmp_path


def test_verify_snapshot_passes_complete_manifest(tmp_path):
    root = _make_snapshot(tmp_path)
    messages = verify_snapshot("2026-04-21", project_root=root)
    assert "alerts rows: 2" in messages
    assert any(msg.startswith("static HTML bytes:") for msg in messages)


def test_verify_snapshot_fails_missing_required_file(tmp_path):
    root = _make_snapshot(tmp_path)
    (root / "data/vix.parquet").unlink()
    with pytest.raises(SnapshotVerificationError, match="Missing required file: data/vix.parquet"):
        verify_snapshot("2026-04-21", project_root=root)


def test_verify_snapshot_fails_count_mismatch(tmp_path):
    root = _make_snapshot(tmp_path, alerts_rows=1)
    with pytest.raises(SnapshotVerificationError, match="alerts rows mismatch"):
        verify_snapshot("2026-04-21", project_root=root)

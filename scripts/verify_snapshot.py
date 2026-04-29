"""Verify that a committed NDX Alert Desk snapshot is reproducible.

This is a local integrity check for the baked demo/cache data. It does not
fetch network data.

Usage:
    python3 scripts/verify_snapshot.py --date 2026-04-21
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class SnapshotVerificationError(RuntimeError):
    """Raised when a committed snapshot is incomplete or inconsistent."""


def _read_manifest(project_root: Path) -> dict:
    path = project_root / "data" / "snapshot_manifest.json"
    if not path.exists():
        raise SnapshotVerificationError(f"Missing snapshot manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(project_root: Path, rel_path: str) -> Path:
    path = project_root / rel_path
    if not path.exists():
        raise SnapshotVerificationError(f"Missing required file: {rel_path}")
    if path.is_file() and path.stat().st_size == 0:
        raise SnapshotVerificationError(f"Required file is empty: {rel_path}")
    return path


def _parquet_rows(project_root: Path, rel_path: str) -> int:
    return int(len(pd.read_parquet(_require_file(project_root, rel_path))))


def _expect(name: str, actual: int, expected: int) -> str:
    if actual != expected:
        raise SnapshotVerificationError(
            f"{name} mismatch: expected {expected}, got {actual}"
        )
    return f"{name}: {actual}"


def verify_snapshot(
    as_of: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    manifest = _read_manifest(project_root)
    if as_of != manifest.get("snapshot_date"):
        raise SnapshotVerificationError(
            f"Manifest snapshot_date is {manifest.get('snapshot_date')!r}, not {as_of!r}"
        )

    for rel_path in manifest.get("required_files", []):
        _require_file(project_root, rel_path.format(date=as_of))

    counts = manifest.get("counts", {})
    messages = [
        _expect("alerts rows", _parquet_rows(project_root, f"data/alerts/{as_of}.parquet"), counts["alerts_rows"]),
        _expect("factor rows", _parquet_rows(project_root, f"data/factors/{as_of}.parquet"), counts["factors_rows"]),
        _expect("options rows", _parquet_rows(project_root, f"data/options_signals/{as_of}.parquet"), counts["options_signals_rows"]),
        _expect("insider signal rows", _parquet_rows(project_root, f"data/insider_signals/{as_of}.parquet"), counts["insider_signals_rows"]),
        _expect("prices rows", _parquet_rows(project_root, "data/prices.parquet"), counts["prices_rows"]),
        _expect("fundamentals rows", _parquet_rows(project_root, "data/fundamentals.parquet"), counts["fundamentals_rows"]),
        _expect("vix rows", _parquet_rows(project_root, "data/vix.parquet"), counts["vix_rows"]),
        _expect("ff rows", _parquet_rows(project_root, "data/ff.parquet"), counts["ff_rows"]),
        _expect("universe rows", _parquet_rows(project_root, "data/universe.parquet"), counts["universe_rows"]),
    ]

    chain_count = len(list((project_root / "data" / "chains" / as_of).glob("*.parquet")))
    messages.append(_expect("chain files", chain_count, counts["chain_files"]))

    form4_files = list((project_root / "data" / "form4").glob("*.parquet"))
    messages.append(_expect("Form 4 cache files", len(form4_files), counts["form4_files"]))
    form4_rows = sum(int(len(pd.read_parquet(path))) for path in form4_files)
    messages.append(_expect("Form 4 rows", form4_rows, counts["form4_rows"]))

    html_path = _require_file(project_root, "NDX Alert Desk.html")
    html_text = html_path.read_text(encoding="utf-8")
    if len(html_text.encode("utf-8")) < counts["static_html_min_bytes"]:
        raise SnapshotVerificationError(
            "Static HTML is smaller than expected; baked data may be missing"
        )
    for needle in ("window.DATA", "04", "Export newsletter"):
        if needle not in html_text:
            raise SnapshotVerificationError(f"Static HTML missing marker: {needle}")
    messages.append(f"static HTML bytes: {html_path.stat().st_size}")

    # Import check for the email export surface without creating artifacts.
    from src import newsletter_export  # noqa: F401

    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                        help="Snapshot date to verify (default: manifest date).")
    args = parser.parse_args()

    manifest = _read_manifest(PROJECT_ROOT)
    as_of = args.date or manifest["snapshot_date"]
    try:
        messages = verify_snapshot(as_of, project_root=PROJECT_ROOT)
    except SnapshotVerificationError as exc:
        print(f"SNAPSHOT VERIFY FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Snapshot {as_of} verified.")
    for msg in messages:
        print(f"  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

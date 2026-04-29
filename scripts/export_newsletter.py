"""Export a timestamped NDX Alert Desk newsletter package.

Examples:
    python3 scripts/export_newsletter.py
    python3 scripts/export_newsletter.py --date 2026-04-21
    python3 scripts/export_newsletter.py --date 2026-04-21 --no-png
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.newsletter_export import export_newsletter  # noqa: E402


def _latest_available_date() -> date:
    alerts_dir = PROJECT_ROOT / "data" / "alerts"
    dates = []
    for p in alerts_dir.glob("*.parquet"):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            continue
    if not dates:
        raise SystemExit(f"No alert parquet files found in {alerts_dir}")
    return max(dates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                        help="As-of date (default: latest alerts parquet).")
    parser.add_argument("--output-root", default=None,
                        help="Export root (default: exports/newsletters).")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip browser PNG rendering.")
    args = parser.parse_args()

    as_of = date.fromisoformat(args.date) if args.date else _latest_available_date()
    output_root = Path(args.output_root) if args.output_root else None
    artifacts = export_newsletter(
        as_of,
        project_root=PROJECT_ROOT,
        output_root=output_root,
        render_png=not args.no_png,
    )

    print(f"Exported newsletter package: {artifacts.output_dir}")
    print(f"  subject: {artifacts.subject}")
    print(f"  eml: {artifacts.eml}")
    print(f"  html: {artifacts.newsletter_html}")
    print(f"  text: {artifacts.newsletter_text}")
    if artifacts.dashboard_html:
        print(f"  dashboard html: {artifacts.dashboard_html}")
    print(f"  png: {artifacts.png_status}")
    if artifacts.dashboard_png:
        print(f"  png path: {artifacts.dashboard_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

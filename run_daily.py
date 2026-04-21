"""Daily orchestrator — pulls data, computes signals, writes alerts.

Idempotent by --date: re-running for the same date overwrites parquet
outputs deterministically.

Usage:
    python3 run_daily.py                        # today
    python3 run_daily.py --date 2026-04-21      # specific day
    python3 run_daily.py --skip-options         # factor + insider only

SEC requires SEC_USER_AGENT env var for Form 4 pulls. Example:
    export SEC_USER_AGENT='Kevin Chung kevin@example.com'
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    from src.trading_day import last_completed_trading_day, warn_if_horizon_exhausted
    default_date = last_completed_trading_day().isoformat()
    parser = argparse.ArgumentParser(description="Daily NDX 100 alert pipeline.")
    parser.add_argument("--date", default=default_date,
                        help=f"as-of date (default {default_date} = last completed trading day)")
    parser.add_argument("--skip-universe", action="store_true")
    parser.add_argument(
        "--refresh-universe", action="store_true",
        help="Force Wikipedia + yfinance refresh of NDX constituents "
             "(overwrites cache JSONs). Run after NDX rebalances.",
    )
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-options", action="store_true")
    parser.add_argument("--skip-edgar", action="store_true")
    parser.add_argument("--skip-ff", action="store_true")
    parser.add_argument("--edgar-lookback-days", type=int, default=365)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_daily")
    as_of = date.fromisoformat(args.date)
    warn_if_horizon_exhausted(as_of)
    log.info("Pipeline as-of date: %s", as_of)

    # Lazy imports so a broken module doesn't kill --help.
    from src.alert_engine import build_alerts, save_alerts
    from src.data_edgar import main as edgar_main  # noqa: F401
    from src.data_ff import main as ff_main  # noqa: F401
    from src.data_options import main as options_main  # noqa: F401
    from src.data_prices import main as prices_main  # noqa: F401
    from src.factors import compute_factor_panel, save_factors
    from src.insider_signals import compute_insider_panel, save_insider_panel
    from src.options_signals import compute_options_panel, save_options_panel
    from src.universe import build_universe, save_universe

    # 1. Universe (Mondays + first run only; --refresh-universe forces Wikipedia pull)
    if not args.skip_universe:
        if args.refresh_universe:
            log.info("[1/7] Refreshing NDX 100 universe (Wikipedia + yfinance)")
        else:
            log.info("[1/7] Resolving NDX 100 universe")
        save_universe(build_universe(refresh=args.refresh_universe))

    # 2. Prices + fundamentals + VIX
    if not args.skip_prices:
        log.info("[2/7] Pulling prices / fundamentals / VIX")
        from src.data_prices import (
            fetch_fundamentals, fetch_prices, fetch_vix,
            save_fundamentals, save_prices, save_vix,
        )
        from src.universe import load_universe
        tickers = load_universe()["ticker"].tolist()
        save_prices(fetch_prices(tickers))
        save_fundamentals(fetch_fundamentals(tickers))
        save_vix(fetch_vix())

    # 3. FF factors
    if not args.skip_ff:
        log.info("[3/7] Fetching Fama-French factors")
        from src.data_ff import fetch_ff_daily, save_ff
        try:
            save_ff(fetch_ff_daily())
        except Exception as exc:
            log.warning("FF fetch failed (non-fatal): %s", exc)

    # 4. Options chain snapshot
    if not args.skip_options:
        log.info("[4/7] Snapshotting options chains")
        from src.data_options import fetch_chain, save_chain
        from src.universe import load_universe
        tickers = load_universe()["ticker"].tolist()
        for t in tickers:
            try:
                ch = fetch_chain(t, as_of)
                if not ch.empty:
                    save_chain(t, as_of, ch)
            except Exception as exc:
                log.warning("options %s: %s", t, exc)

    # 5. EDGAR Form 4
    if not args.skip_edgar:
        log.info("[5/7] Pulling Form 4 filings from EDGAR")
        if not os.environ.get("SEC_USER_AGENT"):
            log.warning(
                "SEC_USER_AGENT not set -- skipping EDGAR. "
                "Set it to 'Name email@domain' and re-run --skip-universe --skip-prices --skip-options --skip-ff."
            )
        else:
            from src.data_edgar import fetch_form4_for_ticker, save_form4
            from src.universe import load_universe
            tickers = load_universe()["ticker"].tolist()
            for t in tickers:
                try:
                    df = fetch_form4_for_ticker(t, lookback_days=args.edgar_lookback_days)
                    if not df.empty:
                        save_form4(t, df)
                except Exception as exc:
                    log.warning("form4 %s: %s", t, exc)

    # 6. Compute signals
    log.info("[6/7] Computing factor / options / insider panels")
    factor_df = compute_factor_panel(as_of)
    save_factors(factor_df, as_of)
    try:
        options_df = compute_options_panel(as_of)
        save_options_panel(options_df, as_of)
    except FileNotFoundError as exc:
        log.warning("options signals skipped: %s", exc)
    try:
        insider_df = compute_insider_panel(as_of)
        save_insider_panel(insider_df, as_of)
    except Exception as exc:
        log.warning("insider signals skipped: %s", exc)

    # 7. Composite alerts
    log.info("[7/7] Building composite alerts")
    alerts = build_alerts(as_of)
    path = save_alerts(alerts, as_of)
    tiers = alerts["tier"].value_counts().to_dict()
    log.info("Alerts written to %s   tiers=%s", path, tiers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

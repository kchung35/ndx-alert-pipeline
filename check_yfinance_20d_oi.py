"""
Check whether yfinance provides 20-day options open interest history.

This script uses two complementary approaches:

1. Static inspection of the installed yfinance source code:
   - confirms what `Ticker.option_chain()` is designed to return
   - confirms whether `openInterest` is part of the option-chain snapshot schema

2. Live Yahoo pull (if network access is available):
   - fetches a current option chain for a stock
   - selects one contract symbol from that chain
   - asks yfinance for 20 days of history on that *option contract ticker*
   - checks whether the returned history contains an `Open Interest` column

Expected result:
    - Current snapshot OI: usually YES
    - 20-day historical OI via yfinance alone: usually NO

Run examples:
    python3 Project/check_yfinance_20d_oi.py
    python3 Project/check_yfinance_20d_oi.py --symbol MSFT
"""

from __future__ import annotations

import argparse
import inspect
import sys
import textwrap
from dataclasses import dataclass

import yfinance as yf


@dataclass
class Finding:
    name: str
    ok: bool
    detail: str


def emit(finding: Finding) -> None:
    status = "PASS" if finding.ok else "FAIL"
    print(f"[{status}] {finding.name:<34} {finding.detail}")


def static_source_checks() -> list[Finding]:
    findings: list[Finding] = []

    ticker_cls = yf.Ticker
    source_option_chain = inspect.getsource(ticker_cls.option_chain)
    source_download = inspect.getsource(ticker_cls._download_options)
    source_options2df = inspect.getsource(ticker_cls._options2df)

    findings.append(
        Finding(
            "yfinance version",
            True,
            yf.__version__,
        )
    )

    findings.append(
        Finding(
            "option_chain uses options endpoint",
            "/v7/finance/options/" in source_download,
            "Ticker._download_options() hits Yahoo's options endpoint",
        )
    )

    findings.append(
        Finding(
            "snapshot schema includes openInterest",
            "'openInterest'" in source_options2df or '"openInterest"' in source_options2df,
            "Ticker._options2df() includes an openInterest column in the option chain",
        )
    )

    findings.append(
        Finding(
            "option_chain has no history parameter",
            "history" not in source_option_chain.lower(),
            "Installed source shows option_chain(date=None, tz=None) only returns one snapshot",
        )
    )

    return findings


def live_checks(symbol: str) -> list[Finding]:
    findings: list[Finding] = []
    tk = yf.Ticker(symbol)

    try:
        expiries = tk.options
        findings.append(
            Finding(
                "live expiry list",
                len(expiries) > 0,
                f"{len(expiries)} expiries found" if expiries else "No expiries returned",
            )
        )
    except Exception as exc:
        findings.append(
            Finding(
                "live expiry list",
                False,
                f"Could not reach Yahoo options endpoint: {type(exc).__name__}: {exc}",
            )
        )
        return findings

    if not expiries:
        return findings

    expiry = expiries[0]

    try:
        chain = tk.option_chain(expiry)
        calls = chain.calls
        puts = chain.puts
        has_snapshot_oi = (
            calls is not None
            and puts is not None
            and "openInterest" in calls.columns
            and "openInterest" in puts.columns
        )
        findings.append(
            Finding(
                "current chain snapshot OI",
                has_snapshot_oi,
                (
                    f"expiry={expiry}, call rows={len(calls)}, put rows={len(puts)}"
                    if calls is not None and puts is not None
                    else f"expiry={expiry}, chain was empty"
                ),
            )
        )
    except Exception as exc:
        findings.append(
            Finding(
                "current chain snapshot OI",
                False,
                f"Could not fetch current option chain: {type(exc).__name__}: {exc}",
            )
        )
        return findings

    if calls is None or calls.empty:
        findings.append(
            Finding(
                "20-day option contract history",
                False,
                "Calls dataframe empty, cannot test contract history",
            )
        )
        return findings

    contract_symbol = str(calls.iloc[0]["contractSymbol"])

    try:
        contract_hist = yf.Ticker(contract_symbol).history(
            period="20d",
            interval="1d",
            auto_adjust=False,
        )
        cols = list(contract_hist.columns)
        has_hist_oi = any(c.lower().replace("_", " ") == "open interest" for c in cols)
        findings.append(
            Finding(
                "20-day option contract history",
                not contract_hist.empty,
                f"contract={contract_symbol}, rows={len(contract_hist)}, cols={cols}",
            )
        )
        findings.append(
            Finding(
                "20-day historical OI column",
                has_hist_oi,
                (
                    "Open interest is present in option contract history"
                    if has_hist_oi
                    else "No Open Interest column in option contract history"
                ),
            )
        )
    except Exception as exc:
        findings.append(
            Finding(
                "20-day option contract history",
                False,
                f"Could not fetch option contract history: {type(exc).__name__}: {exc}",
            )
        )

    return findings


def print_source_summary() -> None:
    print("\nSource summary")
    print("-" * 78)
    print(
        textwrap.fill(
            (
                "The installed yfinance source exposes openInterest inside the current "
                "option-chain dataframe, but its public option_chain() method only takes "
                "an expiry date and timezone. That strongly suggests snapshot access, not "
                "native 20-day OI history."
            ),
            width=78,
        )
    )


def print_verdict(findings: list[Finding]) -> None:
    print("\nVerdict")
    print("-" * 78)

    by_name = {f.name: f for f in findings}
    snapshot = by_name.get("current chain snapshot OI")
    hist_oi = by_name.get("20-day historical OI column")
    hist_contract = by_name.get("20-day option contract history")

    if snapshot and snapshot.ok and hist_oi and not hist_oi.ok:
        print(
            "yfinance appears to provide current option-chain open interest, but not a "
            "native 20-day open-interest history series."
        )
        print(
            "If you want 20-day average OI, the practical workflow is to snapshot the "
            "option chain each day and store it locally."
        )
    elif hist_oi and hist_oi.ok:
        print(
            "This environment returned an Open Interest column for 20-day option "
            "contract history."
        )
    elif hist_contract and not hist_contract.ok:
        print(
            "The live Yahoo test could not complete here, so rely on the source-code "
            "inspection result plus rerun the script in a network-enabled environment."
        )
    else:
        print(
            "No evidence of native 20-day option OI history was found. The source "
            "inspection points to snapshot-only OI access."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether yfinance provides 20-day options open interest history."
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Underlying stock symbol to test, default: AAPL",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Testing symbol: {args.symbol}")
    print(f"Installed yfinance version: {yf.__version__}")

    findings = static_source_checks()
    for finding in findings:
        emit(finding)

    print_source_summary()

    print("\nLive Yahoo test")
    print("-" * 78)
    live = live_checks(args.symbol)
    for finding in live:
        emit(finding)

    findings.extend(live)
    print_verdict(findings)
    return 0


if __name__ == "__main__":
    sys.exit(main())

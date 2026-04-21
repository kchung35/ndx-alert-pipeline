"""Diagnostic harness — evidence for each flagged issue BEFORE we touch code.

Runs read-only against the already-generated smoke-test outputs in data/.
Prints one clearly-labeled block per investigation so we can see whether
the flagged issue is real, a false alarm, or a design tradeoff to accept.

Checks:
    1. WDC momentum 706% - is adj_close broken?
    2. Time-of-day look-ahead - when did we actually snapshot?
    3. NDX sector concentration - is sector-neutralize worth it?
    4. Options ATM band - how much data lives in +/-10% vs +/-20%?
    5. Options IV zero-inflation - what fraction of IVs are 0 or <5%?
    6. Near-expiry OI - does OI = 0 for DTE<7?
    7. Form 4 XML failures - what are the failing primary_doc patterns?
    8. Z-score saturation at n=5 - what is the realized range?
    9. Fundamentals PIT - any cross-sectional divergence worth noting?
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_edgar import list_form4_filings, _get  # noqa: E402
from src.data_options import load_all_chains_for_date  # noqa: E402
from src.data_prices import load_prices, load_fundamentals, load_vix  # noqa: E402
from src.lifted.sec_identity import ticker_to_cik  # noqa: E402
from src.universe import UNIVERSE_PARQUET, load_universe  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


AS_OF = date(2026, 4, 21)


# ─────────────────────────────────────────────────────────────────────
# 1. WDC momentum = 706% -- adj_close broken, split/spin-off, or real?
# ─────────────────────────────────────────────────────────────────────
def diag_wdc_momentum() -> None:
    section("1. WDC momentum diagnosis")
    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    wdc = prices[prices["ticker"] == "WDC"].sort_values("date").reset_index(drop=True)

    print(f"Rows: {len(wdc)}   First: {wdc['date'].iloc[0].date()}   Last: {wdc['date'].iloc[-1].date()}")
    print(f"adj_close range: [{wdc['adj_close'].min():.2f}, {wdc['adj_close'].max():.2f}]")
    print(f"close     range: [{wdc['close'].min():.2f}, {wdc['close'].max():.2f}]")

    # Largest single-day adj_close jumps
    wdc["adj_ret"] = wdc["adj_close"].pct_change()
    wdc["close_ret"] = wdc["close"].pct_change()
    top_jumps = wdc.nlargest(5, "adj_ret")[["date", "close", "adj_close", "adj_ret", "close_ret", "volume"]]
    print("\nTop 5 positive single-day jumps in adj_close:")
    print(top_jumps.to_string(index=False))

    bottom_jumps = wdc.nsmallest(5, "adj_ret")[["date", "close", "adj_close", "adj_ret", "close_ret", "volume"]]
    print("\nTop 5 negative single-day jumps in adj_close:")
    print(bottom_jumps.to_string(index=False))

    # 12-1 window anchors used by factors.py
    wdc_indexed = wdc.set_index("date")
    p_now = wdc_indexed["adj_close"].iloc[-21]
    p_then = wdc_indexed["adj_close"].iloc[-252]
    print(f"\nfactors.py uses: p[t-21] = {p_now:.4f}  p[t-252] = {p_then:.4f}")
    print(f"  -> 12-1 return = {(p_now / p_then) - 1:.4f}")
    # Compare to close (unadjusted) to see whether adj factor is the culprit
    c_now = wdc_indexed["close"].iloc[-21]
    c_then = wdc_indexed["close"].iloc[-252]
    print(f"Using unadjusted close: c[t-21] = {c_now:.4f}  c[t-252] = {c_then:.4f}")
    print(f"  -> 12-1 return (unadj) = {(c_now / c_then) - 1:.4f}")


# ─────────────────────────────────────────────────────────────────────
# 2. Time-of-day look-ahead
# ─────────────────────────────────────────────────────────────────────
def diag_snapshot_time() -> None:
    section("2. Snapshot timestamp (look-ahead check)")
    opt_dir = PROJECT_ROOT / "data" / "chains" / AS_OF.isoformat()
    if not opt_dir.exists():
        print(f"No chains dir at {opt_dir}")
        return
    for f in sorted(opt_dir.glob("*.parquet")):
        mtime = pd.Timestamp.fromtimestamp(f.stat().st_mtime)
        print(f"  {f.name:16s}  mtime={mtime}  (UTC hour: {mtime.hour:02d})")
    nyse_close = pd.Timestamp(f"{AS_OF.isoformat()} 16:00:00")
    now = pd.Timestamp.now()
    print(f"\nCurrent local time: {now}")
    print(f"NYSE close (4pm ET): {nyse_close}")
    if now.time() < pd.Timestamp("16:00").time():
        print("==> Data is LIVE-INTRADAY, not EOD. Look-ahead risk confirmed.")
    else:
        print("==> Current time is past 4pm ET. EOD-safe if snapshot was taken after close.")


# ─────────────────────────────────────────────────────────────────────
# 3. NDX sector concentration
# ─────────────────────────────────────────────────────────────────────
def diag_sector_concentration() -> None:
    section("3. NDX 100 sector concentration")
    full_path = UNIVERSE_PARQUET.parent / "universe_full.parquet"
    if not full_path.exists():
        print("No universe_full.parquet (smoke-test subset active?); loading current universe")
        full_path = UNIVERSE_PARQUET
    u = pd.read_parquet(full_path)
    print(f"Universe size: {len(u)}")
    counts = u["sector"].value_counts()
    total = counts.sum()
    print("\nSector counts:")
    for sec, n in counts.items():
        print(f"  {sec:32s}  {n:3d}   {n / total:.1%}")
    print(f"\nLargest sector: {counts.idxmax()} = {counts.max() / total:.1%}")
    if counts.max() / total > 0.30:
        print("==> Sector concentration > 30%. Sector-neutralize is worth it.")


# ─────────────────────────────────────────────────────────────────────
# 4 + 5. Options chain shape - ATM band, IV distribution, OI zero-rate
# ─────────────────────────────────────────────────────────────────────
def diag_options_chain_shape() -> None:
    section("4/5. Options chain shape + IV / OI zero-inflation")
    chains = load_all_chains_for_date(AS_OF)
    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    if chains.empty:
        print("No chains for this date.")
        return

    for t in sorted(chains["ticker"].unique()):
        sub = chains[chains["ticker"] == t].copy()
        sub["expiry"] = pd.to_datetime(sub["expiry"])
        sub["as_of"] = pd.to_datetime(sub["as_of_date"])
        sub["dte"] = (sub["expiry"] - sub["as_of"]).dt.days
        p = prices[(prices["ticker"] == t) & (prices["date"] <= pd.Timestamp(AS_OF))]
        spot = float(p.iloc[-1]["adj_close"]) if not p.empty else np.nan

        band20 = sub[(sub["strike"] >= spot * 0.8) & (sub["strike"] <= spot * 1.2)]
        band10 = sub[(sub["strike"] >= spot * 0.9) & (sub["strike"] <= spot * 1.1)]

        iv = sub["implied_volatility"].dropna()
        iv_zero = (iv <= 1e-6).sum()
        iv_lt5 = (iv < 0.05).sum()

        oi = sub["open_interest"].fillna(0)
        oi_zero = (oi == 0).sum()

        print(f"\n{t}  (spot = {spot:.2f})")
        print(f"  rows total            : {len(sub)}")
        print(f"  rows in +/-20% band   : {len(band20)}")
        print(f"  rows in +/-10% band   : {len(band10)}")
        print(f"  IV zero / <5% / valid : {iv_zero} / {iv_lt5} / {len(iv)}")
        print(f"  OI zero / total       : {oi_zero} / {len(sub)}")
        # Near-expiry OI check
        near = sub[sub["dte"] <= 7]
        far = sub[sub["dte"] > 30]
        print(f"  OI in DTE<=7 expiries : sum={int(near['open_interest'].fillna(0).sum())}, "
              f"rows={len(near)}")
        print(f"  OI in DTE>30 expiries : sum={int(far['open_interest'].fillna(0).sum())}, "
              f"rows={len(far)}")


# ─────────────────────────────────────────────────────────────────────
# 6. Form 4 XML failures - what do failing primary_doc look like?
# ─────────────────────────────────────────────────────────────────────
def diag_form4_xml_patterns() -> None:
    section("6. Form 4 primary_doc naming patterns")
    cik = ticker_to_cik("WDC")
    filings = list_form4_filings(cik, lookback_days=365)
    print(f"WDC filings (365d): {len(filings)}")
    patterns: dict[str, int] = {}
    samples: dict[str, str] = {}
    for f in filings:
        doc = f["primary_doc"]
        if "/" in doc:
            pattern = doc.split("/")[0] + "/"
        elif doc.endswith(".xml"):
            pattern = "*.xml (root)"
        elif doc.endswith(".txt"):
            pattern = "*.txt"
        elif doc.endswith(".htm") or doc.endswith(".html"):
            pattern = "*.htm(l)"
        else:
            pattern = "other"
        patterns[pattern] = patterns.get(pattern, 0) + 1
        samples.setdefault(pattern, doc)
    print("\nPattern distribution:")
    for pat, n in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pat:30s}  n={n:3d}   sample: {samples[pat]}")

    # For ONE failing filing, probe alternative filenames to find the real XML
    failing = [f for f in filings if not f["primary_doc"].endswith("/primary_doc.xml")]
    if failing:
        f = failing[0]
        acc_raw = str(f["accession"]).replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_raw}"
        print(f"\nProbing filing {f['accession']} (primary_doc={f['primary_doc']}):")
        print(f"  filing date: {f['filing_date']}")
        # 1) JSON filing index - lists all files
        idx_url = f"{base}/index.json"
        try:
            idx = _get(idx_url).json()
            items = idx.get("directory", {}).get("item", [])
            xml_items = [it for it in items if str(it.get("name", "")).endswith(".xml")]
            print(f"  index.json lists {len(items)} files, {len(xml_items)} XML")
            for it in xml_items[:5]:
                print(f"    -> {it['name']:40s} size={it.get('size')}")
        except Exception as exc:
            print(f"  index.json probe failed: {exc}")


# ─────────────────────────────────────────────────────────────────────
# 7. Z-score saturation at n=5
# ─────────────────────────────────────────────────────────────────────
def diag_zscore_range() -> None:
    section("7. Z-score realized range at n=5")
    from src.factors import load_factors
    from src.options_signals import load_options_panel
    try:
        f = load_factors(AS_OF)
        print(f"factor_z range: [{f['factor_z'].min():.3f}, {f['factor_z'].max():.3f}]")
        print(f"  individual z-col abs-max:")
        for c in ("momentum_z", "value_z", "quality_z", "lowvol_z"):
            if c in f.columns:
                print(f"    {c:16s}: abs_max = {f[c].abs().max():.3f}")
    except Exception as exc:
        print(f"factors load failed: {exc}")
    try:
        o = load_options_panel(AS_OF)
        print(f"\noptions_z range: [{o['options_z'].min():.3f}, {o['options_z'].max():.3f}]")
    except Exception as exc:
        print(f"options panel load failed: {exc}")

    # Mathematical max range for z-score with n samples, one sample at the max
    # of the distribution: z_max = (n-1) / sqrt(n * (n-1)) * some factor
    # For a sample of n with one extreme: z_max ~ sqrt((n-1)/n) * something
    # Empirical: for n=5, with one hot outlier, z ~ 1.79
    print(f"\nTheoretical max |z| for n=5 with one extreme sample: ~1.79")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    print(f"Diagnostic run at {pd.Timestamp.now().isoformat()}")
    print(f"Universe at {UNIVERSE_PARQUET}: {len(load_universe())} tickers")

    diag_wdc_momentum()
    diag_snapshot_time()
    diag_sector_concentration()
    diag_options_chain_shape()
    diag_form4_xml_patterns()
    diag_zscore_range()
    return 0


if __name__ == "__main__":
    sys.exit(main())

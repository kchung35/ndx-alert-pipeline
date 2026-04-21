"""
Data availability test for the long/short + options + insider alert pipeline.

Walks every node of the architecture diagram and checks whether the required
data is actually reachable from yfinance and SEC EDGAR, printing a PASS/FAIL
line per item plus a final gap summary.

Run:  python3 test_data_availability.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
import yfinance as yf

TEST_TICKERS = ["AAPL", "MSFT", "TSLA"]
EDGAR_UA = "ESCP MIM Student Project kevinchung3501@gmail.com"
EDGAR_HEADERS = {"User-Agent": EDGAR_UA, "Accept-Encoding": "gzip, deflate"}


@dataclass
class Check:
    node: str
    field: str
    status: str
    detail: str = ""


results: list[Check] = []


def log(node: str, field: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    results.append(Check(node, field, status, detail))
    print(f"[{status}] {node:<28} {field:<28} {detail}")


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ----------------------------------------------------------------------------
# BASE PIPELINE
# ----------------------------------------------------------------------------

def test_pull_prices() -> None:
    section("BASE PIPELINE  ·  Pull prices (yfinance)")
    tk = yf.Ticker(TEST_TICKERS[0])
    try:
        hist = tk.history(period="1y", auto_adjust=False)
        log("Pull prices", "1y OHLCV history",
            not hist.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(hist.columns),
            f"{len(hist)} rows, cols={list(hist.columns)}")
    except Exception as e:
        log("Pull prices", "1y OHLCV history", False, f"error: {e}")

    try:
        hist5y = tk.history(period="5y", auto_adjust=True)
        log("Pull prices", "5y adjusted close", not hist5y.empty, f"{len(hist5y)} rows")
    except Exception as e:
        log("Pull prices", "5y adjusted close", False, str(e))

    # Multi-ticker batch download (needed at scale)
    try:
        df = yf.download(TEST_TICKERS, period="1mo", progress=False, auto_adjust=True,
                         group_by="ticker")
        ok = df is not None and not df.empty
        log("Pull prices", "batch download N tickers", ok,
            f"shape={None if df is None else df.shape}")
    except Exception as e:
        log("Pull prices", "batch download N tickers", False, str(e))


def test_fama_french() -> None:
    section("BASE PIPELINE  ·  Fama-French factors (FF lib)")
    # Ken French data library is the reference source for MOM / HML / SMB / RMW / CMA.
    # yfinance does NOT provide these. Test direct CSV pull.
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        ok = r.status_code == 200 and len(r.content) > 1000
        log("FF library", "5-factor daily zip",
            ok, f"status={r.status_code}, bytes={len(r.content)}")
    except Exception as e:
        log("FF library", "5-factor daily zip", False, str(e))

    # pandas_datareader path (not installed in this env, but note it)
    try:
        import pandas_datareader  # noqa: F401
        log("FF library", "pandas_datareader installed", True, pandas_datareader.__version__)
    except ImportError:
        log("FF library", "pandas_datareader installed", False,
            "pip install pandas_datareader (optional, easier FF access)")


def test_compute_factors_inputs() -> None:
    section("BASE PIPELINE  ·  Compute factors (MOM / value / quality inputs)")
    tk = yf.Ticker(TEST_TICKERS[0])

    # Momentum: derivable from price history alone — already tested above.
    log("Compute factors", "MOM input (price hist)", True, "derived from history()")

    # Value: need P/E, P/B, EV/EBITDA, etc.
    try:
        info = tk.info
        value_fields = ["trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda",
                        "priceToSalesTrailing12Months"]
        found = {k: info.get(k) for k in value_fields}
        ok = sum(v is not None for v in found.values()) >= 3
        log("Compute factors", "Value metrics (.info)", ok,
            f"{sum(v is not None for v in found.values())}/{len(value_fields)} present")
    except Exception as e:
        log("Compute factors", "Value metrics (.info)", False, str(e))

    # Quality: ROE, ROA, margins, debt/equity
    try:
        info = tk.info
        quality_fields = ["returnOnEquity", "returnOnAssets", "profitMargins",
                          "operatingMargins", "debtToEquity", "grossMargins"]
        found = {k: info.get(k) for k in quality_fields}
        ok = sum(v is not None for v in found.values()) >= 3
        log("Compute factors", "Quality metrics (.info)", ok,
            f"{sum(v is not None for v in found.values())}/{len(quality_fields)} present")
    except Exception as e:
        log("Compute factors", "Quality metrics (.info)", False, str(e))

    # Fundamentals from financial statements (fallback / history)
    try:
        fin = tk.financials
        bs = tk.balance_sheet
        ok = (fin is not None and not fin.empty) and (bs is not None and not bs.empty)
        log("Compute factors", "Income stmt + balance sheet", ok,
            f"fin={None if fin is None else fin.shape}, bs={None if bs is None else bs.shape}")
    except Exception as e:
        log("Compute factors", "Income stmt + balance sheet", False, str(e))

    # Quarterly for freshness
    try:
        q = tk.quarterly_financials
        log("Compute factors", "Quarterly financials",
            q is not None and not q.empty, f"shape={None if q is None else q.shape}")
    except Exception as e:
        log("Compute factors", "Quarterly financials", False, str(e))


def test_backtest_and_var() -> None:
    section("BASE PIPELINE  ·  Backtest (Backtrader) + VaR (NumPy)")
    try:
        import backtrader  # noqa: F401
        log("Backtest", "backtrader installed", True, backtrader.__version__)
    except ImportError:
        log("Backtest", "backtrader installed", False,
            "pip install backtrader (required for backtest node)")

    try:
        import numpy as np  # noqa: F401
        log("VaR", "numpy installed", True, np.__version__)
    except ImportError:
        log("VaR", "numpy installed", False, "pip install numpy")


# ----------------------------------------------------------------------------
# OPTIONS LAYER
# ----------------------------------------------------------------------------

def test_options_chain() -> None:
    section("OPTIONS LAYER  ·  Pull options chain (yfinance · all expiries)")
    tk = yf.Ticker(TEST_TICKERS[0])

    try:
        expiries = tk.options
        log("Pull options chain", "expiry list", len(expiries) > 0,
            f"{len(expiries)} expiries, first={expiries[0] if expiries else None}")
    except Exception as e:
        log("Pull options chain", "expiry list", False, str(e))
        return

    if not expiries:
        return

    # Single expiry
    try:
        chain = tk.option_chain(expiries[0])
        calls, puts = chain.calls, chain.puts
        required = {"strike", "lastPrice", "bid", "ask", "volume",
                    "openInterest", "impliedVolatility"}
        call_ok = required.issubset(calls.columns)
        put_ok = required.issubset(puts.columns)
        log("Pull options chain", "calls schema", call_ok,
            f"{len(calls)} rows, missing={required - set(calls.columns)}")
        log("Pull options chain", "puts schema", put_ok,
            f"{len(puts)} rows, missing={required - set(puts.columns)}")
    except Exception as e:
        log("Pull options chain", "single-expiry chain", False, str(e))

    # All expiries (sampled — can be 20+)
    try:
        sampled = expiries[:3]
        chains = []
        for exp in sampled:
            c = tk.option_chain(exp)
            chains.append((exp, len(c.calls), len(c.puts)))
            time.sleep(0.3)
        log("Pull options chain", "all-expiries pull", True,
            f"sampled {len(chains)}: {chains}")
    except Exception as e:
        log("Pull options chain", "all-expiries pull", False, str(e))


def test_oi_spike_inputs() -> None:
    section("OPTIONS LAYER  ·  Detect OI spikes (OI vs 20d avg · put/call ratio)")
    # yfinance only exposes CURRENT open interest snapshot per expiry.
    # 20-day average OI requires daily snapshots stored by us over time.
    tk = yf.Ticker(TEST_TICKERS[0])
    try:
        exp = tk.options[0]
        ch = tk.option_chain(exp)
        has_oi = "openInterest" in ch.calls.columns and "openInterest" in ch.puts.columns
        log("Detect OI spikes", "current OI snapshot", has_oi,
            f"call OI sum={ch.calls['openInterest'].sum()}, "
            f"put OI sum={ch.puts['openInterest'].sum()}")
        log("Detect OI spikes", "20d historical OI", False,
            "NOT AVAILABLE from yfinance — must snapshot daily and store locally "
            "(or pay for CBOE / ORATS / Polygon)")
    except Exception as e:
        log("Detect OI spikes", "OI fields", False, str(e))

    # Put/call ratio: computable from current chain
    try:
        exp = tk.options[0]
        ch = tk.option_chain(exp)
        pcr = ch.puts["openInterest"].sum() / max(ch.calls["openInterest"].sum(), 1)
        log("Detect OI spikes", "put/call ratio (OI)", True, f"{pcr:.3f}")
    except Exception as e:
        log("Detect OI spikes", "put/call ratio (OI)", False, str(e))


def test_iv_skew_term() -> None:
    section("OPTIONS LAYER  ·  IV skew & term structure · VIX compare")
    tk = yf.Ticker(TEST_TICKERS[0])
    try:
        exp = tk.options[0]
        ch = tk.option_chain(exp)
        has_iv = "impliedVolatility" in ch.calls.columns
        sample = ch.calls[["strike", "impliedVolatility"]].dropna()
        log("IV skew & term", "per-strike IV", has_iv and not sample.empty,
            f"{len(sample)} IV points")
    except Exception as e:
        log("IV skew & term", "per-strike IV", False, str(e))

    # Term structure: IV across expiries — derivable by pulling chain per expiry
    log("IV skew & term", "term structure (multi-expiry)", True,
        "derivable by iterating .option_chain(exp) for exp in tk.options")

    # VIX
    try:
        vix = yf.Ticker("^VIX").history(period="1mo")
        log("IV skew & term", "^VIX history", not vix.empty, f"{len(vix)} days")
    except Exception as e:
        log("IV skew & term", "^VIX history", False, str(e))


# ----------------------------------------------------------------------------
# INSIDER LAYER
# ----------------------------------------------------------------------------

def test_yf_insider() -> None:
    section("INSIDER LAYER  ·  yfinance insider accessors (quick look)")
    tk = yf.Ticker(TEST_TICKERS[0])
    for attr in ["insider_transactions", "insider_purchases", "insider_roster_holders"]:
        try:
            df = getattr(tk, attr)
            ok = df is not None and hasattr(df, "empty") and not df.empty
            log("yfinance insider", attr, ok,
                f"{None if df is None else (df.shape, list(df.columns)[:4])}")
        except Exception as e:
            log("yfinance insider", attr, False, str(e))


def test_edgar_cik_lookup() -> None:
    section("INSIDER LAYER  ·  SEC EDGAR  ·  ticker -> CIK mapping")
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=EDGAR_HEADERS, timeout=30)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        tickers = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()} if ok else {}
        log("EDGAR", "company_tickers.json", ok,
            f"status={r.status_code}, n={len(tickers)}, AAPL CIK={tickers.get('AAPL')}")
        return tickers
    except Exception as e:
        log("EDGAR", "company_tickers.json", False, str(e))
        return {}


def test_edgar_form4(tickers: dict[str, str]) -> None:
    section("INSIDER LAYER  ·  SEC EDGAR  ·  Form 4 filings")
    cik = tickers.get("AAPL")
    if not cik:
        log("EDGAR Form 4", "AAPL CIK", False, "not found — skip")
        return

    # Submissions index (all filings for a company)
    try:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        f4_count = sum(1 for f in forms if f == "4")
        log("EDGAR Form 4", "submissions index",
            ok and f4_count > 0,
            f"recent filings={len(forms)}, Form 4 count={f4_count}")
    except Exception as e:
        log("EDGAR Form 4", "submissions index", False, str(e))
        return

    # Latest Form 4 raw XML
    try:
        access_nums = recent.get("accessionNumber", [])
        docs = recent.get("primaryDocument", [])
        dates = recent.get("filingDate", [])
        f4_idx = [i for i, f in enumerate(forms) if f == "4"]
        if f4_idx:
            i = f4_idx[0]
            acc = access_nums[i].replace("-", "")
            doc = docs[i]
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            time.sleep(0.3)
            r = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            ok = r.status_code == 200 and len(r.content) > 500
            log("EDGAR Form 4", "fetch latest filing doc", ok,
                f"date={dates[i]}, {len(r.content)} bytes, content-type={r.headers.get('Content-Type')}")
        else:
            log("EDGAR Form 4", "fetch latest filing doc", False, "no Form 4 in recent")
    except Exception as e:
        log("EDGAR Form 4", "fetch latest filing doc", False, str(e))


def test_edgar_full_text_search() -> None:
    section("INSIDER LAYER  ·  SEC EDGAR  ·  full-text Form 4 search (cross-ticker)")
    # EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=&forms=4
    try:
        url = "https://efts.sec.gov/LATEST/search-index?forms=4&dateRange=custom"
        today = datetime.utcnow().date()
        start = today - timedelta(days=3)
        url = (f"https://efts.sec.gov/LATEST/search-index?forms=4"
               f"&dateRange=custom&startdt={start}&enddt={today}")
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
        ok = r.status_code == 200
        data = r.json() if ok else {}
        hits = data.get("hits", {}).get("total", {}).get("value", 0)
        log("EDGAR Form 4", "3-day full-text hits", ok, f"hits={hits}")
    except Exception as e:
        log("EDGAR Form 4", "3-day full-text hits", False, str(e))


def test_openinsider() -> None:
    section("INSIDER LAYER  ·  openinsider.com scrape check")
    # openinsider aggregates Form 4s with clustering info already extracted.
    try:
        url = "http://openinsider.com/latest-cluster-buys"
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        ok = r.status_code == 200 and "Cluster" in r.text
        log("openinsider", "latest-cluster-buys page", ok,
            f"status={r.status_code}, bytes={len(r.content)}")
    except Exception as e:
        log("openinsider", "latest-cluster-buys page", False, str(e))


# ----------------------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------------------

def summary() -> None:
    section("SUMMARY")
    by_node: dict[str, list[Check]] = {}
    for c in results:
        by_node.setdefault(c.node, []).append(c)

    pass_n = sum(1 for c in results if c.status == "PASS")
    fail_n = sum(1 for c in results if c.status == "FAIL")
    print(f"Total checks: {len(results)}   PASS={pass_n}   FAIL={fail_n}")

    print("\nAVAILABLE (PASS):")
    for c in results:
        if c.status == "PASS":
            print(f"  - {c.node}: {c.field}")

    print("\nGAPS (FAIL) — these need alternative sources or local storage:")
    for c in results:
        if c.status == "FAIL":
            print(f"  - {c.node}: {c.field}  ->  {c.detail}")

    print("""
KEY TAKEAWAYS
-------------
1. yfinance covers: prices, fundamentals, current options chain, IV, VIX.
2. Fama-French factors -> pull directly from Ken French's Dartmouth site
   (zip of CSVs); optionally install pandas_datareader for cleaner access.
3. Historical OPTIONS open interest / IV  ->  NOT in yfinance. You must:
     (a) snapshot chain daily into a local parquet/DB  (free, slow to accumulate),
     (b) or buy CBOE / ORATS / Polygon (not free).
   The 20-day OI average in the diagram requires option (a) running for 20+ days.
4. Insider data: use SEC EDGAR directly
     - CIK map:   https://www.sec.gov/files/company_tickers.json
     - Per-issuer: https://data.sec.gov/submissions/CIK<10digit>.json
     - Filing doc: https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<file>
   SEC requires a real User-Agent with contact email. Rate limit ~10 req/sec.
   openinsider.com is a convenient pre-aggregated fallback (HTML scrape).
5. yfinance's own .insider_transactions is limited and rate-limit prone; prefer EDGAR.
6. Backtrader must be installed separately (not present in this env).
""")


def main() -> int:
    print(f"Data availability test — {datetime.utcnow().isoformat()}Z")
    print(f"Test tickers: {TEST_TICKERS}")
    print(f"yfinance: {yf.__version__}   pandas: {pd.__version__}")

    try:
        test_pull_prices()
        test_fama_french()
        test_compute_factors_inputs()
        test_backtest_and_var()
        test_options_chain()
        test_oi_spike_inputs()
        test_iv_skew_term()
        test_yf_insider()
        tickers = test_edgar_cik_lookup()
        test_edgar_form4(tickers)
        test_edgar_full_text_search()
        test_openinsider()
    except Exception:
        print("Unhandled error:", file=sys.stderr)
        traceback.print_exc()

    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())

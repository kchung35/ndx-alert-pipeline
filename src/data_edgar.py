"""SEC EDGAR Form 4 fetcher (raw REST API, no edgartools).

Replaces pms_app's insiders.py (2,043 lines, /tmp lock file, Streamlit
caching, DB coupling) with a focused ~300-line implementation:

  1. Look up the CIK via sec_identity.
  2. Hit https://data.sec.gov/submissions/CIK{cik}.json to list filings.
  3. Filter form == "4" and fetch each filing's primary XML document.
  4. Parse transaction code / shares / price / officer title with lxml.
  5. Persist cumulative rows at data/form4/{ticker}.parquet.

Rate limit: 10 req/s per SEC rules (we conservatively cap at 8 rps).
User-Agent: comes from the SEC_USER_AGENT env var.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from lxml import etree

from src.lifted.insider_utils import (
    classify_transaction, is_corporate_entity, officer_weight,
)
from src.lifted.sec_identity import ticker_to_cik
from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

FORM4_DIR = PROJECT_ROOT / "data" / "form4"

_FORM4_COLS = [
    "ticker", "cik", "accession", "filing_date", "transaction_date",
    "insider_name", "position",
    "tx_code", "is_derivative", "is_10b5_1",
    "shares", "price", "value",
    "signal_label", "signal_weight", "officer_weight",
]


class _TokenBucket:
    def __init__(self, max_per_sec: float):
        self._min_interval = 1.0 / max_per_sec
        self._last = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


_SEC_GATE = _TokenBucket(max_per_sec=8.0)


def _sec_headers() -> dict:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT env var not set. Example:\n"
            "  export SEC_USER_AGENT='Kevin Chung kevin@example.com'"
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _get(url: str, timeout: int = 30) -> requests.Response:
    _SEC_GATE.acquire()
    resp = requests.get(url, headers=_sec_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp


# ── filing list per CIK ─────────────────────────────────────────────────

def list_form4_filings(
    cik: str,
    lookback_days: int = 365,
    as_of: date | None = None,
) -> list[dict]:
    """Return list of {accession, filing_date, primary_doc} for recent Form 4s."""
    cik_padded = str(cik).strip().zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    data = _get(url).json()
    recent = data.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])

    as_of_date = as_of or date.today()
    as_of_ts = pd.Timestamp(as_of_date)
    cutoff = as_of_ts - pd.Timedelta(days=lookback_days)
    out: list[dict] = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            fd = pd.Timestamp(dates[i])
        except Exception:
            continue
        if fd < cutoff or fd > as_of_ts:
            continue
        out.append({
            "accession": accs[i],
            "filing_date": dates[i],
            "primary_doc": docs[i],
        })
    return out


# ── Form 4 XML parse ────────────────────────────────────────────────────

def _text(elem, xpath: str, default: str = "") -> str:
    res = elem.xpath(xpath)
    if not res:
        return default
    node = res[0]
    txt = getattr(node, "text", None) if hasattr(node, "text") else node
    return (txt or "").strip() if isinstance(txt, str) else default


def _float(elem, xpath: str) -> float | None:
    val = _text(elem, xpath)
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _bool(elem, xpath: str) -> bool | None:
    val = _text(elem, xpath).lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return None


def parse_form4_xml(content: bytes, ticker: str, cik: str,
                    accession: str, filing_date: str) -> list[dict]:
    """Extract non-derivative + derivative transactions from a Form 4 XML doc."""
    try:
        tree = etree.fromstring(content)
    except etree.XMLSyntaxError:
        # Some filings return an HTML index page; skip quietly.
        return []

    # Insider name + position (reportingOwner section)
    name = _text(tree, ".//reportingOwner/reportingOwnerId/rptOwnerName/text()")
    is_officer = (_text(tree, ".//reportingOwner/reportingOwnerRelationship/isOfficer/text()") or "").lower() in ("1", "true")
    is_director = (_text(tree, ".//reportingOwner/reportingOwnerRelationship/isDirector/text()") or "").lower() in ("1", "true")
    is_ten_pct = (_text(tree, ".//reportingOwner/reportingOwnerRelationship/isTenPercentOwner/text()") or "").lower() in ("1", "true")
    officer_title = _text(tree, ".//reportingOwner/reportingOwnerRelationship/officerTitle/text()")

    if is_officer and officer_title:
        position = officer_title
    elif is_director:
        position = "Director"
    elif is_ten_pct:
        position = "10% Owner"
    else:
        position = officer_title or ""

    if not name or is_corporate_entity(name):
        return []

    rows: list[dict] = []
    for deriv_flag, section in (
        (False, ".//nonDerivativeTable/nonDerivativeTransaction"),
        (True, ".//derivativeTable/derivativeTransaction"),
    ):
        for tx in tree.xpath(section):
            code = _text(tx, ".//transactionCoding/transactionCode/text()")
            if not code:
                continue
            tx_date = _text(tx, ".//transactionDate/value/text()")
            shares = _float(tx, ".//transactionAmounts/transactionShares/value/text()")
            price = _float(tx, ".//transactionAmounts/transactionPricePerShare/value/text()")
            is_10b5_1 = _bool(tx, ".//transactionCoding/rule10b5_1Flag/value/text()")
            if shares is None:
                continue
            value = (shares or 0.0) * (price or 0.0) if price else None
            label, weight = classify_transaction(code, deriv_flag, is_10b5_1)

            rows.append({
                "ticker": ticker,
                "cik": cik,
                "accession": accession,
                "filing_date": filing_date,
                "transaction_date": tx_date or filing_date,
                "insider_name": name,
                "position": position,
                "tx_code": code,
                "is_derivative": deriv_flag,
                "is_10b5_1": is_10b5_1,
                "shares": shares,
                "price": price,
                "value": value,
                "signal_label": label,
                "signal_weight": weight,
                "officer_weight": officer_weight(position),
            })
    return rows


# ── top-level fetch per ticker ──────────────────────────────────────────

def _list_xml_candidates(base: str) -> list[str]:
    """Probe the filing's index.json and return .xml files in the root dir.

    Diagnostics against WDC's 148 Form 4 filings showed two real-world
    naming conventions coexist:
        * modern (xslF345X06): XML at primary_doc.xml
        * older  (xslF345X05): XML at edgardoc.xml
        * pre-2013           : accession-based filename (e.g. wk-form4_*.xml)
    index.json enumerates them all, so we probe once per filing rather than
    guessing at filenames.
    """
    try:
        data = _get(f"{base}/index.json").json()
    except Exception as exc:
        logger.debug("index.json probe failed %s: %s", base, exc)
        return []
    items = data.get("directory", {}).get("item", []) or []
    names = [str(it.get("name", "")) for it in items]
    # Prefer primary_doc.xml / edgardoc.xml first, then other .xml files.
    xml_names = [n for n in names if n.endswith(".xml") and "/" not in n]
    preferred = [n for n in xml_names if n in {"primary_doc.xml", "edgardoc.xml"}]
    others = [n for n in xml_names if n not in preferred]
    return preferred + others


def _fetch_form4_xml(cik: str, accession: str, primary_doc: str) -> bytes | None:
    """Fetch the structured Form 4 XML by probing the filing directory.

    Tries (in order): the XML files listed in index.json, then a hard-coded
    primary_doc.xml / edgardoc.xml fallback if index.json is unreachable,
    then the raw primaryDocument field as last resort.
    """
    acc_raw = str(accession).replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_raw}"

    candidates = _list_xml_candidates(base)
    # Hard-coded fallbacks in case index.json is unreachable.
    for fallback in ("primary_doc.xml", "edgardoc.xml", primary_doc):
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        url = f"{base}/{candidate}"
        try:
            resp = _get(url)
        except Exception as exc:
            logger.debug("XML probe failed %s: %s", url, exc)
            continue
        if resp.content.startswith(b"<?xml"):
            return resp.content
    return None


def _cached_accessions(ticker: str) -> set[str]:
    """Return already persisted Form 4 accession IDs for incremental refresh."""
    path = FORM4_DIR / f"{ticker}.parquet"
    if not path.exists():
        return set()
    try:
        existing = pd.read_parquet(path, columns=["accession"])
    except Exception as exc:
        logger.warning("%s: could not read cached accessions: %s", ticker, exc)
        return set()
    if "accession" not in existing.columns:
        return set()
    return set(existing["accession"].dropna().astype(str))


def fetch_form4_for_ticker(
    ticker: str,
    lookback_days: int = 365,
    as_of: date | None = None,
    skip_cached: bool = True,
) -> pd.DataFrame:
    cik = ticker_to_cik(ticker)
    if not cik:
        logger.warning("no CIK for %s", ticker)
        return pd.DataFrame(columns=_FORM4_COLS)

    filings = list_form4_filings(cik, lookback_days=lookback_days, as_of=as_of)
    known = _cached_accessions(ticker) if skip_cached else set()
    new_filings = [f for f in filings if str(f["accession"]) not in known]
    skipped_cached = len(filings) - len(new_filings)
    logger.info(
        "%s (CIK %s): %d Form 4 filings in %dd through %s; %d cached, %d new",
        ticker, cik, len(filings), lookback_days, as_of or date.today(),
        skipped_cached, len(new_filings),
    )
    all_rows: list[dict] = []
    missing_xml = 0
    for f in new_filings:
        xml = _fetch_form4_xml(cik, f["accession"], f["primary_doc"])
        if xml is None:
            missing_xml += 1
            continue
        rows = parse_form4_xml(
            xml, ticker, cik, f["accession"], f["filing_date"],
        )
        all_rows.extend(rows)
    if missing_xml:
        logger.warning("%s: %d/%d new filings had no parsable XML",
                       ticker, missing_xml, len(new_filings))
    if not all_rows:
        return pd.DataFrame(columns=_FORM4_COLS)
    df = pd.DataFrame(all_rows)
    # Dedup in case of re-runs
    return df.drop_duplicates(
        subset=["accession", "tx_code", "transaction_date", "shares", "price"],
        keep="last",
    )


def save_form4(ticker: str, df: pd.DataFrame) -> Path:
    FORM4_DIR.mkdir(parents=True, exist_ok=True)
    path = FORM4_DIR / f"{ticker}.parquet"
    # Merge with existing if present (cumulative)
    if path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["accession", "tx_code", "transaction_date", "shares", "price"],
            keep="last",
        )
    df.to_parquet(path, index=False)
    return path


def load_form4(ticker: str) -> pd.DataFrame:
    path = FORM4_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=_FORM4_COLS)
    return pd.read_parquet(path)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Pull NDX 100 Form 4 filings from EDGAR.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat(),
                        help="As-of date for the filing lookback window.")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--force-refetch", action="store_true",
                        help="Fetch XML for cached accessions too.")
    parser.add_argument("--tickers", nargs="*", help="Subset for debugging")
    args = parser.parse_args()
    as_of = date.fromisoformat(args.date)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_universe()["ticker"].tolist()

    total_rows = 0
    for t in tickers:
        try:
            df = fetch_form4_for_ticker(
                t,
                lookback_days=args.lookback_days,
                as_of=as_of,
                skip_cached=not args.force_refetch,
            )
        except Exception as exc:
            logger.warning("%s: failed %s", t, exc)
            continue
        if df.empty:
            continue
        save_form4(t, df)
        total_rows += len(df)
    logger.info("Processed %d tickers, %d total Form 4 rows", len(tickers), total_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

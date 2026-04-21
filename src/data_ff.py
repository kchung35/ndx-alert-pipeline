"""Ken French Data Library loader (5-factor daily).

Clean fresh implementation — the pms_app version is tightly coupled to a
DB persistence layer. Here we simply download the zip, parse the quirky
CSV format, and write parquet.

Factors returned (as decimal returns, e.g. 0.01 = 1%):
    Mkt-RF, SMB, HML, RMW, CMA, RF
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.universe import PROJECT_ROOT

logger = logging.getLogger(__name__)

FF_PARQUET = PROJECT_ROOT / "data" / "ff.parquet"
_FF_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)


def fetch_ff_daily() -> pd.DataFrame:
    resp = requests.get(_FF_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError("FF zip is empty")
        raw = zf.read(names[0]).decode("latin-1")

    # The FF CSV has a header block with metadata we need to skip. Find the
    # first line whose first column parses as YYYYMMDD.
    lines = raw.splitlines()
    start = None
    for i, line in enumerate(lines):
        first = line.split(",", 1)[0].strip()
        if first.isdigit() and len(first) == 8:
            start = i
            break
    if start is None:
        raise RuntimeError("Could not locate FF data rows")

    # Headers are on the line immediately before the first data row
    header_line = lines[start - 1]
    headers = [h.strip() for h in header_line.split(",")]
    headers[0] = "date"

    body = "\n".join([",".join(headers), *lines[start:]])
    df = pd.read_csv(io.StringIO(body))
    # Stop at the first non-numeric date (file may have an annual section after daily)
    df = df[df["date"].astype(str).str.match(r"^\d{8}$", na=False)]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    # FF values are in percentage points with -99.99 sentinel for missing
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].replace(-99.99, pd.NA)
    df[numeric_cols] = df[numeric_cols].astype(float) / 100.0
    return df.reset_index(drop=True)


def save_ff(df: pd.DataFrame) -> Path:
    FF_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FF_PARQUET, index=False)
    return FF_PARQUET


def load_ff() -> pd.DataFrame:
    if not FF_PARQUET.exists():
        raise FileNotFoundError(FF_PARQUET)
    return pd.read_parquet(FF_PARQUET)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = fetch_ff_daily()
    save_ff(df)
    logger.info("Wrote %s (%d rows, %d cols)", FF_PARQUET, len(df), df.shape[1])
    print(df.tail(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

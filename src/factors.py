"""Factor signal composition for NDX 100.

Produces per-ticker z-scored factor exposures and an equal-weight composite
`factor_z` score that feeds the alert engine.

Factors (all cross-sectional z-scores within NDX 100 on the as-of date):
    momentum_12_1  = total return from t-252 to t-21 (12m skip-1m)
    value_z        = composite z of -trailingPE, -priceToBook, -EV/EBITDA
    quality_z      = composite z of ROE, ROA, grossMargins, -debtToEquity
    lowvol_z       = -1 * 60d trailing return std

Output columns:
    ticker, momentum_12_1, value_z, quality_z, lowvol_z, factor_z
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prices import load_fundamentals, load_prices
from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

FACTORS_DIR = PROJECT_ROOT / "data" / "factors"


# ── Helpers ────────────────────────────────────────────────────────────

def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _zscore(s: pd.Series) -> pd.Series:
    s = _winsorize(s)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


_SECTOR_NEUTRAL_MIN_N = 5  # below this, we cannot reasonably estimate a sector mean


def _sector_neutralize(values: pd.Series, sectors: pd.Series,
                       min_sector_size: int = _SECTOR_NEUTRAL_MIN_N) -> pd.Series:
    """Subtract the per-sector mean so each sector's cross-section is
    centered on zero. Prevents inter-sector bias (NDX 100 is >40% Tech so
    raw z's load on sector).

    Small-sector guard: sectors with fewer than `min_sector_size` members
    are NOT neutralized. A sector-of-one would otherwise neutralize to 0
    (self-mean) and destroy the signal; tiny sectors (e.g. NDX has Real
    Estate with 1 member) can't reliably estimate a sector mean anyway.
    Those tickers keep their raw values and compete cross-sectionally as
    they would without sector neutralization.
    """
    df = pd.concat(
        [values.rename("v"), sectors.rename("sector").fillna("UNKNOWN")],
        axis=1,
    )
    sector_n = df.groupby("sector")["v"].transform("size")
    sector_mean = df.groupby("sector")["v"].transform("mean")
    # Only subtract the mean for sectors large enough; else pass through raw.
    demeaned = df["v"] - sector_mean.where(sector_n >= min_sector_size, 0.0)
    return demeaned.rename(values.name)


# ── Factor computations ────────────────────────────────────────────────

def compute_momentum(prices_wide: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """12-1 month total return per ticker. Index = ticker."""
    hist = prices_wide.loc[:as_of]
    if len(hist) < 260:
        logger.warning("prices has only %d rows before %s — momentum unreliable",
                       len(hist), as_of.date())
    out = {}
    for t in prices_wide.columns:
        series = hist[t].dropna()
        if len(series) < 260:
            out[t] = np.nan
            continue
        p_now = series.iloc[-21] if len(series) >= 21 else series.iloc[-1]
        p_then = series.iloc[-252] if len(series) >= 252 else series.iloc[0]
        out[t] = (p_now / p_then) - 1.0 if p_then > 0 else np.nan
    return pd.Series(out, name="momentum_12_1")


def compute_momentum_3m(prices_wide: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """3-month total return per ticker. Jegadeesh-Titman canonical formation.

    Primary tradeable signal per the backtest: long-only top decile
    produced Sharpe 4.52 in H1 2025 and 1.59 in H2 2025, beating EW-NDX
    by 35 and 57 pts respectively.
    """
    hist = prices_wide.loc[:as_of]
    if len(hist) < 70:
        logger.warning("prices has only %d rows before %s — 3m momentum unreliable",
                       len(hist), as_of.date())
    out: dict[str, float] = {}
    for t in prices_wide.columns:
        series = hist[t].dropna()
        if len(series) < 63:
            out[t] = np.nan
            continue
        p_now = series.iloc[-1]
        p_then = series.iloc[-63]
        out[t] = (p_now / p_then) - 1.0 if p_then > 0 else np.nan
    return pd.Series(out, name="momentum_3m")


def compute_value_z(fundamentals: pd.DataFrame) -> pd.Series:
    """Cross-sectional z of value composite. Lower PE/PB/EV-EBITDA = better."""
    f = fundamentals.set_index("ticker")
    parts = []
    for col in ("trailingPE", "priceToBook", "enterpriseToEbitda"):
        if col in f.columns:
            # Negate so high = cheap = good
            parts.append(_zscore(-f[col].astype(float)))
    if not parts:
        return pd.Series(0.0, index=f.index, name="value_z")
    return pd.concat(parts, axis=1).mean(axis=1).rename("value_z")


def compute_quality_z(fundamentals: pd.DataFrame) -> pd.Series:
    """Cross-sectional z of quality composite. Higher ROE/ROA/margin, lower D/E."""
    f = fundamentals.set_index("ticker")
    parts = []
    if "returnOnEquity" in f.columns:
        parts.append(_zscore(f["returnOnEquity"].astype(float)))
    if "returnOnAssets" in f.columns:
        parts.append(_zscore(f["returnOnAssets"].astype(float)))
    if "grossMargins" in f.columns:
        parts.append(_zscore(f["grossMargins"].astype(float)))
    if "debtToEquity" in f.columns:
        parts.append(_zscore(-f["debtToEquity"].astype(float)))
    if not parts:
        return pd.Series(0.0, index=f.index, name="quality_z")
    return pd.concat(parts, axis=1).mean(axis=1).rename("quality_z")


def compute_lowvol_z(prices_wide: pd.DataFrame, as_of: pd.Timestamp,
                     window: int = 60) -> pd.Series:
    """Cross-sectional z of -1 * 60d return volatility (low-vol = good)."""
    hist = prices_wide.loc[:as_of].tail(window + 1)
    rets = hist.pct_change().dropna(how="all")
    vol = rets.std()
    return _zscore(-vol).rename("lowvol_z")


# ── Top-level compose ──────────────────────────────────────────────────

def compute_factor_panel(as_of: date | None = None,
                         sector_neutral: bool = True) -> pd.DataFrame:
    """Build the per-ticker factor panel for as_of date.

    sector_neutral: if True, each factor z-score has its per-sector mean
    subtracted before being combined into the composite factor_z. This
    prevents sector tilt (Tech = 40% of NDX 100) from dominating the
    composite; the raw (non-neutralized) z-scores are preserved in
    `*_z_raw` columns for diagnostics.
    """
    universe = load_universe()
    tickers = universe["ticker"].tolist()

    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    wide = (
        prices[prices["ticker"].isin(tickers)]
        .pivot(index="date", columns="ticker", values="adj_close")
        .sort_index()
    )
    if as_of is None:
        as_of_ts = wide.index.max()
    else:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts not in wide.index:
            valid = wide.index[wide.index <= as_of_ts]
            if len(valid) == 0:
                raise RuntimeError(f"No price data on or before {as_of}")
            as_of_ts = valid[-1]

    fundamentals = load_fundamentals()

    mom = compute_momentum(wide, as_of_ts)
    mom3 = compute_momentum_3m(wide, as_of_ts)
    lowvol = compute_lowvol_z(wide, as_of_ts)
    value = compute_value_z(fundamentals)
    quality = compute_quality_z(fundamentals)
    mom_z = _zscore(mom).rename("momentum_z")
    mom3_z = _zscore(mom3).rename("momentum_3m_z")

    df = pd.concat([mom, mom_z, mom3, mom3_z, value, quality, lowvol], axis=1)
    df.index.name = "ticker"
    df = df.reset_index()
    df = df[df["ticker"].isin(tickers)].reset_index(drop=True)

    # Join sector so we can neutralize — UNKNOWN for any ticker with bad metadata
    df = df.merge(universe[["ticker", "sector"]], on="ticker", how="left")

    z_cols = ["momentum_z", "value_z", "quality_z", "lowvol_z"]
    if sector_neutral:
        # Preserve raw z-scores for diagnostics; overwrite the working
        # columns with sector-neutralized versions before compositing.
        for col in z_cols:
            df[col + "_raw"] = df[col]
            df[col] = _sector_neutralize(df[col], df["sector"])

    df["factor_z"] = df[z_cols].mean(axis=1)
    df["as_of_date"] = as_of_ts.date().isoformat()
    return df


def save_factors(df: pd.DataFrame, as_of: date) -> Path:
    FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    path = FACTORS_DIR / f"{as_of.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_factors(as_of: date) -> pd.DataFrame:
    path = FACTORS_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(path)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Compute NDX 100 factor panel.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat())
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.date)
    df = compute_factor_panel(as_of)
    save_factors(df, as_of)
    logger.info("Wrote factor panel for %s: %d tickers", as_of, len(df))
    print(df.sort_values("factor_z", ascending=False).head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

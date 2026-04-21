"""Composite alert engine for NDX 100.

Combines three independent signal panels (factor, options, insider) into
a signed composite, and marks tickers across five meaningful tiers:

    STRONG_BULLISH      composite > 1.5   AND all 3 components > +0.3
    STRONG_BEARISH      composite < -1.5  AND all 3 components < -0.3
    MOMENTUM_LONG       ticker in top-10% by 3-month momentum z-score
                        (primary tradeable tier -- backtested Sharpe
                        1.6-4.5 across 2025; see backtest.py)
    CONFLUENCE_BULLISH  composite > 1.0   AND >= 2 of 3 components > +0.15
    CONFLUENCE_BEARISH  composite < -1.0  AND >= 2 of 3 components < -0.15
    NO_ALERT            else

Precedence when multiple rules match a single ticker (first wins):
    STRONG_BULLISH > STRONG_BEARISH > MOMENTUM_LONG
                   > CONFLUENCE_BULLISH > CONFLUENCE_BEARISH > NO_ALERT

Output columns:
    ticker, tier, composite, factor_z, options_z, insider_z,
    momentum_3m_z, momentum_long, rationale, as_of_date
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.factors import load_factors
from src.insider_signals import load_insider_panel
from src.options_signals import load_options_panel
from src.universe import PROJECT_ROOT

logger = logging.getLogger(__name__)

ALERTS_DIR = PROJECT_ROOT / "data" / "alerts"

# Composite weights
FACTOR_W = 0.40
OPTIONS_W = 0.30
INSIDER_W = 0.30

# Tier thresholds — hard (STRONG) and relaxed (CONFLUENCE)
BULL_THRESH = 1.5
BEAR_THRESH = -1.5
COMPONENT_BULL = 0.30
COMPONENT_BEAR = -0.30
CONFLUENCE_COMPOSITE = 1.0
CONFLUENCE_COMPONENT = 0.15

# Momentum tier: top decile by 3m momentum z
MOMENTUM_DECILE_FRAC = 0.10

TIER_BULLISH = "STRONG_BULLISH"
TIER_BEARISH = "STRONG_BEARISH"
TIER_MOMENTUM_LONG = "MOMENTUM_LONG"
TIER_CONFLUENCE_BULL = "CONFLUENCE_BULLISH"
TIER_CONFLUENCE_BEAR = "CONFLUENCE_BEARISH"
TIER_NONE = "NO_ALERT"


def _tier_of(row: pd.Series) -> str:
    c = row["composite"]
    fz, oz, iz = row["factor_z"], row["options_z"], row["insider_z"]

    # Tier 1 — hard bullish / bearish (original composite rule)
    if (c > BULL_THRESH and fz > COMPONENT_BULL and oz > COMPONENT_BULL
            and iz > COMPONENT_BULL):
        return TIER_BULLISH
    if (c < BEAR_THRESH and fz < COMPONENT_BEAR and oz < COMPONENT_BEAR
            and iz < COMPONENT_BEAR):
        return TIER_BEARISH

    # Tier 2 — momentum long (primary actionable tier)
    if bool(row.get("momentum_long", False)):
        return TIER_MOMENTUM_LONG

    # Tier 3 — softer confluence.
    # Rule: >= 2 of 3 components agree AND no component disagrees strongly.
    # The "no strong disagreement" veto preserves the diagram's original
    # spirit ("conflicting signals -> no alert"): two strong bulls with
    # one strong bear still lacks real consensus.
    pos_count = sum(x > CONFLUENCE_COMPONENT for x in (fz, oz, iz))
    neg_count = sum(x < -CONFLUENCE_COMPONENT for x in (fz, oz, iz))
    if c > CONFLUENCE_COMPOSITE and pos_count >= 2 and neg_count == 0:
        return TIER_CONFLUENCE_BULL
    if c < -CONFLUENCE_COMPOSITE and neg_count >= 2 and pos_count == 0:
        return TIER_CONFLUENCE_BEAR

    return TIER_NONE


def _rationale(row: pd.Series) -> str:
    parts = []
    if bool(row.get("momentum_long", False)):
        mz = row.get("momentum_3m_z")
        if mz is not None:
            parts.append(f"3m-mom=+{float(mz):.2f}")
        else:
            parts.append("3m-mom TOP")
    for label, z in (("factor", row["factor_z"]),
                     ("options", row["options_z"]),
                     ("insider", row["insider_z"])):
        if abs(z) >= 0.5:
            direction = "+" if z > 0 else "-"
            parts.append(f"{label}={direction}{abs(z):.2f}")
    if not parts:
        return "mixed / weak"
    return ", ".join(parts)


def build_alerts(as_of: date) -> pd.DataFrame:
    factor_df = load_factors(as_of)[["ticker", "factor_z", "momentum_3m_z"]]
    options_df = load_options_panel(as_of)[["ticker", "options_z"]]
    insider_df = load_insider_panel(as_of)[["ticker", "insider_z"]]

    merged = (
        factor_df
        .merge(options_df, on="ticker", how="outer")
        .merge(insider_df, on="ticker", how="outer")
        .fillna(0.0)
    )
    merged["composite"] = (
        FACTOR_W * merged["factor_z"]
        + OPTIONS_W * merged["options_z"]
        + INSIDER_W * merged["insider_z"]
    )

    # Flag top-decile 3-month momentum — primary tradeable signal
    n = len(merged)
    k = max(1, int(n * MOMENTUM_DECILE_FRAC))
    # nlargest(k).min() gives the kth-ranked value (the cut line)
    cut = merged["momentum_3m_z"].nlargest(k).min() if n else 0.0
    merged["momentum_long"] = merged["momentum_3m_z"] >= cut

    merged["tier"] = merged.apply(_tier_of, axis=1)
    merged["rationale"] = merged.apply(_rationale, axis=1)
    merged["as_of_date"] = as_of.isoformat()

    # Sort bullish composite first, then momentum_long, then bearish
    tier_order = {
        TIER_BULLISH: 0, TIER_MOMENTUM_LONG: 1, TIER_CONFLUENCE_BULL: 2,
        TIER_NONE: 3, TIER_CONFLUENCE_BEAR: 4, TIER_BEARISH: 5,
    }
    merged["_order"] = merged["tier"].map(tier_order).fillna(3)
    merged = merged.sort_values(
        by=["_order", "composite"], ascending=[True, False]
    ).drop(columns=["_order"]).reset_index(drop=True)

    return merged[[
        "ticker", "tier", "composite", "factor_z", "options_z", "insider_z",
        "momentum_3m_z", "momentum_long", "rationale", "as_of_date",
    ]]


def save_alerts(df: pd.DataFrame, as_of: date) -> Path:
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_DIR / f"{as_of.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_alerts(as_of: date) -> pd.DataFrame:
    path = ALERTS_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(path)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Build composite NDX 100 alerts.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    as_of = date.fromisoformat(args.date)
    df = build_alerts(as_of)
    save_alerts(df, as_of)
    tier_counts = df["tier"].value_counts().to_dict()
    logger.info("Alerts for %s: %s", as_of, tier_counts)
    print("\nTop 5 bulls:")
    print(df.head(5).to_string())
    print("\nTop 5 bears:")
    print(df.tail(5).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

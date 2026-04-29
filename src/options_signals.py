"""Snapshot-only options signal for the NDX 100 cross-section.

The data source is a yfinance option-chain snapshot, not OPRA tape. That means
we do not know trade direction, opening/closing status, or true option deltas.
The model therefore treats the chain as a robust, quality-gated proxy:

    Directional flow : call dollar share and call V/OI minus put V/OI
    Downside pressure: OTM put IV richer than OTM call IV
    Vol stress       : front ATM IV elevated versus back tenor / market vol

Each raw metric is converted to a robust cross-sectional percentile score in
[-1, +1]. Missing components are neutral. Low-quality snapshots are neutralized
to options_z = 0 instead of emitting false precision.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_options import load_all_chains_for_date
from src.data_prices import load_prices, load_vix
from src.universe import PROJECT_ROOT, load_universe

logger = logging.getLogger(__name__)

OPTIONS_SIG_DIR = PROJECT_ROOT / "data" / "options_signals"

_ATM_BAND = 0.15
_NEAR_EXPIRIES = 2
_IV_MIN = 0.05
_DTE_MIN = 7
_FRONT_TERM_DAYS = 30
_BACK_TERM_DAYS = 90
_TENOR_MAX_DISTANCE_DAYS = 45

_LOW_MIN_CONTRACTS = 80
_LOW_MIN_EXPIRIES = 2
_LOW_MIN_VOLUME_COVERAGE = 0.25
_LOW_MIN_IV_COVERAGE = 0.20
_LOW_MIN_QUOTE_SANITY = 0.95

_HIGH_MIN_CONTRACTS = 300
_HIGH_MIN_EXPIRIES = 5
_HIGH_MIN_VOLUME_COVERAGE = 0.60
_HIGH_MIN_IV_COVERAGE = 0.35
_HIGH_MIN_QUOTE_SANITY = 0.99

_NUMERIC_COLS = [
    "strike", "last_price", "bid", "ask", "mid", "volume",
    "open_interest", "implied_volatility",
]


def _latest_close(ticker: str, prices: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)]
    if sub.empty:
        return None
    return float(sub.sort_values("date").iloc[-1]["adj_close"])


def _empty_metrics() -> dict:
    return {
        "vol_oi_call": np.nan,
        "vol_oi_put": np.nan,
        "flow_ratio": np.nan,
        "iv_skew": np.nan,
        "iv_term": np.nan,
        "iv_rel_vix": np.nan,
        "contract_count": 0,
        "expiry_count": 0,
        "positive_volume_coverage": 0.0,
        "positive_oi_coverage": 0.0,
        "valid_iv_coverage": 0.0,
        "quote_sanity": 0.0,
        "options_coverage": 0.0,
        "options_quality": "LOW",
    }


def _coverage_ratio(mask: pd.Series) -> float:
    if mask.empty:
        return 0.0
    return float(mask.fillna(False).mean())


def _clip01(value: float) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _prepare_chain(chain: pd.DataFrame) -> pd.DataFrame:
    out = chain.copy()
    for col in _NUMERIC_COLS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "side" not in out.columns:
        out["side"] = ""
    if "expiry" not in out.columns:
        out["expiry"] = pd.NaT
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")

    if "as_of_date" in out.columns and out["as_of_date"].notna().any():
        as_of = pd.to_datetime(out["as_of_date"].dropna().iloc[0], errors="coerce")
    else:
        as_of = pd.NaT
    out["dte"] = (out["expiry"] - as_of).dt.days.clip(lower=0)
    return out


def _quality_from_stats(stats: dict) -> tuple[str, float]:
    contract_score = _clip01(stats["contract_count"] / _HIGH_MIN_CONTRACTS)
    expiry_score = _clip01(stats["expiry_count"] / _HIGH_MIN_EXPIRIES)
    volume_score = _clip01(stats["positive_volume_coverage"] / _HIGH_MIN_VOLUME_COVERAGE)
    oi_score = _clip01(stats["positive_oi_coverage"] / 0.15)
    iv_score = _clip01(stats["valid_iv_coverage"] / _HIGH_MIN_IV_COVERAGE)
    quote_score = _clip01(stats["quote_sanity"] / _HIGH_MIN_QUOTE_SANITY)
    coverage = (
        0.20 * contract_score
        + 0.15 * expiry_score
        + 0.20 * volume_score
        + 0.10 * oi_score
        + 0.20 * iv_score
        + 0.15 * quote_score
    )

    low = (
        stats["contract_count"] < _LOW_MIN_CONTRACTS
        or stats["expiry_count"] < _LOW_MIN_EXPIRIES
        or stats["positive_volume_coverage"] < _LOW_MIN_VOLUME_COVERAGE
        or stats["valid_iv_coverage"] < _LOW_MIN_IV_COVERAGE
        or stats["quote_sanity"] < _LOW_MIN_QUOTE_SANITY
    )
    high = (
        stats["contract_count"] >= _HIGH_MIN_CONTRACTS
        and stats["expiry_count"] >= _HIGH_MIN_EXPIRIES
        and stats["positive_volume_coverage"] >= _HIGH_MIN_VOLUME_COVERAGE
        and stats["valid_iv_coverage"] >= _HIGH_MIN_IV_COVERAGE
        and stats["quote_sanity"] >= _HIGH_MIN_QUOTE_SANITY
    )
    quality = "LOW" if low else "HIGH" if high else "MEDIUM"
    return quality, round(float(coverage), 4)


def _chain_quality_stats(chain: pd.DataFrame) -> dict:
    n = len(chain)
    stats = {
        "contract_count": int(n),
        "expiry_count": int(chain["expiry"].dropna().nunique()),
        "positive_volume_coverage": _coverage_ratio(chain["volume"] > 0),
        "positive_oi_coverage": _coverage_ratio(chain["open_interest"] > 0),
        "valid_iv_coverage": _coverage_ratio(chain["implied_volatility"] > _IV_MIN),
        "quote_sanity": _coverage_ratio(
            (chain["bid"] >= 0) & (chain["ask"] >= chain["bid"]) & (chain["mid"] >= 0)
        ),
    }
    quality, coverage = _quality_from_stats(stats)
    stats["options_quality"] = quality
    stats["options_coverage"] = coverage
    return stats


def _metrics_for_ticker(chain: pd.DataFrame, spot: float, vix: float) -> dict:
    """Compute snapshot-safe metrics for one ticker's chain."""
    out = _empty_metrics()
    if chain.empty or spot is None or spot <= 0:
        return out

    chain = _prepare_chain(chain)
    stats = _chain_quality_stats(chain)
    out.update(stats)

    band_all = chain[
        (chain["strike"] >= spot * (1 - _ATM_BAND))
        & (chain["strike"] <= spot * (1 + _ATM_BAND))
        & (chain["dte"] >= _DTE_MIN)
    ].copy()
    if band_all.empty:
        return out

    near_exps = sorted(band_all["expiry"].dropna().unique())[:_NEAR_EXPIRIES]
    near = band_all[band_all["expiry"].isin(near_exps)].copy()
    near_oi = near[near["open_interest"].fillna(0) > 0]

    for side, key in (("call", "vol_oi_call"), ("put", "vol_oi_put")):
        sub = near_oi[near_oi["side"] == side]
        vol_sum = sub["volume"].fillna(0).sum()
        oi_sum = sub["open_interest"].fillna(0).sum()
        out[key] = vol_sum / oi_sum if oi_sum > 0 else np.nan

    price = near["mid"].where(near["mid"] > 0, near["last_price"]).fillna(0.0)
    near = near.assign(dollar=near["volume"].fillna(0) * price)
    call_dollar = near.loc[near["side"] == "call", "dollar"].sum()
    total_dollar = near["dollar"].sum()
    out["flow_ratio"] = call_dollar / total_dollar if total_dollar > 0 else np.nan

    if near_exps:
        nearest_exp = near_exps[0]
        exp_chain = chain[chain["expiry"] == nearest_exp]
        calls = exp_chain[exp_chain["side"] == "call"].sort_values("strike")
        puts = exp_chain[exp_chain["side"] == "put"].sort_values("strike")
        call_iv = _closest_iv(calls, spot * 1.10)
        put_iv = _closest_iv(puts, spot * 0.90)
        if call_iv is not None and put_iv is not None:
            out["iv_skew"] = put_iv - call_iv

    front_iv = _atm_iv_at_tenor(chain, spot, _FRONT_TERM_DAYS)
    back_iv = _atm_iv_at_tenor(chain, spot, _BACK_TERM_DAYS)
    if front_iv and back_iv and back_iv > 0:
        out["iv_term"] = front_iv / back_iv
    # Existing VIX is a market-vol proxy for this snapshot-only model.
    if front_iv and vix and vix > 0:
        out["iv_rel_vix"] = front_iv / (vix / 100.0)

    return out


def _closest_iv(side_df: pd.DataFrame, target_strike: float) -> float | None:
    valid = side_df[side_df["implied_volatility"] > _IV_MIN]
    if valid.empty:
        return None
    idx = (valid["strike"] - target_strike).abs().idxmin()
    iv = valid.loc[idx, "implied_volatility"]
    return float(iv) if pd.notna(iv) else None


def _atm_iv_at_tenor(chain: pd.DataFrame, spot: float, target_days: int) -> float | None:
    """ATM IV at a target tenor, using the nearest valid call/put quotes."""
    if chain.empty or spot is None or spot <= 0:
        return None
    chain = chain.copy()
    if "dte" not in chain.columns:
        chain = _prepare_chain(chain)
    else:
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain["implied_volatility"] = pd.to_numeric(chain["implied_volatility"], errors="coerce")

    chain = chain[
        (chain["dte"] >= _DTE_MIN)
        & chain["strike"].notna()
        & chain["implied_volatility"].notna()
    ]
    if chain.empty:
        return None

    exp_dtes = pd.Series(sorted(chain["dte"].dropna().unique()))
    exp_dtes = exp_dtes[(exp_dtes - target_days).abs() <= _TENOR_MAX_DISTANCE_DAYS]
    if exp_dtes.empty:
        return None

    for target_exp in sorted(exp_dtes, key=lambda d: abs(d - target_days)):
        sub = chain[chain["dte"] == target_exp]
        sub = sub[
            (sub["strike"] >= spot * (1 - _ATM_BAND))
            & (sub["strike"] <= spot * (1 + _ATM_BAND))
            & (sub["implied_volatility"] > _IV_MIN)
        ]
        if sub.empty:
            continue

        ivs: list[float] = []
        for side in ("call", "put"):
            side_df = sub[sub["side"] == side]
            if side_df.empty:
                continue
            idx = (side_df["strike"] - spot).abs().idxmin()
            iv = side_df.loc[idx, "implied_volatility"]
            if pd.notna(iv):
                ivs.append(float(iv))
        if ivs:
            return float(np.mean(ivs))

    return None


def _percentile_score(s: pd.Series, *, higher_is_bullish: bool = True) -> pd.Series:
    clean = s.replace([np.inf, -np.inf], np.nan)
    valid = clean.dropna()
    if len(valid) < 2 or valid.nunique(dropna=True) < 2:
        return pd.Series(0.0, index=s.index)
    score = (clean.rank(method="average", pct=True) - 0.5) * 2.0
    score = score.fillna(0.0).clip(-1.0, 1.0)
    return score if higher_is_bullish else -score


def _score_options_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Attach robust sub-scores and quality-gated options_z."""
    df = df.copy()
    if df.empty:
        return df

    if "options_quality" not in df.columns:
        df["options_quality"] = "LOW"
    for col in [
        "vol_oi_call", "vol_oi_put", "flow_ratio", "iv_skew",
        "iv_term", "iv_rel_vix",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    has_voi = df["vol_oi_call"].notna() | df["vol_oi_put"].notna()
    df["voi_spread"] = df["vol_oi_call"].fillna(0.0) - df["vol_oi_put"].fillna(0.0)
    df.loc[~has_voi, "voi_spread"] = np.nan

    flow_ratio_score = _percentile_score(df["flow_ratio"])
    voi_spread_score = _percentile_score(df["voi_spread"])
    df["flow_score"] = (0.65 * flow_ratio_score + 0.35 * voi_spread_score).clip(-1.0, 1.0)

    df["skew_score"] = _percentile_score(df["iv_skew"], higher_is_bullish=False)

    term_score = _percentile_score(df["iv_term"], higher_is_bullish=False)
    relvix_score = _percentile_score(df["iv_rel_vix"], higher_is_bullish=False)
    df["vol_stress_score"] = (0.55 * term_score + 0.45 * relvix_score).clip(-1.0, 1.0)

    df["options_z_raw"] = (
        0.50 * df["flow_score"]
        + 0.30 * df["skew_score"]
        + 0.20 * df["vol_stress_score"]
    ).clip(-1.0, 1.0)
    df["options_z"] = df["options_z_raw"].where(df["options_quality"] != "LOW", 0.0)
    return df


def compute_options_panel(as_of: date) -> pd.DataFrame:
    """Build the per-ticker options metrics + snapshot-only composite score."""
    universe = load_universe()
    tickers = universe["ticker"].tolist()

    chains = load_all_chains_for_date(as_of)
    if chains.empty:
        raise FileNotFoundError(
            f"No option chains snapshotted for {as_of}. "
            f"Run `python3 -m src.data_options --date {as_of}` first."
        )

    prices = load_prices()
    prices["date"] = pd.to_datetime(prices["date"])
    as_of_ts = pd.Timestamp(as_of)

    vix_df = load_vix()
    vix_df["date"] = pd.to_datetime(vix_df["date"])
    vix_slice = vix_df[vix_df["date"].dt.date <= as_of]
    vix_today = float(vix_slice.iloc[-1]["close"]) if not vix_slice.empty else np.nan

    rows = []
    for t in tickers:
        sub = chains[chains["ticker"] == t]
        spot = _latest_close(t, prices, as_of_ts)
        m = _metrics_for_ticker(sub, spot, vix_today)
        m["ticker"] = t
        rows.append(m)

    df = _score_options_panel(pd.DataFrame(rows))
    df["as_of_date"] = as_of.isoformat()

    preferred = [
        "ticker", "options_z", "options_z_raw", "options_quality", "options_coverage",
        "flow_score", "skew_score", "vol_stress_score",
        "vol_oi_call", "vol_oi_put", "flow_ratio", "iv_skew", "iv_term", "iv_rel_vix",
        "contract_count", "expiry_count", "positive_volume_coverage",
        "positive_oi_coverage", "valid_iv_coverage", "quote_sanity",
        "as_of_date",
    ]
    return df[[c for c in preferred if c in df.columns]]


def save_options_panel(df: pd.DataFrame, as_of: date) -> Path:
    OPTIONS_SIG_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIONS_SIG_DIR / f"{as_of.isoformat()}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_options_panel(as_of: date) -> pd.DataFrame:
    path = OPTIONS_SIG_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(path)


def main() -> int:
    from src.trading_day import last_completed_trading_day
    parser = argparse.ArgumentParser(description="Compute options signal panel.")
    parser.add_argument("--date", default=last_completed_trading_day().isoformat())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    as_of = date.fromisoformat(args.date)
    df = compute_options_panel(as_of)
    save_options_panel(df, as_of)
    print(df.sort_values("options_z", ascending=False).head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

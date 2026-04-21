"""Single-snapshot options signal for NDX 100 cross-section.

Six metrics computed from today's yfinance option chain and z-scored
cross-sectionally across the universe. No historical OI required.

    Band     : ATM +/- 15 % strikes, nearest two expiries
    Filters  : DTE >= 7 (skip near-expiry noise)
               OI > 0 at the contract level (drop never-traded strikes)
               IV > 5 % for the IV metrics (drop stale / zero quotes)

    vol_oi_call : sum call volume / sum call OI   (fresh positioning, calls)
    vol_oi_put  : sum put volume  / sum put OI    (fresh positioning, puts)
    flow_ratio  : dollar-weighted call vol / total vol
    iv_skew     : 25D-put IV - 25D-call IV (nearest-strike OTM proxy)
    iv_term     : 30d ATM IV / 90d ATM IV  (term structure inversion)
    iv_rel_vix  : stock ATM IV / VIX       (idio vol vs market)

Composite options_score (signed, gridsearchable weights):
    0.30 * z(flow_ratio)
  + 0.25 * [ z(vol_oi_call) - z(vol_oi_put) ]
  - 0.20 * z(iv_skew)
  - 0.15 * z(iv_term)
  + 0.10 * -|z(iv_rel_vix)| * sign(flow_ratio - 0.5)
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

# Tuned after diagnose_issues.py inspection of the 5-ticker smoke test:
# ATM +/-15% keeps ~300-500 rows per ticker (enough signal) while cutting
# deep-OTM noise; IV floor 5% drops the ~half of quotes that are stale;
# DTE >= 7 drops the weekly near-expiry strip whose OI was confirmed 0
# across our smoke-test set.
_ATM_BAND = 0.15
_NEAR_EXPIRIES = 2
_IV_MIN = 0.05
_DTE_MIN = 7
_FRONT_TERM_DAYS = 30
_BACK_TERM_DAYS = 90


def _latest_close(ticker: str, prices: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    sub = prices[(prices["ticker"] == ticker) & (prices["date"] <= as_of)]
    if sub.empty:
        return None
    return float(sub.sort_values("date").iloc[-1]["adj_close"])


def _metrics_for_ticker(chain: pd.DataFrame, spot: float, vix: float) -> dict:
    """Compute the six metrics for one ticker's chain. Returns NaN where undefined."""
    out = {
        "vol_oi_call": np.nan,
        "vol_oi_put": np.nan,
        "flow_ratio": np.nan,
        "iv_skew": np.nan,
        "iv_term": np.nan,
        "iv_rel_vix": np.nan,
    }
    if chain.empty or spot is None or spot <= 0:
        return out

    chain = chain.copy()
    chain["expiry"] = pd.to_datetime(chain["expiry"])
    as_of = pd.to_datetime(chain["as_of_date"].iloc[0])
    chain["dte"] = (chain["expiry"] - as_of).dt.days.clip(lower=0)

    # ── ATM band + nearest N expiries for V/OI and flow ──
    # Filter: drop DTE<7 weeklies and OI=0 strikes (both confirmed as noise
    # by diagnose_issues.py). Keeps near-ATM, liquid, monthly-or-later only.
    band = chain[
        (chain["strike"] >= spot * (1 - _ATM_BAND))
        & (chain["strike"] <= spot * (1 + _ATM_BAND))
        & (chain["dte"] >= _DTE_MIN)
        & (chain["open_interest"].fillna(0) > 0)
    ]
    near_exps = sorted(band["expiry"].unique())[:_NEAR_EXPIRIES]
    near = band[band["expiry"].isin(near_exps)]

    for side, key in (("call", "vol_oi_call"), ("put", "vol_oi_put")):
        sub = near[near["side"] == side]
        vol_sum = sub["volume"].fillna(0).sum()
        oi_sum = sub["open_interest"].fillna(0).sum()
        out[key] = vol_sum / oi_sum if oi_sum > 0 else np.nan

    # Flow ratio: dollar-weighted
    near = near.assign(
        dollar=near["volume"].fillna(0) * near["mid"].fillna(0.0),
    )
    call_dollar = near.loc[near["side"] == "call", "dollar"].sum()
    total_dollar = near["dollar"].sum()
    out["flow_ratio"] = call_dollar / total_dollar if total_dollar > 0 else np.nan

    # ── IV skew: nearest 25-delta put IV - 25-delta call IV (OTM proxy) ──
    # 25-delta ~ one strike above/below ATM in the nearest expiry for liquid names.
    # Proxy: for the nearest expiry, take the OTM strike closest to spot * (1 +/- 0.10).
    if near_exps:
        nearest_exp = near_exps[0]
        exp_chain = chain[chain["expiry"] == nearest_exp]
        calls = exp_chain[exp_chain["side"] == "call"].sort_values("strike")
        puts = exp_chain[exp_chain["side"] == "put"].sort_values("strike")
        otm_call_target = spot * 1.10
        otm_put_target = spot * 0.90
        call_iv = _closest_iv(calls, otm_call_target)
        put_iv = _closest_iv(puts, otm_put_target)
        if call_iv is not None and put_iv is not None:
            out["iv_skew"] = put_iv - call_iv

    # ── IV term: front ATM IV / back ATM IV ──
    front_iv = _atm_iv_at_tenor(chain, spot, _FRONT_TERM_DAYS)
    back_iv = _atm_iv_at_tenor(chain, spot, _BACK_TERM_DAYS)
    if front_iv and back_iv and back_iv > 0:
        out["iv_term"] = front_iv / back_iv
    # IV vs VIX
    if front_iv and vix and vix > 0:
        out["iv_rel_vix"] = front_iv / (vix / 100.0)

    return out


def _closest_iv(side_df: pd.DataFrame, target_strike: float) -> float | None:
    # Only consider strikes with a sensible IV (filters stale / zero quotes).
    valid = side_df[side_df["implied_volatility"] > _IV_MIN]
    if valid.empty:
        return None
    idx = (valid["strike"] - target_strike).abs().idxmin()
    iv = valid.loc[idx, "implied_volatility"]
    return float(iv) if pd.notna(iv) else None


def _atm_iv_at_tenor(chain: pd.DataFrame, spot: float, target_days: int) -> float | None:
    """ATM IV at tenor closest to target_days. Average of nearest call + put strike IV."""
    if chain.empty:
        return None
    chain = chain.copy()
    if "dte" not in chain.columns:
        chain["expiry"] = pd.to_datetime(chain["expiry"])
        as_of = pd.to_datetime(chain["as_of_date"].iloc[0])
        chain["dte"] = (chain["expiry"] - as_of).dt.days.clip(lower=0)
    exp_dtes = sorted(chain["dte"].unique())
    if not exp_dtes:
        return None
    target_exp = min(exp_dtes, key=lambda d: abs(d - target_days))
    sub = chain[chain["dte"] == target_exp]
    atm_strike_idx = (sub["strike"] - spot).abs().idxmin()
    atm_strike = sub.loc[atm_strike_idx, "strike"]
    near_atm = sub[sub["strike"] == atm_strike]
    ivs = near_atm["implied_volatility"].dropna()
    # Filter stale / zero-IV quotes; matches _closest_iv + _IV_MIN.
    ivs = ivs[ivs > _IV_MIN]
    if ivs.empty:
        return None
    return float(ivs.mean())


def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def compute_options_panel(as_of: date) -> pd.DataFrame:
    """Build the per-ticker options metrics + composite options_z."""
    universe = load_universe()
    tickers = universe["ticker"].tolist()

    chains = load_all_chains_for_date(as_of)
    if chains.empty:
        raise FileNotFoundError(
            f"No option chains snapshotted for {as_of}. "
            "Run `python3 -m src.data_options --date {as_of}` first."
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

    df = pd.DataFrame(rows)
    # Cross-sectional z-scores. NaN rows (tickers with thin/no chain after
    # Fix B's filters) are imputed to 0 per-component BEFORE compositing so
    # a ticker with partial data contributes partial signal rather than
    # zeroing out the whole options_z via NaN arithmetic propagation.
    df["z_flow"] = _zscore(df["flow_ratio"]).fillna(0.0)
    df["z_voc"] = _zscore(df["vol_oi_call"]).fillna(0.0)
    df["z_vop"] = _zscore(df["vol_oi_put"]).fillna(0.0)
    df["z_skew"] = _zscore(df["iv_skew"]).fillna(0.0)
    df["z_term"] = _zscore(df["iv_term"]).fillna(0.0)
    df["z_relvix"] = _zscore(df["iv_rel_vix"]).fillna(0.0)

    sign_flow = np.sign(df["flow_ratio"].fillna(0.5) - 0.5)
    df["options_z"] = (
        0.30 * df["z_flow"]
        + 0.25 * (df["z_voc"] - df["z_vop"])
        - 0.20 * df["z_skew"]
        - 0.15 * df["z_term"]
        + 0.10 * (-df["z_relvix"].abs()) * sign_flow
    )
    df["as_of_date"] = as_of.isoformat()
    return df


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

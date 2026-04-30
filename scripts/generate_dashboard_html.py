"""Generate NDX Alert Desk.html with real pipeline data baked in.

Rebuilds the standalone HTML preview by replacing the synthetic JS
data-generator (the IIFE in the template) with real `window.DATA`
records sourced from the daily pipeline's parquet outputs:

    alerts/{date}.parquet         -> alert rows (tier, composite, z-scores)
    factors/{date}.parquet        -> per-factor z-score breakdown
    prices.parquet                -> 1y adj_close per ticker
    options_signals/{date}.parquet -> options score, sub-scores, quality
    chains/{date}/{ticker}.parquet -> option snapshot surface per ticker
    form4/{ticker}.parquet        -> last 20 Form 4 transactions per ticker
    universe.parquet              -> ticker + company + sector + market cap
    risk.py / performance_risk.py -> Performance & Risk metrics + validation gate

The resulting HTML is fully self-contained (still opens by double-click)
but reflects the pipeline state as of the specified date.

Run:
    python3 scripts/generate_dashboard_html.py
    python3 scripts/generate_dashboard_html.py --date 2026-04-29
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prices import latest_adj_close_on_or_before  # noqa: E402
from src.performance_risk import options_history_validation_gate  # noqa: E402
from src.risk import risk_report_detail  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
ALERTS_DIR = DATA_DIR / "alerts"
FACTORS_DIR = DATA_DIR / "factors"
OPT_SIG_DIR = DATA_DIR / "options_signals"
CHAINS_DIR = DATA_DIR / "chains"
FORM4_DIR = DATA_DIR / "form4"
UNIVERSE_PARQUET = DATA_DIR / "universe.parquet"
PRICES_PARQUET = DATA_DIR / "prices.parquet"

HTML_PATH = PROJECT_ROOT / "NDX Alert Desk.html"

ATM_BAND = 0.12
MAX_EXPIRIES = 5
PRICE_DAYS = 252
FORM4_ROWS = 20


def _round(v, n=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), n)


def _series_records(series: pd.Series, decimals: int = 6) -> list[dict]:
    if series is None or series.empty:
        return []
    out = []
    for idx, value in series.dropna().items():
        out.append({
            "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
            "value": _round(value, decimals),
        })
    return out


def _summary_records(summary: dict) -> dict:
    out = {}
    for key, value in summary.items():
        if isinstance(value, (float, np.floating)):
            out[key] = _round(value, 6)
        elif isinstance(value, (int, np.integer)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _latest_available_date() -> date:
    if not ALERTS_DIR.exists():
        raise SystemExit(f"No alerts dir at {ALERTS_DIR}. Run `python3 run_daily.py` first.")
    dates = []
    for p in ALERTS_DIR.glob("*.parquet"):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            continue
    if not dates:
        raise SystemExit(f"No alert parquets in {ALERTS_DIR}.")
    return max(dates)


# ── Data block builders ───────────────────────────────────────────────

def build_universe_records() -> list[dict]:
    """Universe from parquet (ticker, company, sector, market_cap in BILLIONS)."""
    u = pd.read_parquet(UNIVERSE_PARQUET)
    out = []
    for r in u.itertuples():
        mcap = float(r.market_cap or 0)
        out.append({
            "ticker": str(r.ticker),
            "company": str(r.company or ""),
            "sector": str(r.sector or ""),
            "market_cap": _round(mcap / 1e9, 1) if mcap > 0 else 0.0,
        })
    return out


def build_alerts_records(as_of: date) -> list[dict]:
    """Alerts + per-factor z-scores merged into a single row per ticker."""
    alerts = pd.read_parquet(ALERTS_DIR / f"{as_of.isoformat()}.parquet")
    factors = pd.read_parquet(FACTORS_DIR / f"{as_of.isoformat()}.parquet").set_index("ticker")
    opt_path = OPT_SIG_DIR / f"{as_of.isoformat()}.parquet"
    options = pd.read_parquet(opt_path).set_index("ticker") if opt_path.exists() else pd.DataFrame()
    uni = pd.read_parquet(UNIVERSE_PARQUET).set_index("ticker")

    rows = []
    for _, a in alerts.iterrows():
        t = str(a["ticker"])
        f = factors.loc[t] if t in factors.index else None
        o = options.loc[t] if not options.empty and t in options.index else None
        u = uni.loc[t] if t in uni.index else None
        rows.append({
            "ticker": t,
            "company": str(u["company"]) if u is not None else "",
            "sector": str(u["sector"]) if u is not None else "",
            "market_cap": _round(float(u["market_cap"] or 0) / 1e9, 1) if u is not None else 0.0,
            "momentum_z": _round(f["momentum_z"]) if f is not None else 0.0,
            "value_z": _round(f["value_z"]) if f is not None else 0.0,
            "quality_z": _round(f["quality_z"]) if f is not None else 0.0,
            "lowvol_z": _round(f["lowvol_z"]) if f is not None else 0.0,
            "factor_z": _round(a["factor_z"]),
            "options_z": _round(a["options_z"]),
            "options_z_raw": _round(o["options_z_raw"]) if o is not None and "options_z_raw" in o else None,
            "options_quality": str(o["options_quality"]) if o is not None and "options_quality" in o else "UNKNOWN",
            "options_coverage": _round(o["options_coverage"]) if o is not None and "options_coverage" in o else None,
            "flow_score": _round(o["flow_score"]) if o is not None and "flow_score" in o else None,
            "skew_score": _round(o["skew_score"]) if o is not None and "skew_score" in o else None,
            "vol_stress_score": _round(o["vol_stress_score"]) if o is not None and "vol_stress_score" in o else None,
            "insider_z": _round(a["insider_z"]),
            "momentum_3m_z": _round(a["momentum_3m_z"]),
            "composite": _round(a["composite"]),
            "tier": str(a["tier"]),
            "rationale": str(a["rationale"]),
        })
    return rows


def build_prices_map(as_of: date) -> dict[str, list[dict]]:
    """Per-ticker last-252-day adj_close series."""
    p = pd.read_parquet(PRICES_PARQUET)
    p["date"] = pd.to_datetime(p["date"])
    out: dict[str, list[dict]] = {}
    for t, grp in p.groupby("ticker"):
        sub = grp[grp["date"] <= pd.Timestamp(as_of)].sort_values("date").tail(PRICE_DAYS)
        out[str(t)] = [
            {"date": d.strftime("%Y-%m-%d"), "close": _round(c, 2)}
            for d, c in zip(sub["date"], sub["adj_close"])
            if pd.notna(c) and c is not None
        ]
    return out


def build_chain_map(as_of: date) -> dict[str, dict]:
    """Per-ticker options chain reduced to {expiries, strikes} structure.

    The HTML renderer expects:
        expiries: [{dte, label}]
        metric:   "V/OI" or "Volume"
        strikes:  [{strike, voi: [v1, v2, ...]}]   # one metric value per expiry

    Aggregates call+put volume and OI per strike+expiry, then computes
    V/OI for the ATM +/- 12% band across the next 5 expiries. If Yahoo has
    too little usable open interest, falls back to volume for display.
    """
    day_dir = CHAINS_DIR / as_of.isoformat()
    spots = _latest_spot_map(as_of)
    out: dict[str, dict] = {}
    for parq in sorted(day_dir.glob("*.parquet")):
        ticker = parq.stem
        spot = spots.get(ticker)
        if spot is None or spot <= 0:
            continue
        try:
            df = pd.read_parquet(parq)
        except Exception:
            continue
        if df.empty:
            continue
        df["expiry"] = pd.to_datetime(df["expiry"])
        df["as_of"] = pd.to_datetime(df["as_of_date"])
        df["dte"] = (df["expiry"] - df["as_of"]).dt.days.clip(lower=0)
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)

        band = df[
            (df["strike"] >= spot * (1 - ATM_BAND))
            & (df["strike"] <= spot * (1 + ATM_BAND))
        ].copy()
        if band.empty:
            continue

        # Aggregate call+put volume and OI per strike x expiry
        agg = (
            band.groupby(["strike", "expiry"], as_index=False)
            .agg(dte=("dte", "first"),
                 volume=("volume", lambda s: s.fillna(0).sum()),
                 open_interest=("open_interest", lambda s: s.fillna(0).sum()))
        )
        agg["voi"] = np.where(agg["open_interest"] > 0, agg["volume"] / agg["open_interest"], np.nan)

        metric_col = "voi"
        metric_label = "V/OI"
        decimals = 2
        metric_rows = agg[agg["voi"].notna() & (agg["voi"] > 0)].copy()
        if len(metric_rows) < 10 or metric_rows["expiry"].nunique() < 2:
            metric_col = "volume"
            metric_label = "Volume"
            decimals = 0
            metric_rows = agg[agg["volume"] > 0].copy()
        if metric_rows.empty:
            continue

        exp_keep = (
            metric_rows[["expiry", "dte"]].drop_duplicates()
            .sort_values("dte")
            .head(MAX_EXPIRIES)
        )
        if exp_keep.empty:
            continue

        expiries_js = [
            {"dte": int(r.dte), "label": r.expiry.strftime("%b %d")}
            for r in exp_keep.itertuples()
        ]
        exp_list = exp_keep["expiry"].tolist()
        metric_rows = metric_rows[metric_rows["expiry"].isin(exp_list)]

        agg = (
            metric_rows.groupby(["strike", "expiry"], as_index=False)
            .agg(volume=("volume", lambda s: s.fillna(0).sum()),
                 open_interest=("open_interest", lambda s: s.fillna(0).sum()),
                 voi=("voi", "mean"))
        )

        strike_rows = []
        for strike, grp in agg.groupby("strike"):
            lookup = dict(zip(grp["expiry"], grp[metric_col]))
            voi_series = []
            for e in exp_list:
                value = lookup.get(e)
                if value is None or pd.isna(value) or value <= 0:
                    voi_series.append(None)
                else:
                    voi_series.append(_round(float(value), decimals))
            if any(v is not None for v in voi_series):
                strike_rows.append({"strike": _round(float(strike), 0), "voi": voi_series})
        strike_rows.sort(key=lambda x: x["strike"])
        if not strike_rows:
            continue

        out[ticker] = {
            "metric": metric_label,
            "decimals": decimals,
            "expiries": expiries_js,
            "strikes": strike_rows,
        }
    return out


def _latest_spot_map(as_of: date) -> dict[str, float]:
    """Latest adj_close on or before as_of, used as the chain spot anchor."""
    p = pd.read_parquet(PRICES_PARQUET)
    p["date"] = pd.to_datetime(p["date"])
    spots: dict[str, float] = {}
    for ticker in sorted(p["ticker"].dropna().unique()):
        spot = latest_adj_close_on_or_before(p, str(ticker), as_of)
        if spot is not None:
            spots[str(ticker)] = spot
    return spots


def build_form4_map() -> dict[str, list[dict]]:
    """Per-ticker last 20 Form 4 transactions, sorted newest first."""
    if not FORM4_DIR.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for parq in FORM4_DIR.glob("*.parquet"):
        t = parq.stem
        try:
            df = pd.read_parquet(parq)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.sort_values("transaction_date", ascending=False).head(FORM4_ROWS)
        rows = []
        for _, r in df.iterrows():
            shares = float(r["shares"]) if pd.notna(r.get("shares")) else 0.0
            value = float(r["value"]) if pd.notna(r.get("value")) else 0.0
            weight = int(r["signal_weight"]) if pd.notna(r.get("signal_weight")) else 0
            rows.append({
                "date": str(r["transaction_date"])[:10],
                "insider": str(r.get("insider_name", "")),
                "position": str(r.get("position", "") or ""),
                "tx_code": str(r.get("tx_code", "")),
                "signal_label": str(r.get("signal_label", "")),
                "signal_weight": weight,
                "shares": _round(shares, 0),
                "value": _round(value, 0),
            })
        out[t] = rows
    return out


def available_dates() -> list[str]:
    """All dates for which we have a written alerts parquet, newest first."""
    if not ALERTS_DIR.exists():
        return []
    dates = []
    for p in ALERTS_DIR.glob("*.parquet"):
        try:
            dates.append(date.fromisoformat(p.stem))
        except ValueError:
            continue
    return [d.isoformat() for d in sorted(dates, reverse=True)]


def _empty_performance_payload(note: str) -> dict:
    return {"summary": {"note": note}, "equity": [], "drawdown": []}


def build_performance_risk(as_of: date) -> dict:
    """Alert-book risk and options-history validation payload."""
    try:
        risk = risk_report_detail(as_of, horizon_days=20)
        risk_payload = {
            "summary": _summary_records(risk["summary"]),
            "equity": _series_records(risk["equity"]),
            "drawdown": _series_records(risk["drawdown"]),
        }
    except Exception as exc:
        risk_payload = _empty_performance_payload(str(exc))

    validation = options_history_validation_gate(as_of, project_root=PROJECT_ROOT)
    return {"risk": risk_payload, "validation": validation}


def build_performance_risk_map(date_strs: list[str]) -> dict[str, dict]:
    """Performance/risk payloads keyed by each selectable as-of date."""
    out: dict[str, dict] = {}
    for date_str in date_strs:
        try:
            out[date_str] = build_performance_risk(date.fromisoformat(date_str))
        except ValueError:
            continue
    return out


# ── HTML patch ────────────────────────────────────────────────────────

def build_real_data_iife(
    as_of: date,
    alerts: list[dict],
    prices: dict[str, list[dict]],
    chains: dict[str, dict],
    form4: dict[str, list[dict]],
    performance_risk_by_date: dict[str, dict],
    dates_available: list[str],
) -> str:
    """The replacement IIFE that wires real parquet data into window.DATA."""
    payload = {
        "AS_OF": as_of.isoformat(),
        "AVAILABLE_DATES": dates_available or [as_of.isoformat()],
        "ALERTS_BY_DATE": {as_of.isoformat(): alerts},
        "PRICES": prices,
        "CHAINS": chains,
        "FORM4": form4,
        "PERFORMANCE_RISK": performance_risk_by_date,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""(function () {{
  // Real data baked in by scripts/generate_dashboard_html.py.
  // Regenerate after each daily pipeline run to refresh the preview.
  const BAKED = {payload_json};

  const EMPTY_CHAIN = {{expiries: [], strikes: []}};

  window.DATA = {{
    AS_OF: BAKED.AS_OF,
    AVAILABLE_DATES: BAKED.AVAILABLE_DATES,
    async ready() {{
      await window.NDX_UNIVERSE_READY;
      return window.NDX_UNIVERSE;
    }},
    getAlerts(dateStr) {{
      return BAKED.ALERTS_BY_DATE[dateStr] || BAKED.ALERTS_BY_DATE[BAKED.AS_OF] || [];
    }},
    getPrices(ticker, _sector, _mcap) {{
      return BAKED.PRICES[ticker] || [];
    }},
    getChain(ticker, _spot) {{
      return BAKED.CHAINS[ticker] || EMPTY_CHAIN;
    }},
    getForm4(ticker, _insider_z) {{
      return BAKED.FORM4[ticker] || [];
    }},
    getPerformanceRisk(dateStr) {{
      const key = dateStr || BAKED.AS_OF;
      return BAKED.PERFORMANCE_RISK[key] || null;
    }},
  }};
}})();
"""


_IIFE_START = "(function () {\n  function mulberry32"
_IIFE_END = "})();"
_BAKED_IIFE_MARKER = "// Real data baked in by scripts/generate_dashboard_html.py."


def patch_html(src: str, new_iife: str) -> str:
    """Replace the synthetic data IIFE with the baked-data IIFE."""
    start = src.find(_IIFE_START)
    if start == -1:
        marker = src.find(_BAKED_IIFE_MARKER)
        if marker == -1:
            raise RuntimeError(
                "Could not locate either the synthetic IIFE or baked-data marker "
                "in the HTML template."
            )
        start = src.rfind("(function () {", 0, marker)
        if start == -1:
            raise RuntimeError("Could not locate baked-data IIFE start.")
    # Find the first `})();` after the start that's balanced with the outer
    # IIFE. The synthetic IIFE ends at the first `})();` at column 0 in this
    # template, so a naive find from `start` works.
    end = src.find(_IIFE_END, start)
    if end == -1:
        raise RuntimeError("Could not locate the IIFE end marker after start.")
    end += len(_IIFE_END)
    return src[:start] + new_iife.strip() + src[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None,
                        help="As-of date (default: latest alerts parquet).")
    args = parser.parse_args()

    as_of = (
        date.fromisoformat(args.date) if args.date else _latest_available_date()
    )
    print(f"Generating dashboard HTML for as_of={as_of}")

    alerts = build_alerts_records(as_of)
    print(f"  alerts: {len(alerts)} rows")
    prices = build_prices_map(as_of)
    print(f"  prices: {len(prices)} tickers, "
          f"{sum(len(v) for v in prices.values())} total price points")
    chains = build_chain_map(as_of)
    print(f"  chains: {len(chains)} tickers with ATM band data")
    form4 = build_form4_map()
    print(f"  form4:  {len(form4)} tickers, "
          f"{sum(len(v) for v in form4.values())} transactions")
    # The standalone HTML is a single baked snapshot. Streamlit is the dynamic
    # multi-date surface; exposing unbaked dates here would mix old selections
    # with the current snapshot's prices, chains, and Form 4 detail.
    dates_baked = [as_of.isoformat()]
    performance_risk_by_date = build_performance_risk_map(dates_baked)
    performance_risk = performance_risk_by_date.get(as_of.isoformat())
    if performance_risk is None:
        performance_risk = build_performance_risk(as_of)
        performance_risk_by_date[as_of.isoformat()] = performance_risk
    risk_note = performance_risk["risk"]["summary"].get("note")
    gate = performance_risk["validation"]
    print(
        "  performance/risk: "
        f"risk={'note' if risk_note else 'ok'}, "
        f"options_gate={gate['status']} "
        f"({gate['usable_days']}/{gate['min_days']} usable days)"
    )

    src_html = HTML_PATH.read_text()
    new_iife = build_real_data_iife(
        as_of=as_of,
        alerts=alerts,
        prices=prices,
        chains=chains,
        form4=form4,
        performance_risk_by_date=performance_risk_by_date,
        dates_available=dates_baked,
    )
    new_html = patch_html(src_html, new_iife)
    HTML_PATH.write_text(new_html)

    size_kb = HTML_PATH.stat().st_size / 1024
    print(f"\nWrote {HTML_PATH.name} ({size_kb:,.0f} KB, "
          f"{len(new_html.splitlines()):,} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

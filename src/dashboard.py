"""NDX 100 alert dashboard — institutional single page.

Deep-Trading aesthetic: Fraunces serif masthead, IBM Plex Sans body,
JetBrains Mono for all numerics, aged-gold accent on a blue-black canvas.
Loads parquet outputs from the daily pipeline and renders three sections:

    01  ALERT LEDGER          compact table, tier chips, micro-bar z-scores
    02  TICKER DEEP DIVE      2x2 grid: price | factors | options | insiders
    03  UNIVERSE CONSTELLATION scatter of factor_z vs options_z, color=insider_z

Run:  streamlit run src/dashboard.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_prices import latest_adj_close_on_or_before
from src.lifted.display import nice_columns
from src.lifted.ui_style import (
    CHART_PALETTE, DIVERGING_SCALE, SEQUENTIAL_GOLD,
    ChartHeight, Colors, fmt_dollar, fmt_z, inject_css,
    plotly_layout, tier_chip_html,
)
from src.newsletter_export import (
    export_newsletter, load_newsletter_context, render_newsletter_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERTS_DIR = PROJECT_ROOT / "data" / "alerts"
PRICES_PARQUET = PROJECT_ROOT / "data" / "prices.parquet"
FACTORS_DIR = PROJECT_ROOT / "data" / "factors"
OPT_SIG_DIR = PROJECT_ROOT / "data" / "options_signals"
INS_SIG_DIR = PROJECT_ROOT / "data" / "insider_signals"
CHAINS_ROOT = PROJECT_ROOT / "data" / "chains"
FORM4_DIR = PROJECT_ROOT / "data" / "form4"
UNIVERSE_PARQUET = PROJECT_ROOT / "data" / "universe.parquet"
EXPORT_ROOT = PROJECT_ROOT / "exports" / "newsletters"
REFRESH_TIMEOUT_SECONDS = 90 * 60


st.set_page_config(page_title="NDX Alert Desk", layout="wide",
                   page_icon="🜚", initial_sidebar_state="collapsed")
inject_css()


# ── Data loaders (UI-context caching is OK per skill) ────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _load_alerts(as_of: date) -> pd.DataFrame:
    return pd.read_parquet(ALERTS_DIR / f"{as_of.isoformat()}.parquet")


@st.cache_data(ttl=300, show_spinner=False)
def _load_prices() -> pd.DataFrame:
    if not PRICES_PARQUET.exists():
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])
    df = pd.read_parquet(PRICES_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_factor(as_of: date) -> pd.DataFrame:
    p = FACTORS_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _load_options_panel(as_of: date) -> pd.DataFrame:
    p = OPT_SIG_DIR / f"{as_of.isoformat()}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _load_chain(ticker: str, as_of: date) -> pd.DataFrame:
    p = CHAINS_ROOT / as_of.isoformat() / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _load_form4(ticker: str) -> pd.DataFrame:
    p = FORM4_DIR / f"{ticker}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_universe() -> pd.DataFrame:
    if not UNIVERSE_PARQUET.exists():
        return pd.DataFrame(columns=["ticker", "company", "sector", "market_cap"])
    return pd.read_parquet(UNIVERSE_PARQUET)


def _available_dates() -> list[date]:
    if not ALERTS_DIR.exists():
        return []
    out = []
    for p in sorted(ALERTS_DIR.glob("*.parquet")):
        try:
            out.append(date.fromisoformat(p.stem))
        except ValueError:
            continue
    return sorted(out, reverse=True)


def _tail(text: str, max_chars: int = 8000) -> str:
    """Keep subprocess output readable inside Streamlit."""
    if len(text) <= max_chars:
        return text
    return "... output truncated ...\n" + text[-max_chars:]


def _run_dashboard_refresh(
    as_of: date,
    *,
    include_edgar: bool,
    sec_user_agent: str,
    refresh_universe: bool,
    refresh_prices: bool,
    refresh_options: bool,
    refresh_ff: bool,
    regenerate_html: bool,
) -> dict:
    """Run the local pipeline from Streamlit and optionally rebuild static HTML."""
    env = os.environ.copy()
    if sec_user_agent.strip():
        env["SEC_USER_AGENT"] = sec_user_agent.strip()
    if include_edgar and not env.get("SEC_USER_AGENT", "").strip():
        raise ValueError("SEC_USER_AGENT is required when EDGAR refresh is enabled.")

    cmd = [sys.executable, str(PROJECT_ROOT / "run_daily.py"), "--date", as_of.isoformat()]
    if not refresh_universe:
        cmd.append("--skip-universe")
    if not refresh_prices:
        cmd.append("--skip-prices")
    if not refresh_options:
        cmd.append("--skip-options")
    if not include_edgar:
        cmd.append("--skip-edgar")
    if not refresh_ff:
        cmd.append("--skip-ff")

    steps = []
    run_result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=REFRESH_TIMEOUT_SECONDS,
    )
    steps.append(("pipeline", cmd, run_result.returncode, run_result.stdout, run_result.stderr))
    if run_result.returncode != 0:
        return {"ok": False, "steps": steps}

    if regenerate_html:
        html_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_dashboard_html.py"),
            "--date",
            as_of.isoformat(),
        ]
        html_result = subprocess.run(
            html_cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15 * 60,
        )
        steps.append(("static-html", html_cmd, html_result.returncode, html_result.stdout, html_result.stderr))
        if html_result.returncode != 0:
            return {"ok": False, "steps": steps}

    st.cache_data.clear()
    return {"ok": True, "steps": steps}


# ── Date + gate ──────────────────────────────────────────────────────

dates = _available_dates()
if not dates:
    st.markdown(
        """
        <div style='max-width:620px;margin:6rem auto;padding:2rem;
                    border:1px solid #242834;background:#12151C;border-radius:2px;
                    font-family:"IBM Plex Sans",sans-serif;'>
          <h2 style='font-family:Fraunces,serif;font-weight:500;
                     margin:0 0 0.6rem 0;'>No alerts on disk</h2>
          <p style='color:#8C92A3;font-size:0.9rem;line-height:1.6;margin:0;'>
            Run <code style='color:#E8B84E;background:#0A0C10;padding:2px 6px;
            border-radius:2px;'>python3 run_daily.py</code> from the project
            root. The pipeline writes a parquet to
            <code style='color:#E8B84E;'>data/alerts/{date}.parquet</code>
            that this dashboard reads.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Top-level state: selected date + ticker ──────────────────────────

col_date_sel, _spacer = st.columns([1, 3])
with col_date_sel:
    sel = st.selectbox("As-of date", dates, index=0,
                       format_func=lambda d: d.isoformat(),
                       label_visibility="collapsed")


alerts = _load_alerts(sel)
universe = _load_universe()
universe_size = len(universe) if not universe.empty else alerts["ticker"].nunique()

n_bull = int((alerts["tier"] == "STRONG_BULLISH").sum())
n_bear = int((alerts["tier"] == "STRONG_BEARISH").sum())
n_watch = int(len(alerts) - n_bull - n_bear)
run_time_ny = datetime.now(tz=ZoneInfo("America/New_York"))


# ── Masthead ─────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class='masthead'>
      <div>
        <div class='masthead__brand'>
          <span class='ticker'>NDX</span>
          <span class='sep'>·</span>Alert Desk
        </div>
        <div class='masthead__tagline'>
          Factor &nbsp;·&nbsp; Options flow &nbsp;·&nbsp; Insider transactions
        </div>
      </div>
      <div class='masthead__meta'>
        <div><span class='label'>As-of</span>
             <span class='val accent'>{sel.isoformat()}</span></div>
        <div><span class='label'>Generated</span>
             <span class='val'>{run_time_ny.strftime('%Y-%m-%d %H:%M')} ET</span></div>
        <div><span class='label'>Universe</span>
             <span class='val'>{universe_size}&nbsp;tickers</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── KPI strip ────────────────────────────────────────────────────────

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Long alerts", n_bull)
with k2:
    st.metric("Short alerts", n_bear)
with k3:
    st.metric("Watch list", n_watch)
with k4:
    composite_range = (
        f"{alerts['composite'].min():+.2f}  …  {alerts['composite'].max():+.2f}"
        if len(alerts) else "—"
    )
    st.metric("Composite range", composite_range)


# ─────────────────────────────────────────────────────────────────────
# 01  ALERT LEDGER
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2><span class='section-num'>01</span>Alert ledger</h2>",
    unsafe_allow_html=True,
)

# Build an HTML table — Streamlit's native dataframe can't render chip HTML,
# so we hand-roll the ledger row markup. This also lets us align the mini-bar.

def _bar(v: float, span: float = 3.0) -> str:
    """HTML/CSS micro-bar scaled to +/- span z, with visible numeric z-score."""
    if pd.isna(v):
        return f"<span style='color:{Colors.TEXT_MUTED}'>—</span>"
    clipped = max(-span, min(span, v))
    bar_w = abs(clipped) / span * 50
    x0 = 50 - bar_w if clipped < 0 else 50
    color = Colors.POSITIVE if v >= 0 else Colors.NEGATIVE
    return (
        "<span style='display:inline-flex;align-items:center;gap:0.45rem;min-width:126px;'>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;"
        f"color:{Colors.TEXT_SECONDARY};width:42px;text-align:right;'>{v:+.2f}</span>"
        "<span style='display:inline-block;position:relative;width:72px;height:10px;'>"
        f"<span style='position:absolute;left:0;right:0;top:4px;height:2px;"
        f"background:{Colors.BORDER_SUBTLE};'></span>"
        f"<span style='position:absolute;left:50%;top:0;width:1px;height:10px;"
        f"background:{Colors.TEXT_MUTED};opacity:0.35;'></span>"
        f"<span style='position:absolute;left:{x0:.2f}%;top:2px;width:{bar_w:.2f}%;"
        f"height:6px;background:{color};'></span>"
        "</span></span>"
    )


def _quality_chip_html(quality: str) -> str:
    q = str(quality or "LOW").upper()
    color = Colors.POSITIVE if q == "HIGH" else Colors.ACCENT if q == "MEDIUM" else Colors.NEGATIVE
    return (
        f"<span style='display:inline-block;border:1px solid {color};"
        f"color:{color};padding:0.08rem 0.35rem;border-radius:2px;"
        f"font-family:JetBrains Mono,monospace;font-size:0.62rem;"
        f"letter-spacing:0.08em;font-weight:600;'>{escape(q)}</span>"
    )


def _render_options_snapshot(opt_row: pd.Series) -> str:
    def score_cell(label: str, key: str) -> str:
        v = opt_row.get(key)
        if pd.isna(v):
            txt = "—"
            color = Colors.TEXT_MUTED
        else:
            txt = f"{float(v):+.2f}"
            color = Colors.POSITIVE if float(v) >= 0 else Colors.NEGATIVE
        return (
            f"<div style='min-width:96px;'>"
            f"<div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.12em;"
            f"color:{Colors.TEXT_MUTED};margin-bottom:0.12rem;'>{escape(label)}</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;"
            f"color:{color};font-weight:600;'>{txt}</div>"
            f"</div>"
        )

    quality = str(opt_row.get("options_quality", "LOW") or "LOW")
    coverage = opt_row.get("options_coverage")
    coverage_txt = "—" if pd.isna(coverage) else f"{float(coverage):.0%}"
    return (
        f"<div style='border:1px solid {Colors.BORDER_SUBTLE};background:{Colors.BG_SURFACE_1};"
        f"padding:0.7rem 0.8rem;margin-bottom:0.75rem;'>"
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;gap:0.8rem;'>"
        f"<div style='display:flex;gap:1.1rem;flex-wrap:wrap;'>"
        f"{score_cell('Options z', 'options_z')}"
        f"{score_cell('Flow', 'flow_score')}"
        f"{score_cell('Skew', 'skew_score')}"
        f"{score_cell('Vol stress', 'vol_stress_score')}"
        f"</div>"
        f"<div style='text-align:right;font-family:JetBrains Mono,monospace;'>"
        f"{_quality_chip_html(quality)}"
        f"<div style='font-size:0.65rem;color:{Colors.TEXT_MUTED};margin-top:0.25rem;'>"
        f"coverage {coverage_txt}</div>"
        f"</div></div>"
        f"<div style='font-size:0.68rem;color:{Colors.TEXT_MUTED};margin-top:0.45rem;"
        f"font-family:JetBrains Mono,monospace;'>"
        f"snapshot score only · heatmap metric below is visualization-only</div>"
        f"</div>"
    )


def _render_ledger(df: pd.DataFrame) -> str:
    universe_lookup = universe.set_index("ticker") if not universe.empty else pd.DataFrame()
    rows_html = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"])
        company = ""
        sector = ""
        if not universe_lookup.empty and ticker in universe_lookup.index:
            company = str(universe_lookup.loc[ticker].get("company", "") or "")
            sector = str(universe_lookup.loc[ticker].get("sector", "") or "")
        rationale = str(row.get("rationale", "") or "")
        rows_html.append(
            "<tr>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:top;width:92px;'>{tier_chip_html(row['tier'])}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:top;width:140px;'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-weight:600;"
            f"font-size:0.95rem;color:{Colors.TEXT_PRIMARY};letter-spacing:0.02em;'>{escape(ticker)}</div>"
            f"<div style='font-size:0.72rem;color:{Colors.TEXT_MUTED};margin-top:1px;'>{escape(sector)}</div>"
            "</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:top;color:{Colors.TEXT_SECONDARY};font-size:0.86rem;max-width:260px;'>"
            f"{escape(company)}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:middle;font-family:JetBrains Mono,monospace;font-size:0.88rem;"
            f"color:{Colors.TEXT_PRIMARY};font-weight:500;text-align:right;width:90px;'>"
            f"{row['composite']:+.2f}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:middle;width:108px;'>{_bar(row['factor_z'])}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:middle;width:108px;'>{_bar(row['options_z'])}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:middle;width:108px;'>{_bar(row['insider_z'])}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
            f"vertical-align:middle;color:{Colors.TEXT_SECONDARY};font-size:0.78rem;"
            f"font-family:JetBrains Mono,monospace;'>{escape(rationale)}</td>"
            "</tr>"
        )
    header_cell = (
        f"padding:0.4rem 0.9rem;text-align:left;font-size:0.63rem;"
        f"text-transform:uppercase;letter-spacing:0.14em;color:{Colors.TEXT_MUTED};"
        f"font-weight:600;border-bottom:1px solid {Colors.BORDER_DEFAULT};"
        f"background:{Colors.BG_SURFACE_2};"
    )
    header = (
        f"<tr>"
        f"<th style='{header_cell}'>Tier</th>"
        f"<th style='{header_cell}'>Ticker</th>"
        f"<th style='{header_cell}'>Company</th>"
        f"<th style='{header_cell};text-align:right'>Composite</th>"
        f"<th style='{header_cell}'>Factor z</th>"
        f"<th style='{header_cell}'>Options z</th>"
        f"<th style='{header_cell}'>Insider z</th>"
        f"<th style='{header_cell}'>Rationale</th>"
        f"</tr>"
    )
    return (
        f"<div style='border:1px solid {Colors.BORDER_SUBTLE};border-radius:2px;"
        f"overflow:hidden;background:{Colors.BG_SURFACE_1};'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"font-family:IBM Plex Sans,sans-serif;'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


filt_col1, filt_col2, filt_col3 = st.columns([1, 1, 4])
with filt_col1:
    tier_filter = st.radio(
        "Tier filter",
        options=["All", "Long only", "Short only", "Alerts only"],
        index=0, horizontal=True,
        label_visibility="collapsed",
    )
with filt_col2:
    top_n = st.number_input("Top N", min_value=5, max_value=len(alerts) or 5,
                            value=min(20, len(alerts) or 5), step=5,
                            label_visibility="collapsed")

view = alerts.copy()
if tier_filter == "Long only":
    view = view[view["tier"] == "STRONG_BULLISH"]
elif tier_filter == "Short only":
    view = view[view["tier"] == "STRONG_BEARISH"]
elif tier_filter == "Alerts only":
    view = view[view["tier"] != "NO_ALERT"]
view = view.head(int(top_n))

if view.empty:
    st.markdown(
        f"<div style='padding:1.5rem;color:{Colors.TEXT_MUTED};"
        f"font-size:0.9rem;text-align:center;'>"
        f"No rows match the current filter.</div>",
        unsafe_allow_html=True,
    )
else:
    st.html(_render_ledger(view))


# ─────────────────────────────────────────────────────────────────────
# 02  DEEP DIVE
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2><span class='section-num'>02</span>Ticker deep dive</h2>",
    unsafe_allow_html=True,
)

dd_col1, _ = st.columns([1, 4])
with dd_col1:
    ticker = st.selectbox("Ticker",
                          alerts["ticker"].tolist(),
                          index=0 if len(alerts) else None,
                          label_visibility="collapsed")

if ticker:
    row = alerts[alerts["ticker"] == ticker].iloc[0]

    # Sub-header: ticker masthead micro-strip
    universe_lookup = universe.set_index("ticker") if not universe.empty else pd.DataFrame()
    company = ""
    sector = ""
    if not universe_lookup.empty and ticker in universe_lookup.index:
        company = str(universe_lookup.loc[ticker].get("company", "") or "")
        sector = str(universe_lookup.loc[ticker].get("sector", "") or "")

    st.markdown(
        f"""
        <div style='display:flex;justify-content:space-between;align-items:flex-end;
                    padding:0.6rem 0 1rem 0;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                    margin-bottom:1.2rem;'>
          <div>
            <div style='font-family:Fraunces,serif;font-size:1.8rem;font-weight:500;
                        color:{Colors.ACCENT};letter-spacing:-0.01em;line-height:1;'>
              {ticker}</div>
            <div style='font-size:0.78rem;color:{Colors.TEXT_SECONDARY};margin-top:0.25rem;'>
              {company} <span style='color:{Colors.TEXT_MUTED}'>·</span>
              <span style='color:{Colors.TEXT_MUTED}'>{sector}</span>
            </div>
          </div>
          <div style='text-align:right;font-family:JetBrains Mono,monospace;font-size:0.8rem;'>
            <div><span style='color:{Colors.TEXT_MUTED};font-size:0.65rem;
                        letter-spacing:0.14em;text-transform:uppercase;margin-right:0.3rem;'>
                  Tier</span>{tier_chip_html(row["tier"])}</div>
            <div style='margin-top:0.35rem;color:{Colors.TEXT_PRIMARY};'>
              <span style='color:{Colors.TEXT_MUTED};font-size:0.65rem;
                   letter-spacing:0.14em;text-transform:uppercase;margin-right:0.3rem;'>
                Composite</span>
              <span style='color:{Colors.ACCENT};'>{row["composite"]:+.3f}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2 x 2 grid: price | factors  /  options | insiders
    grid_r1_c1, grid_r1_c2 = st.columns(2)

    # ── Price chart (1y adj close + 50-day SMA) ──────────────────────
    with grid_r1_c1:
        st.markdown("<h3>Price · 1Y</h3>", unsafe_allow_html=True)
        prices = _load_prices()
        sub = prices[prices["ticker"] == ticker].sort_values("date").tail(252)
        if sub.empty:
            st.info(f"No price history for {ticker}.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["adj_close"],
                mode="lines",
                line=dict(color=Colors.ACCENT, width=1.5),
                hovertemplate="%{x|%Y-%m-%d}  $%{y:.2f}<extra></extra>",
                name="Adj close",
            ))
            # 50d SMA overlay
            sma50 = sub["adj_close"].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sma50, mode="lines",
                line=dict(color=Colors.TEXT_MUTED, width=1, dash="dot"),
                hovertemplate="SMA50 $%{y:.2f}<extra></extra>",
                name="SMA50",
            ))
            last_px = sub["adj_close"].iloc[-1]
            ret_1m = (last_px / sub["adj_close"].iloc[-21] - 1) * 100 if len(sub) >= 21 else None
            fig.update_layout(**plotly_layout(
                height=ChartHeight.MEDIUM,
                showlegend=False,
                margin=dict(t=16, b=32, l=48, r=12),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if ret_1m is not None:
                color = Colors.POSITIVE if ret_1m >= 0 else Colors.NEGATIVE
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;"
                    f"font-size:0.78rem;color:{Colors.TEXT_MUTED};'>"
                    f"Last <span style='color:{Colors.TEXT_PRIMARY}'>${last_px:,.2f}</span>"
                    f"  &nbsp; 1M "
                    f"<span style='color:{color}'>{ret_1m:+.2f}%</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Factor breakdown bars ────────────────────────────────────────
    with grid_r1_c2:
        st.markdown("<h3>Factor decomposition</h3>", unsafe_allow_html=True)
        fdf = _load_factor(sel)
        frow = fdf[fdf["ticker"] == ticker] if not fdf.empty else pd.DataFrame()
        if frow.empty:
            st.info("No factor panel for this date.")
        else:
            r = frow.iloc[0]
            z_cols = ["momentum_z", "value_z", "quality_z", "lowvol_z"]
            labels = ["Momentum", "Value", "Quality", "Low-vol"]
            vals = [float(r[c]) for c in z_cols]
            colors_bar = [Colors.POSITIVE if v >= 0 else Colors.NEGATIVE for v in vals]
            fig = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=colors_bar,
                hovertemplate="%{y}  %{x:+.2f}<extra></extra>",
                text=[f"{v:+.2f}" for v in vals],
                textposition="outside",
                textfont=dict(family="JetBrains Mono, monospace",
                              color=Colors.TEXT_SECONDARY, size=11),
            ))
            fig.update_layout(**plotly_layout(
                height=ChartHeight.MEDIUM,
                margin=dict(t=16, b=32, l=92, r=40),
                showlegend=False,
                xaxis=dict(gridcolor=Colors.BORDER_SUBTLE, zeroline=True,
                           zerolinecolor=Colors.BORDER_DEFAULT,
                           tickfont=dict(family="JetBrains Mono, monospace",
                                         color=Colors.TEXT_SECONDARY, size=10)),
                yaxis=dict(autorange="reversed",
                           tickfont=dict(family="IBM Plex Sans, sans-serif",
                                         color=Colors.TEXT_SECONDARY, size=11)),
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            raw_available = any(c + "_raw" in frow.columns for c in z_cols)
            if raw_available:
                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.72rem;"
                    f"color:{Colors.TEXT_MUTED};margin-top:-0.8rem;'>"
                    f"Sector-neutralized z-scores (pre-neutralize raw in parquet)"
                    f"</div>", unsafe_allow_html=True,
                )

    grid_r2_c1, grid_r2_c2 = st.columns(2)

    # ── Options heatmap (V/OI by strike x expiry) ────────────────────
    with grid_r2_c1:
        st.markdown("<h3>Options · Snapshot surface</h3>", unsafe_allow_html=True)
        opt_panel = _load_options_panel(sel)
        opt_info = opt_panel[opt_panel["ticker"] == ticker] if not opt_panel.empty else pd.DataFrame()
        if not opt_info.empty:
            st.html(_render_options_snapshot(opt_info.iloc[0]))
        chain = _load_chain(ticker, sel)
        if chain.empty:
            st.info(f"No options chain snapshot for {ticker} on {sel}.")
        else:
            ch = chain.copy()
            ch["expiry"] = pd.to_datetime(ch["expiry"])
            ch["strike"] = pd.to_numeric(ch["strike"], errors="coerce")
            ch["volume"] = pd.to_numeric(ch["volume"], errors="coerce").fillna(0.0)
            ch["open_interest"] = pd.to_numeric(ch["open_interest"], errors="coerce").fillna(0.0)
            ch["v_oi"] = ch["volume"] / ch["open_interest"].replace(0, pd.NA)
            spot = latest_adj_close_on_or_before(prices, ticker, sel)
            if spot is None:
                st.info(f"No price history for {ticker} on or before {sel}.")
            else:
                ch["dte"] = (ch["expiry"] - pd.Timestamp(sel)).dt.days.clip(lower=0)
                # Focus on +/- 15% ATM band; prefer V/OI, but fall back to
                # volume when Yahoo reports near-expiry open interest as zero.
                band = ch[(ch["strike"] >= spot * 0.85) & (ch["strike"] <= spot * 1.15)]
                metric_col = "v_oi"
                metric_label = "V/OI"
                metric_format = ".3f"
                metric_band = band[band[metric_col].notna()].copy()
                aggfunc = "mean"
                if len(metric_band) < 10 or metric_band["expiry"].nunique() < 2:
                    metric_col = "volume"
                    metric_label = "Volume"
                    metric_format = ",.0f"
                    metric_band = band[band["volume"] > 0].copy()
                    aggfunc = "sum"

                exp_keep = (
                    metric_band[["expiry", "dte"]].drop_duplicates()
                    .sort_values("dte")
                    .head(5)["expiry"]
                    .tolist()
                )
                band = metric_band[metric_band["expiry"].isin(exp_keep)]
                pivot = band.pivot_table(
                    index="strike", columns="expiry", values=metric_col,
                    aggfunc=aggfunc,
                ).sort_index()
                if pivot.empty:
                    st.info("Chain has no usable volume or open-interest data inside the ATM band.")
                else:
                    z_values = pd.Series(pivot.to_numpy().ravel()).dropna()
                    zmax = max(1.0, float(z_values.max())) if not z_values.empty else 1.0
                    if metric_label == "Volume":
                        st.caption("Open interest is unavailable in the selected ATM band; showing volume.")
                    else:
                        st.caption("Surface metric: V/OI. This is visualization-only.")
                    fig = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=[d.strftime("%b %d") for d in pivot.columns],
                        y=[f"{s:g}" for s in pivot.index],
                        colorscale=SEQUENTIAL_GOLD,
                        zmin=0, zmax=zmax,
                        colorbar=dict(
                            thickness=8, len=0.65,
                            tickfont=dict(family="JetBrains Mono, monospace",
                                          color=Colors.TEXT_SECONDARY, size=9),
                        ),
                        hovertemplate=(
                            f"K %{{y}}  exp %{{x}}  {metric_label} %{{z:{metric_format}}}"
                            "<extra></extra>"
                        ),
                    ))
                    # Spot line, only if spot lands on a rendered strike row.
                    spot_label = f"{spot:g}"
                    if spot_label in [f"{s:g}" for s in pivot.index]:
                        fig.add_hline(y=spot_label, line_color=Colors.ACCENT,
                                      line_width=1, line_dash="dot")
                    fig.update_layout(**plotly_layout(
                        height=ChartHeight.MEDIUM,
                        margin=dict(t=16, b=32, l=58, r=12),
                        xaxis=dict(side="bottom",
                                   tickfont=dict(family="JetBrains Mono, monospace",
                                                 color=Colors.TEXT_SECONDARY, size=10)),
                        yaxis=dict(autorange="reversed",
                                   tickfont=dict(family="JetBrains Mono, monospace",
                                                 color=Colors.TEXT_SECONDARY, size=10)),
                    ))
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Insider transactions ─────────────────────────────────────────
    with grid_r2_c2:
        st.markdown("<h3>Insiders · last 20</h3>", unsafe_allow_html=True)
        f4 = _load_form4(ticker)
        if f4.empty:
            st.info("No Form 4 history for this ticker.")
        else:
            recent = f4.sort_values("transaction_date", ascending=False).head(20).copy()
            recent["value_fmt"] = recent["value"].apply(
                lambda v: fmt_dollar(v) if pd.notna(v) and v > 0 else "—"
            )
            recent["date_fmt"] = pd.to_datetime(recent["transaction_date"]).dt.strftime("%Y-%m-%d")

            rows = []
            for _, rr in recent.iterrows():
                signal_weight = int(rr.get("signal_weight", 0) or 0)
                if signal_weight > 0:
                    col = Colors.POSITIVE
                    code_label = rr["signal_label"]
                elif signal_weight < 0:
                    col = Colors.NEGATIVE
                    code_label = rr["signal_label"]
                else:
                    col = Colors.TEXT_MUTED
                    code_label = rr["signal_label"]
                rows.append(
                    "<tr>"
                    f"<td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
                    f"font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                    f"color:{Colors.TEXT_SECONDARY};'>{escape(str(rr['date_fmt']))}</td>"
                    f"<td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
                    f"font-size:0.8rem;color:{Colors.TEXT_PRIMARY};max-width:160px;overflow:hidden;"
                    f"text-overflow:ellipsis;white-space:nowrap;'>{escape(str(rr['insider_name']))}</td>"
                    f"<td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
                    f"font-size:0.72rem;color:{Colors.TEXT_MUTED};'>"
                    f"{escape(str(rr.get('position', '') or ''))}</td>"
                    f"<td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
                    f"font-family:JetBrains Mono,monospace;font-size:0.72rem;color:{col};"
                    f"letter-spacing:0.04em;'>{escape(str(code_label))}</td>"
                    f"<td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};"
                    f"text-align:right;font-family:JetBrains Mono,monospace;font-size:0.78rem;"
                    f"color:{Colors.TEXT_PRIMARY};'>{escape(str(rr['value_fmt']))}</td>"
                    "</tr>"
                )
            header_cell = (
                f"padding:0.3rem 0.7rem;text-align:left;font-size:0.58rem;"
                f"text-transform:uppercase;letter-spacing:0.14em;color:{Colors.TEXT_MUTED};"
                f"font-weight:600;border-bottom:1px solid {Colors.BORDER_DEFAULT};"
                f"background:{Colors.BG_SURFACE_2};"
            )
            st.html(
                f"<div style='border:1px solid {Colors.BORDER_SUBTLE};border-radius:2px;"
                f"overflow:hidden;background:{Colors.BG_SURFACE_1};max-height:340px;overflow-y:auto;'>"
                f"<table style='width:100%;border-collapse:collapse;font-family:IBM Plex Sans,sans-serif;'>"
                f"<thead><tr>"
                f"<th style='{header_cell}'>Date</th>"
                f"<th style='{header_cell}'>Insider</th>"
                f"<th style='{header_cell}'>Role</th>"
                f"<th style='{header_cell}'>Signal</th>"
                f"<th style='{header_cell};text-align:right'>Value</th>"
                f"</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                f"</table></div>"
            )


# ─────────────────────────────────────────────────────────────────────
# 03  UNIVERSE CONSTELLATION
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2><span class='section-num'>03</span>Universe constellation</h2>",
    unsafe_allow_html=True,
)

scat_df = alerts.dropna(subset=["factor_z", "options_z", "insider_z"]).copy()
if not scat_df.empty:
    # Join sector for hover tooltip
    if not universe.empty:
        scat_df = scat_df.merge(universe[["ticker", "sector", "company"]],
                                on="ticker", how="left")
    else:
        scat_df["sector"] = ""
        scat_df["company"] = ""

    fig = go.Figure()

    # Faint quadrant dividers — give users a visual frame
    fig.add_vline(x=0, line_color=Colors.BORDER_DEFAULT, line_width=1, opacity=0.6)
    fig.add_hline(y=0, line_color=Colors.BORDER_DEFAULT, line_width=1, opacity=0.6)

    tiered = scat_df[scat_df["tier"] != "NO_ALERT"]
    untiered = scat_df[scat_df["tier"] == "NO_ALERT"]

    fig.add_trace(go.Scatter(
        x=untiered["factor_z"], y=untiered["options_z"],
        mode="markers",
        marker=dict(
            size=9,
            color=untiered["insider_z"],
            colorscale=DIVERGING_SCALE,
            cmid=0,
            line=dict(color=Colors.BORDER_DEFAULT, width=0.5),
            opacity=0.85,
            colorbar=dict(
                title=dict(
                    text="Insider z",
                    font=dict(family="IBM Plex Sans, sans-serif",
                              size=10, color=Colors.TEXT_MUTED),
                ),
                thickness=8, len=0.65,
                tickfont=dict(family="JetBrains Mono, monospace",
                              size=9, color=Colors.TEXT_SECONDARY),
            ),
        ),
        text=untiered["ticker"],
        customdata=untiered[["company", "sector"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "%{customdata[0]} · %{customdata[1]}<br>"
            "factor z %{x:+.2f} · options z %{y:+.2f}<extra></extra>"
        ),
        name="Watch",
        showlegend=False,
    ))
    if not tiered.empty:
        fig.add_trace(go.Scatter(
            x=tiered["factor_z"], y=tiered["options_z"],
            mode="markers+text",
            marker=dict(
                size=13,
                color=tiered["insider_z"],
                colorscale=DIVERGING_SCALE, cmid=0,
                line=dict(color=Colors.ACCENT, width=1.2),
                opacity=1.0,
            ),
            text=tiered["ticker"],
            textposition="top center",
            textfont=dict(family="JetBrains Mono, monospace",
                          size=10, color=Colors.ACCENT),
            customdata=tiered[["company", "sector", "tier"]].values,
            hovertemplate=(
                "<b>%{text}</b>  (%{customdata[2]})<br>"
                "%{customdata[0]} · %{customdata[1]}<br>"
                "factor z %{x:+.2f} · options z %{y:+.2f}<extra></extra>"
            ),
            name="Alerts",
            showlegend=False,
        ))

    fig.update_layout(**plotly_layout(
        height=ChartHeight.LARGE,
        xaxis_title="Factor composite z",
        yaxis_title="Options flow z",
        hovermode="closest",
        margin=dict(t=24, b=44, l=64, r=24),
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Quadrant legend
    st.markdown(
        f"""
        <div style='display:flex;gap:2rem;font-family:JetBrains Mono,monospace;
                    font-size:0.72rem;color:{Colors.TEXT_MUTED};padding:0 0.4rem;'>
          <span><span class='pill pos'>↗</span>
                Q1: positive factor + positive flow → tailwind</span>
          <span><span class='pill warn'>↖</span>
                Q2: weak factor but bullish flow → contrarian watch</span>
          <span><span class='pill neg'>↙</span>
                Q3: negative factor + bearish flow → short lean</span>
          <span><span class='pill'>↘</span>
                Q4: strong factor, bearish flow → divergence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────
# 04  EXPORT NEWSLETTER
# ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h2><span class='section-num'>04</span>Export newsletter</h2>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style='border:1px solid {Colors.BORDER_SUBTLE};background:{Colors.BG_SURFACE_1};
                padding:0.95rem 1.1rem;margin-bottom:0.9rem;'>
      <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.16em;
                  color:{Colors.TEXT_MUTED};margin-bottom:0.35rem;'>Local data refresh</div>
      <div style='font-family:JetBrains Mono,monospace;color:{Colors.TEXT_PRIMARY};
                  font-size:0.86rem;line-height:1.55;'>
        Fetches data into local parquet files, recomputes signals, and optionally rebuilds
        <span style='color:{Colors.ACCENT};'>NDX Alert Desk.html</span>.
      </div>
      <div style='font-family:JetBrains Mono,monospace;color:{Colors.TEXT_MUTED};
                  font-size:0.7rem;margin-top:0.55rem;'>
        Static file exports cannot run Python or write data; use this Streamlit control
        or the CLI for real refreshes.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

refresh_col1, refresh_col2, refresh_col3 = st.columns([0.8, 1.1, 1.1])
with refresh_col1:
    refresh_date = st.date_input("Refresh date", value=sel)
    refresh_clicked = st.button("Fetch data / rebuild", type="secondary")
with refresh_col2:
    refresh_universe = st.checkbox("Refresh universe", value=False)
    refresh_prices = st.checkbox("Refresh prices + fundamentals + VIX", value=True)
    refresh_options = st.checkbox("Refresh options chains", value=True)
with refresh_col3:
    refresh_ff = st.checkbox("Refresh Fama-French factors", value=True)
    include_edgar = st.checkbox(
        "Include incremental EDGAR Form 4",
        value=bool(os.environ.get("SEC_USER_AGENT", "").strip()),
    )
    regenerate_html = st.checkbox("Regenerate static HTML", value=True)

sec_user_agent = ""
if include_edgar:
    sec_user_agent = st.text_input(
        "SEC_USER_AGENT",
        value=os.environ.get("SEC_USER_AGENT", ""),
        placeholder="Your Name your@email.com",
        help="Required by SEC EDGAR. This is contact metadata, not a password.",
    )

if refresh_clicked:
    with st.spinner("Running local refresh. This can take several minutes for options chains."):
        try:
            refresh_result = _run_dashboard_refresh(
                refresh_date,
                include_edgar=include_edgar,
                sec_user_agent=sec_user_agent,
                refresh_universe=refresh_universe,
                refresh_prices=refresh_prices,
                refresh_options=refresh_options,
                refresh_ff=refresh_ff,
                regenerate_html=regenerate_html,
            )
        except subprocess.TimeoutExpired as exc:
            st.error(f"Refresh timed out after {exc.timeout} seconds.")
            refresh_result = None
        except Exception as exc:
            st.error(f"Refresh failed before launch: {exc}")
            refresh_result = None

    if refresh_result is not None:
        if refresh_result["ok"]:
            st.success(
                "Refresh completed. Reload the page or choose the new date from the date selector."
            )
        else:
            st.error("Refresh command failed. See the command output below.")

        for label, cmd, returncode, stdout, stderr in refresh_result["steps"]:
            with st.expander(f"{label} · exit {returncode}", expanded=returncode != 0):
                st.code(" ".join(str(part) for part in cmd), language="bash")
                combined = "\n".join(part for part in (_tail(stdout), _tail(stderr)) if part)
                st.code(combined or "(no output)", language="text")

try:
    preview_ctx = load_newsletter_context(sel, project_root=PROJECT_ROOT)
    preview_text = render_newsletter_text(preview_ctx)
except Exception as exc:
    st.warning(f"Newsletter preview unavailable: {exc}")
    preview_ctx = None
    preview_text = ""

if preview_ctx is not None:
    st.markdown(
        f"""
        <div style='border:1px solid {Colors.BORDER_SUBTLE};background:{Colors.BG_SURFACE_1};
                    padding:0.95rem 1.1rem;margin-bottom:0.9rem;'>
          <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.16em;
                      color:{Colors.TEXT_MUTED};margin-bottom:0.35rem;'>Email subject</div>
          <div style='font-family:JetBrains Mono,monospace;color:{Colors.TEXT_PRIMARY};
                      font-size:0.86rem;'>{escape(preview_ctx.subject)}</div>
          <div style='font-family:JetBrains Mono,monospace;color:{Colors.TEXT_MUTED};
                      font-size:0.7rem;margin-top:0.55rem;'>
            creates a local .eml draft with blank recipients · never sends email · export root:
            {escape(str(EXPORT_ROOT))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns([1, 2])
    with exp_col1:
        include_png = st.checkbox(
            "Render PNG dashboard snapshot",
            value=True,
            help="Uses optional Playwright/Chromium. If unavailable, the export still succeeds without PNG.",
        )
        generate_clicked = st.button("Generate email package", type="primary")
    with exp_col2:
        st.text_area(
            "Plain-text preview",
            preview_text,
            height=210,
            label_visibility="collapsed",
        )

    if generate_clicked:
        with st.spinner("Generating newsletter export package..."):
            artifacts = export_newsletter(
                sel,
                project_root=PROJECT_ROOT,
                render_png=include_png,
            )
        st.session_state["newsletter_artifacts"] = artifacts

    artifacts = st.session_state.get("newsletter_artifacts")
    if artifacts is not None:
        st.success(f"Newsletter package ready: {artifacts.output_dir}")
        st.caption(f"PNG status: {artifacts.png_status}")

        downloads = [
            ("Download EML draft", artifacts.eml, "message/rfc822"),
            ("Download newsletter HTML", artifacts.newsletter_html, "text/html"),
            ("Download newsletter text", artifacts.newsletter_text, "text/plain"),
        ]
        if artifacts.dashboard_html is not None:
            downloads.append(("Download dashboard HTML", artifacts.dashboard_html, "text/html"))
        if artifacts.dashboard_png is not None:
            downloads.append(("Download dashboard PNG", artifacts.dashboard_png, "image/png"))

        dl_cols = st.columns(min(5, len(downloads)))
        for i, (label, path, mime) in enumerate(downloads):
            with dl_cols[i % len(dl_cols)]:
                st.download_button(
                    label,
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                    key=f"download-{path.name}",
                )

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

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.lifted.display import nice_columns
from src.lifted.ui_style import (
    CHART_PALETTE, DIVERGING_SCALE, SEQUENTIAL_GOLD,
    ChartHeight, Colors, fmt_dollar, fmt_z, inject_css,
    plotly_layout, tier_chip_html,
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
    """Inline-SVG micro-bar scaled to +/- span z. 72px wide, 10px tall."""
    if pd.isna(v):
        return f"<span style='color:{Colors.TEXT_MUTED}'>—</span>"
    width_px = 72
    mid = width_px / 2
    clipped = max(-span, min(span, v))
    bar_w = abs(clipped) / span * (width_px / 2)
    x0 = mid - bar_w if clipped < 0 else mid
    color = Colors.POSITIVE if v >= 0 else Colors.NEGATIVE
    return (
        f"<svg width='{width_px}' height='10' style='vertical-align:middle'>"
        f"<rect x='0' y='4' width='{width_px}' height='2' fill='{Colors.BORDER_SUBTLE}'/>"
        f"<rect x='{x0}' y='2' width='{bar_w}' height='6' fill='{color}'/>"
        f"<rect x='{mid - 0.5}' y='0' width='1' height='10' fill='{Colors.TEXT_MUTED}' opacity='0.35'/>"
        f"</svg>"
    )


def _render_ledger(df: pd.DataFrame) -> str:
    universe_lookup = universe.set_index("ticker") if not universe.empty else pd.DataFrame()
    rows_html = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        company = ""
        sector = ""
        if not universe_lookup.empty and ticker in universe_lookup.index:
            company = str(universe_lookup.loc[ticker].get("company", "") or "")
            sector = str(universe_lookup.loc[ticker].get("sector", "") or "")
        rows_html.append(
            f"""
            <tr>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:top;width:92px;'>{tier_chip_html(row['tier'])}</td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:top;width:140px;'>
                <div style='font-family:JetBrains Mono,monospace;font-weight:600;
                            font-size:0.95rem;color:{Colors.TEXT_PRIMARY};
                            letter-spacing:0.02em;'>{ticker}</div>
                <div style='font-size:0.72rem;color:{Colors.TEXT_MUTED};
                            margin-top:1px;'>{sector}</div>
              </td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:top;color:{Colors.TEXT_SECONDARY};
                         font-size:0.86rem;max-width:260px;'>{company}</td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:middle;font-family:JetBrains Mono,monospace;
                         font-size:0.88rem;color:{Colors.TEXT_PRIMARY};
                         font-weight:500;text-align:right;width:90px;'>
                {row['composite']:+.2f}
              </td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:middle;width:108px;'>{_bar(row['factor_z'])}</td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:middle;width:108px;'>{_bar(row['options_z'])}</td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:middle;width:108px;'>{_bar(row['insider_z'])}</td>
              <td style='padding:0.55rem 0.9rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                         vertical-align:middle;color:{Colors.TEXT_SECONDARY};
                         font-size:0.78rem;font-family:JetBrains Mono,monospace;'>
                {row['rationale']}
              </td>
            </tr>
            """
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
    st.markdown(_render_ledger(view), unsafe_allow_html=True)


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

    # ── Price + 20d holding overlay ──────────────────────────────────
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
        st.markdown("<h3>Options · V/OI surface</h3>", unsafe_allow_html=True)
        chain = _load_chain(ticker, sel)
        if chain.empty:
            st.info(f"No options chain snapshot for {ticker} on {sel}.")
        else:
            ch = chain.copy()
            ch["expiry"] = pd.to_datetime(ch["expiry"])
            # V/OI ratio per strike x expiry; aggregate calls+puts here
            ch["v_oi"] = (
                ch["volume"].fillna(0)
                / ch["open_interest"].replace(0, pd.NA)
            )
            spot = float(prices[prices["ticker"] == ticker]["adj_close"].iloc[-1])
            # Focus on +/- 15% ATM band and next 5 expiries
            band = ch[(ch["strike"] >= spot * 0.85) & (ch["strike"] <= spot * 1.15)]
            exp_keep = sorted(band["expiry"].unique())[:5]
            band = band[band["expiry"].isin(exp_keep)]
            pivot = band.pivot_table(
                index="strike", columns="expiry", values="v_oi",
                aggfunc="mean",
            ).sort_index()
            if pivot.empty:
                st.info("Chain is too thin after ATM filter.")
            else:
                fig = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=[d.strftime("%b %d") for d in pivot.columns],
                    y=[f"{s:g}" for s in pivot.index],
                    colorscale=SEQUENTIAL_GOLD,
                    zmin=0, zmax=max(1.0, float(pivot.values[~pd.isna(pivot.values)].max() or 1)),
                    colorbar=dict(
                        thickness=8, len=0.65,
                        tickfont=dict(family="JetBrains Mono, monospace",
                                      color=Colors.TEXT_SECONDARY, size=9),
                    ),
                    hovertemplate="K %{y}  exp %{x}  V/OI %{z:.2f}<extra></extra>",
                ))
                # Spot line
                fig.add_hline(y=f"{spot:g}" if f"{spot:g}" in [f"{s:g}" for s in pivot.index]
                              else None,
                              line_color=Colors.ACCENT, line_width=1, line_dash="dot")
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
                    f"""
                    <tr>
                      <td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                                 font-family:JetBrains Mono,monospace;font-size:0.78rem;
                                 color:{Colors.TEXT_SECONDARY};'>{rr['date_fmt']}</td>
                      <td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                                 font-size:0.8rem;color:{Colors.TEXT_PRIMARY};
                                 max-width:160px;overflow:hidden;text-overflow:ellipsis;
                                 white-space:nowrap;'>{rr['insider_name']}</td>
                      <td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                                 font-size:0.72rem;color:{Colors.TEXT_MUTED};'>
                                 {rr.get('position', '') or ''}</td>
                      <td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                                 font-family:JetBrains Mono,monospace;font-size:0.72rem;
                                 color:{col};letter-spacing:0.04em;'>{code_label}</td>
                      <td style='padding:0.35rem 0.7rem;border-bottom:1px solid {Colors.BORDER_SUBTLE};
                                 text-align:right;font-family:JetBrains Mono,monospace;
                                 font-size:0.78rem;color:{Colors.TEXT_PRIMARY};'>
                                 {rr['value_fmt']}</td>
                    </tr>
                    """
                )
            header_cell = (
                f"padding:0.3rem 0.7rem;text-align:left;font-size:0.58rem;"
                f"text-transform:uppercase;letter-spacing:0.14em;color:{Colors.TEXT_MUTED};"
                f"font-weight:600;border-bottom:1px solid {Colors.BORDER_DEFAULT};"
                f"background:{Colors.BG_SURFACE_2};"
            )
            st.markdown(
                f"""
                <div style='border:1px solid {Colors.BORDER_SUBTLE};border-radius:2px;
                            overflow:hidden;background:{Colors.BG_SURFACE_1};
                            max-height:340px;overflow-y:auto;'>
                  <table style='width:100%;border-collapse:collapse;
                               font-family:IBM Plex Sans,sans-serif;'>
                    <thead><tr>
                      <th style='{header_cell}'>Date</th>
                      <th style='{header_cell}'>Insider</th>
                      <th style='{header_cell}'>Role</th>
                      <th style='{header_cell}'>Signal</th>
                      <th style='{header_cell};text-align:right'>Value</th>
                    </tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True,
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

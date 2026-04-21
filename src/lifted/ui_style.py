"""Institutional visual system for the NDX 100 alert dashboard.

Aesthetic direction: "Deep Trading" — quant-desk gravitas, editorial
typographic rhythm, aged-gold accent. Inspired by Bloomberg Terminal
density and Koyfin's restraint, but with a serif masthead for
newspaper-of-record feel.

Typography (all free on Google Fonts):
    Display / headers : Fraunces  (variable serif, optical sizes)
    Body / UI         : IBM Plex Sans  (technical neo-grotesque)
    Numbers / data    : JetBrains Mono  (tabular figures)

Palette (blue-black canvas, desaturated semantic colors, aged-gold accent):
    Canvas #0A0C10   Panel #12151C   Elevated #1A1E27
    Border #242834 / #2D3240
    Gold   #E8B84E (accent)
    Emerald #5FB98A (positive)   Crimson #E06868 (negative)
"""

from __future__ import annotations


class Colors:
    """Institutional color tokens. Reference only, never hard-code hex."""
    # Canvas / surface
    BG_BASE      = "#0A0C10"   # deep blue-black, app canvas
    BG_SURFACE_1 = "#12151C"   # panels, table rows
    BG_SURFACE_2 = "#1A1E27"   # hover, elevated cards
    BG_SURFACE_3 = "#242834"   # borders, section dividers

    # Accent
    ACCENT       = "#E8B84E"   # aged gold — masthead, active chips
    ACCENT_SOFT  = "#8A6B1F"   # desaturated gold for muted states
    ACCENT_HOVER = "#F0C968"

    # Semantic (desaturated — institutional palette, not "app" candy colors)
    POSITIVE     = "#5FB98A"   # emerald, long / bullish
    NEGATIVE     = "#E06868"   # dusty red, short / bearish
    WARNING      = "#E8B84E"   # reuse gold for warning (same hue, role-specific)
    NEUTRAL      = "#8C92A3"   # muted blue-grey

    # Text
    TEXT_PRIMARY   = "#F0F1F5"   # warm white
    TEXT_SECONDARY = "#B4B9C6"
    TEXT_MUTED     = "#6B7182"
    TEXT_ACCENT    = "#E8B84E"

    # Borders
    BORDER_SUBTLE  = "#1E2230"
    BORDER_DEFAULT = "#2D3240"
    BORDER_STRONG  = "#3D4356"

    # Gradient endpoints for heatmap
    DEEP_RED     = "#8E2626"
    DEEP_GREEN   = "#2F6B4C"
    LIGHT_GREEN  = "#7FCBA0"
    DARK_NEUTRAL = "#12151C"


class ChartHeight:
    SPARK  = 120   # inline micro-bars
    SMALL  = 240
    MEDIUM = 320
    LARGE  = 420
    XLARGE = 520
    HERO   = 640


# Diverging scale for z-score scatter, V/OI heatmap
DIVERGING_SCALE = [
    [0.00, Colors.DEEP_RED],
    [0.25, Colors.NEGATIVE],
    [0.50, Colors.DARK_NEUTRAL],
    [0.75, Colors.POSITIVE],
    [1.00, Colors.DEEP_GREEN],
]

SEQUENTIAL_GOLD = [
    [0.0, Colors.DARK_NEUTRAL],
    [0.5, Colors.ACCENT_SOFT],
    [1.0, Colors.ACCENT],
]

CHART_PALETTE = [
    Colors.ACCENT,
    Colors.POSITIVE,
    Colors.NEGATIVE,
    Colors.NEUTRAL,
    "#A580D3",   # dusty violet, 5th series
    "#65B5C9",   # dusty cyan,   6th series
]


# ── Plotly layout ───────────────────────────────────────────────────────

def plotly_layout(**overrides) -> dict:
    """Institutional Plotly layout. Merge with per-chart overrides.

    Design decisions:
      - Paper + plot bg are the SAME color as the surface panel Streamlit
        places charts on -> chart reads as part of the panel, not a
        floating card.
      - Grid lines are nearly-invisible (#1E2230) so data dominates.
      - Monospace axis ticks: JetBrains Mono for tabular figures.
      - Margins are tight (tr margin = 16) to reclaim pixels for data.
    """
    defaults = dict(
        paper_bgcolor=Colors.BG_SURFACE_1,
        plot_bgcolor=Colors.BG_SURFACE_1,
        font=dict(
            family="'IBM Plex Sans', sans-serif",
            color=Colors.TEXT_PRIMARY,
            size=12,
        ),
        margin=dict(t=40, b=32, l=56, r=16),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=Colors.BG_SURFACE_2,
            bordercolor=Colors.BORDER_DEFAULT,
            font=dict(
                family="'JetBrains Mono', monospace",
                color=Colors.TEXT_PRIMARY,
                size=11,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=Colors.TEXT_SECONDARY),
        ),
        xaxis=dict(
            gridcolor=Colors.BORDER_SUBTLE,
            zeroline=False,
            tickfont=dict(family="'JetBrains Mono', monospace", size=10,
                          color=Colors.TEXT_SECONDARY),
            linecolor=Colors.BORDER_DEFAULT,
        ),
        yaxis=dict(
            gridcolor=Colors.BORDER_SUBTLE,
            zeroline=False,
            tickfont=dict(family="'JetBrains Mono', monospace", size=10,
                          color=Colors.TEXT_SECONDARY),
            linecolor=Colors.BORDER_DEFAULT,
        ),
    )
    defaults.update(overrides)
    return defaults


# ── Formatters ──────────────────────────────────────────────────────────

def fmt_dollar(v, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    abs_v = abs(v)
    sign = "-" if v < 0 else ""
    if abs_v >= 1e12:
        return f"{sign}${abs_v / 1e12:,.{decimals}f}T"
    if abs_v >= 1e9:
        return f"{sign}${abs_v / 1e9:,.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{sign}${abs_v / 1e6:,.{decimals}f}M"
    if abs_v >= 1e4:
        return f"{sign}${abs_v / 1e3:,.{decimals}f}K"
    return f"{sign}${abs_v:,.0f}"


def fmt_z(v, width: int = 6) -> str:
    """Right-aligned signed z-score with fixed column width for alignment."""
    if v is None or (isinstance(v, float) and v != v):
        return "  —   "
    s = f"{v:+.2f}"
    return s.rjust(width)


def pnl_color(v) -> str:
    if isinstance(v, (int, float)):
        if v > 0:
            return f"color: {Colors.POSITIVE}; font-weight: 500"
        if v < 0:
            return f"color: {Colors.NEGATIVE}; font-weight: 500"
    return f"color: {Colors.TEXT_SECONDARY}"


# ── CSS injection ───────────────────────────────────────────────────────

# Imports of Fraunces (variable serif), IBM Plex Sans, and JetBrains Mono
# in a single Google Fonts call to minimize the render-blocking fetch.
_FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=JetBrains+Mono:wght@400;500;600&display=swap"
)


def _build_css() -> str:
    c = Colors
    return f"""
<style>
    @import url('{_FONT_IMPORT}');

    /* ══════════════════════════════════════════════════════════════ */
    /*  Canvas + global typography                                    */
    /* ══════════════════════════════════════════════════════════════ */
    html, body, .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {c.BG_BASE} !important;
        color: {c.TEXT_PRIMARY} !important;
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        font-feature-settings: 'tnum' 1, 'ss01' 1;
    }}

    /* Subtle noise texture overlay — gives the canvas a physicality
       that solid colors never achieve. Very low opacity. */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            radial-gradient(circle at 20% 30%, rgba(232,184,78,0.015), transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(95,185,138,0.012), transparent 50%);
        z-index: 0;
    }}

    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
    }}

    .block-container {{
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1500px;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Type system                                                    */
    /* ══════════════════════════════════════════════════════════════ */
    h1, h2, h3, h4 {{
        font-family: 'Fraunces', Georgia, serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em;
        color: {c.TEXT_PRIMARY} !important;
        font-variation-settings: 'opsz' 96;
    }}
    h1 {{ font-size: 2.4rem; margin: 0.2rem 0 0.2rem 0 !important; }}
    h2 {{ font-size: 1.35rem; margin: 2rem 0 0.6rem 0 !important;
          padding-top: 1rem;
          border-top: 1px solid {c.BORDER_SUBTLE}; }}
    h3 {{ font-size: 1.05rem; font-weight: 600 !important;
          margin: 1.2rem 0 0.5rem 0 !important;
          color: {c.TEXT_SECONDARY} !important;
          letter-spacing: 0.05em; text-transform: uppercase;
          font-family: 'IBM Plex Sans', sans-serif !important; }}

    code, pre, .stDataFrame td, .stDataFrame th,
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', 'Menlo', monospace !important;
        font-feature-settings: 'tnum' 1, 'calt' 0;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Masthead — the hand-crafted top strip                          */
    /* ══════════════════════════════════════════════════════════════ */
    .masthead {{
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding: 0 0 0.9rem 0;
        margin: -0.2rem 0 1.4rem 0;
        border-bottom: 1px solid {c.ACCENT_SOFT};
        position: relative;
    }}
    .masthead::after {{
        content: '';
        position: absolute;
        left: 0; right: 0; bottom: -3px;
        height: 1px;
        background: linear-gradient(90deg,
            transparent 0%, {c.ACCENT} 50%, transparent 100%);
        opacity: 0.25;
    }}
    .masthead__brand {{
        font-family: 'Fraunces', serif;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        line-height: 1;
        font-variation-settings: 'opsz' 144;
        color: {c.TEXT_PRIMARY};
    }}
    .masthead__brand .ticker {{
        color: {c.ACCENT};
        font-style: italic;
    }}
    .masthead__brand .sep {{
        color: {c.TEXT_MUTED};
        font-weight: 400;
        margin: 0 0.35em;
    }}
    .masthead__tagline {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.72rem;
        color: {c.TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.25em;
        margin-top: 0.25rem;
    }}
    .masthead__meta {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: {c.TEXT_SECONDARY};
        text-align: right;
        line-height: 1.55;
    }}
    .masthead__meta .label {{
        color: {c.TEXT_MUTED};
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-size: 0.62rem;
        margin-right: 0.4rem;
    }}
    .masthead__meta .val {{ color: {c.TEXT_PRIMARY}; }}
    .masthead__meta .val.accent {{ color: {c.ACCENT}; }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Tier chips                                                     */
    /* ══════════════════════════════════════════════════════════════ */
    .tier-chip {{
        display: inline-block;
        padding: 0.12rem 0.55rem;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        border: 1px solid;
    }}
    .tier-chip--bull {{
        color: {c.POSITIVE};
        border-color: {c.POSITIVE}66;
        background: {c.POSITIVE}14;
        box-shadow: inset 0 0 8px {c.POSITIVE}22;
    }}
    .tier-chip--bear {{
        color: {c.NEGATIVE};
        border-color: {c.NEGATIVE}66;
        background: {c.NEGATIVE}14;
        box-shadow: inset 0 0 8px {c.NEGATIVE}22;
    }}
    .tier-chip--none {{
        color: {c.TEXT_MUTED};
        border-color: {c.BORDER_DEFAULT};
        background: transparent;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Dataframes — institutional ledger style                        */
    /* ══════════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {{
        border: 1px solid {c.BORDER_SUBTLE};
        border-radius: 2px;
        overflow: hidden;
    }}
    [data-testid="stDataFrame"] .row-widget {{
        background: {c.BG_SURFACE_1};
    }}
    .stDataFrame thead tr th {{
        background: {c.BG_SURFACE_2} !important;
        color: {c.TEXT_MUTED} !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        border-bottom: 1px solid {c.BORDER_DEFAULT} !important;
    }}
    .stDataFrame tbody tr td {{
        font-size: 0.85rem !important;
        border-bottom: 1px solid {c.BORDER_SUBTLE} !important;
    }}
    .stDataFrame tbody tr:hover td {{
        background: {c.BG_SURFACE_2} !important;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Metric cards                                                   */
    /* ══════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {{
        background: {c.BG_SURFACE_1};
        border: 1px solid {c.BORDER_SUBTLE};
        border-left: 2px solid {c.ACCENT_SOFT};
        padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        border-radius: 2px;
    }}
    [data-testid="stMetricLabel"] {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-size: 0.6rem !important;
        color: {c.TEXT_MUTED} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.6rem !important;
        font-weight: 500 !important;
        color: {c.TEXT_PRIMARY} !important;
        letter-spacing: -0.01em;
    }}
    [data-testid="stMetricDelta"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Select boxes + inputs                                          */
    /* ══════════════════════════════════════════════════════════════ */
    .stSelectbox label, .stRadio label {{
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: {c.TEXT_MUTED} !important;
    }}
    [data-baseweb="select"] > div {{
        background: {c.BG_SURFACE_1} !important;
        border-color: {c.BORDER_DEFAULT} !important;
        border-radius: 2px !important;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Section number flourish                                        */
    /* ══════════════════════════════════════════════════════════════ */
    .section-num {{
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-size: 0.85rem;
        color: {c.ACCENT};
        margin-right: 0.6rem;
        vertical-align: 0.2em;
    }}

    /* ══════════════════════════════════════════════════════════════ */
    /*  Pill badges for rationale / status                             */
    /* ══════════════════════════════════════════════════════════════ */
    .pill {{
        display: inline-block;
        padding: 0.05rem 0.45rem;
        margin-right: 0.3rem;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        border: 1px solid {c.BORDER_SUBTLE};
        color: {c.TEXT_SECONDARY};
        background: {c.BG_SURFACE_2};
    }}
    .pill.pos {{ color: {c.POSITIVE}; border-color: {c.POSITIVE}44; }}
    .pill.neg {{ color: {c.NEGATIVE}; border-color: {c.NEGATIVE}44; }}
    .pill.warn {{ color: {c.ACCENT}; border-color: {c.ACCENT}44; }}

    /* Scrollbars — bring them in line with the palette */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {c.BG_BASE}; }}
    ::-webkit-scrollbar-thumb {{
        background: {c.BORDER_DEFAULT};
        border-radius: 2px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: {c.BORDER_STRONG}; }}

    /* Hide Streamlit default footer + menu — terminal UI aesthetic */
    #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def inject_css() -> None:
    """Inject the institutional stylesheet. Call once at the top of the app."""
    import streamlit as st
    st.markdown(_build_css(), unsafe_allow_html=True)


def tier_chip_html(tier: str) -> str:
    """Return HTML for a tier chip (bull / bear / neutral)."""
    if "BULL" in tier:
        cls = "tier-chip--bull"
        label = "LONG"
    elif "BEAR" in tier:
        cls = "tier-chip--bear"
        label = "SHORT"
    else:
        cls = "tier-chip--none"
        label = "WATCH"
    return f'<span class="tier-chip {cls}">{label}</span>'

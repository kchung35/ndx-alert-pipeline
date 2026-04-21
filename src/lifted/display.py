"""Column display helpers.

Lifted from pms_app/engine/display.py — pared back to the columns
this project actually emits (alerts, factors, options metrics, insider
transactions). Falls back to Title Case for any column not explicitly
mapped so the dashboard never shows raw snake_case.
"""

from __future__ import annotations

COLUMN_DISPLAY_NAMES: dict[str, str] = {
    "ticker": "Ticker",
    "company": "Company",
    "sector": "Sector",
    "industry": "Industry",
    "date": "Date",
    "tier": "Tier",
    "composite": "Composite",
    "factor_z": "Factor z",
    "options_z": "Options z",
    "insider_z": "Insider z",
    "rationale": "Rationale",

    "momentum_12_1": "Momentum 12-1",
    "value_z": "Value z",
    "quality_z": "Quality z",
    "lowvol_z": "Low-vol z",
    "trailing_pe": "Trailing P/E",
    "forward_pe": "Forward P/E",
    "price_to_book": "P/B",
    "ev_to_ebitda": "EV/EBITDA",
    "roe": "ROE",
    "roa": "ROA",
    "gross_margin": "Gross Margin",
    "debt_to_equity": "Debt/Equity",

    "vol_oi_call": "V/OI (calls)",
    "vol_oi_put": "V/OI (puts)",
    "flow_ratio": "Flow ratio ($)",
    "iv_skew": "IV skew",
    "iv_term": "IV term",
    "iv_rel_vix": "IV / VIX",

    "insider_name": "Insider",
    "position": "Position",
    "filing_date": "Filing Date",
    "transaction_date": "Tx Date",
    "tx_code": "Code",
    "shares": "Shares",
    "price": "Price",
    "value": "Value",
    "signal_label": "Signal",
    "signal_weight": "Weight",
    "officer_weight": "Officer Wt",
    "is_derivative": "Derivative",
    "is_10b5_1": "10b5-1",
}


def nice_columns(col_name: str) -> str:
    """Map an internal column name to its display name."""
    if col_name in COLUMN_DISPLAY_NAMES:
        return COLUMN_DISPLAY_NAMES[col_name]
    return col_name.replace("_", " ").title()


def nice_format(fmt_map: dict[str, str]) -> dict[str, str]:
    """Translate a {internal_col: fmt_spec} dict to {display_col: fmt_spec}."""
    return {nice_columns(k): v for k, v in fmt_map.items()}

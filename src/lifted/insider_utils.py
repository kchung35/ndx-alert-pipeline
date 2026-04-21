"""Insider-transaction utilities — pure classifier + corporate entity filter.

Lifted surgically from pms_app/engine/insiders.py (_CODE_CLASSIFICATION,
_OFFICER_RANK, _officer_weight, classify_transaction) and
pms_app/engine/insider_analytics.py (_is_corporate_entity,
_is_corporate_entity_series). Everything here is pure — no DB, no EDGAR
fetch, no Streamlit cache. The actual Form 4 fetch is done by
src/data_edgar.py using the raw SEC REST API.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ── Transaction code classification ──────────────────────────────────────
# Weight: positive = bullish, negative = bearish, 0 = noise.
_CODE_CLASSIFICATION: dict[str, tuple[str, int]] = {
    "P": ("Open Market Purchase", 3),
    "S": ("Open Market Sale", -2),
    "M": ("Option Exercise", 0),
    "X": ("Option Exercise", 0),
    "F": ("Tax Withholding", 0),
    "A": ("Award/Grant", 0),
    "G": ("Gift", 0),
    "C": ("Conversion", 0),
    "D": ("Disposition to Issuer", 0),
    "E": ("Expiration", 0),
    "H": ("Expiration (Long)", 0),
    "I": ("Other Disposition", -1),
    "J": ("Other Acquisition", 1),
    "U": ("Tender Disposition", -1),
    "Z": ("Voting Trust", 0),
}

# ── Officer rank weights (higher = more senior / more informative) ──────
_OFFICER_RANK: dict[str, int] = {
    "ceo": 5,
    "chief executive": 5,
    "chairman": 4,
    "cfo": 4,
    "chief financial": 4,
    "president": 4,
    "coo": 3,
    "chief operating": 3,
    "cto": 3,
    "chief technology": 3,
    "evp": 3,
    "svp": 2,
    "vp": 2,
    "gm": 2,
    "general manager": 2,
    "general counsel": 3,
    "director": 1,
    "10% owner": 2,
}

_OFFICER_RANK_MULTIWORD = {k: v for k, v in _OFFICER_RANK.items() if " " in k or len(k) > 3}
_OFFICER_RANK_SHORT = {k: v for k, v in _OFFICER_RANK.items() if " " not in k and len(k) <= 3}
_OFFICER_RANK_SHORT_PATTERNS = {
    k: re.compile(rf"\b{re.escape(k)}\b") for k in _OFFICER_RANK_SHORT
}


def officer_weight(position: str) -> int:
    """Score insider's position. Higher = more senior. Fallback = 1."""
    if not position:
        return 1
    pos_lower = position.lower()
    best = 1
    for key, weight in _OFFICER_RANK_MULTIWORD.items():
        if key in pos_lower:
            best = max(best, weight)
    for key, weight in _OFFICER_RANK_SHORT.items():
        if _OFFICER_RANK_SHORT_PATTERNS[key].search(pos_lower):
            best = max(best, weight)
    return best


def classify_transaction(
    code: str,
    is_derivative: bool = False,
    is_10b5_1: Optional[bool] = None,
) -> tuple[str, int]:
    """Classify a Form 4 transaction.

    Returns:
        (label, weight) — weight > 0 is bullish, < 0 bearish, 0 noise.
    """
    label, weight = _CODE_CLASSIFICATION.get(code, ("Other", 0))
    if code == "S" and is_10b5_1 is True:
        label = "10b5-1 Plan Sale"
        weight = -1
    if is_derivative and code not in ("P", "S"):
        weight = 0
    return label, weight


# ── Corporate entity filter ─────────────────────────────────────────────
# SEC Form 4 uses compound names for beneficial ownership chains:
#   "BERKSHIRE HATHAWAY INC / Warren E Buffett" IS a real insider (Buffett).
#   "GENERAL ELECTRIC CO" is a pure corporate entity.
# We keep chains containing a real person and drop pure corporate entities.
_CORP_ENTITY_KEYWORDS: tuple[str, ...] = (
    " INC.", " INC,", " INC ",
    " CORP.", " CORP,", " CORP ",
    " CO.", " CO,", " CO ",
    " LLC", " LP ", " LP,", " LTD", " L.P.", " L.L.C.",
    " N.V.", " PLC", " S.A.", " AG ", " GMBH", " B.V.", " S.A.R.L",
    " GROUP ", " GROUP,",
    " PARTNERSHIP",
    " HOLDINGS,", " HOLDINGS ",
    " INVESTMENTS,", " INVESTMENTS ",
)

_CORP_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in _CORP_ENTITY_KEYWORDS),
    re.IGNORECASE,
)


def _segment_is_corporate(segment: str) -> bool:
    seg = segment.strip()
    if not seg:
        return True
    upper = f" {seg.upper()} "
    return any(kw in upper for kw in _CORP_ENTITY_KEYWORDS)


def is_corporate_entity(name: str) -> bool:
    """True if insider_name is a corporate entity (no real person in the chain)."""
    if not name or not isinstance(name, str):
        return False
    upper = f" {name.upper()} "
    if not any(kw in upper for kw in _CORP_ENTITY_KEYWORDS):
        return False
    for seg in name.split("/"):
        if not _segment_is_corporate(seg):
            return False
    return True


def is_corporate_entity_series(names: pd.Series) -> pd.Series:
    """Vectorized version for DataFrame columns."""
    padded = " " + names.fillna("") + " "
    has_keyword = padded.str.contains(_CORP_PATTERN, na=False)
    # Default everyone to False, then flip only the rows flagged by the regex.
    # Allocating as object dtype avoids pandas' bool/object upcast FutureWarning
    # when we assign back an array of bools.
    result = pd.Series(False, index=names.index, dtype=object)
    if has_keyword.any():
        result.loc[has_keyword] = names.loc[has_keyword].apply(is_corporate_entity)
    return result.astype(bool)

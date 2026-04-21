"""Unit tests for lifted analytics — confirms the math survives the lift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.lifted.analytics import (
    annualized_return, annualized_vol, conditional_var, historical_var,
    max_drawdown, sharpe_ratio, value_at_risk,
)


def _gauss_returns(n=252, mu=0.0005, sigma=0.01, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=pd.date_range("2024-01-01", periods=n))


def test_annualized_return_positive_when_mu_positive():
    r = _gauss_returns(mu=0.001)
    assert annualized_return(r) > 0


def test_annualized_vol_close_to_theoretical():
    r = _gauss_returns(sigma=0.01)
    ann_v = annualized_vol(r)
    assert 0.10 < ann_v < 0.25


def test_sharpe_ratio_rf_zero_matches_ratio():
    r = _gauss_returns(mu=0.001, sigma=0.01)
    s = sharpe_ratio(r, rf_annual=0.0)
    assert s > 0


def test_var_positive_and_scales_with_horizon():
    r = _gauss_returns()
    var_1d = value_at_risk(r, confidence=0.95, horizon_days=1)
    var_10d = value_at_risk(r, confidence=0.95, horizon_days=10)
    assert var_1d > 0
    assert var_10d > var_1d


def test_historical_var_positive():
    r = _gauss_returns()
    assert historical_var(r, confidence=0.95, horizon_days=10) > 0


def test_cvar_ge_var():
    r = _gauss_returns()
    var = historical_var(r, confidence=0.95, horizon_days=10)
    cvar = conditional_var(r, confidence=0.95, horizon_days=10)
    assert cvar >= var


def test_max_drawdown_nonpositive():
    r = _gauss_returns()
    equity = (1 + r).cumprod()
    mdd = max_drawdown(equity)
    assert mdd <= 0


def test_handles_tiny_series():
    r = pd.Series([0.01])
    assert np.isnan(annualized_return(r))
    assert np.isnan(value_at_risk(r))

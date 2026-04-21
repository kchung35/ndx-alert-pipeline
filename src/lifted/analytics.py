"""Risk / performance analytics — pure math, no DB or Streamlit.

Lifted from pms_app/engine/analytics.py. Kept only the universally useful
functions; dropped portfolio-specific helpers (risk_decomposition,
daily_pnl_attribution, proxy_backfill_returns, compute_full_analytics,
blended_benchmark_return, _filter_trading_days) because this project has no
portfolio.

All functions use SIMPLE (arithmetic) returns, not log returns.
Simple returns are correct for portfolio aggregation (w1·r1 + w2·r2) and for
compounding via (1+r).prod().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns. Does NOT drop NaN — per-ticker NaN handling
    is delegated to consumers so a single sparse ticker cannot truncate
    the entire window."""
    rets = prices / prices.shift(1) - 1
    return rets.iloc[1:]


def annualized_return(returns: pd.Series, trading_days: int = 252) -> float:
    """Annualized return from daily simple returns via geometric compounding."""
    if len(returns) < 2:
        return np.nan
    cum = (1 + returns).prod()
    n_years = len(returns) / trading_days
    return cum ** (1 / n_years) - 1 if n_years > 0 else np.nan


def annualized_vol(returns: pd.Series, trading_days: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    return returns.std() * np.sqrt(trading_days)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0435,
                 trading_days: int = 252) -> float:
    ann_ret = annualized_return(returns, trading_days)
    ann_v = annualized_vol(returns, trading_days)
    if ann_v is None or ann_v == 0 or np.isnan(ann_v):
        return np.nan
    return (ann_ret - rf_annual) / ann_v


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0435,
                  trading_days: int = 252) -> float:
    ann_ret = annualized_return(returns, trading_days)
    downside = returns[returns < 0]
    if len(downside) < 2:
        return np.nan
    downside_vol = downside.std() * np.sqrt(trading_days)
    if downside_vol == 0:
        return np.nan
    return (ann_ret - rf_annual) / downside_vol


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown from peak."""
    if len(prices) < 2:
        return np.nan
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


def value_at_risk(returns: pd.Series, confidence: float = 0.95,
                  horizon_days: int = 10, portfolio_value: float = 1.0) -> float:
    """Parametric (Gaussian) VaR. Scales daily VaR to horizon via sqrt(T)."""
    if len(returns) < 10:
        return np.nan
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    daily_var = -(mu + z * sigma)
    return daily_var * np.sqrt(horizon_days) * portfolio_value


def historical_var(returns: pd.Series, confidence: float = 0.95,
                   horizon_days: int = 10, portfolio_value: float = 1.0) -> float:
    """Historical VaR. Uses overlapping multi-day returns when horizon > 1."""
    if len(returns) < 10:
        return np.nan
    if horizon_days > 1 and len(returns) >= horizon_days + 10:
        multi_day = returns.rolling(horizon_days).sum().dropna()
        if len(multi_day) >= 10:
            cutoff = multi_day.quantile(1 - confidence)
            return -cutoff * portfolio_value
    cutoff = returns.quantile(1 - confidence)
    return -cutoff * np.sqrt(horizon_days) * portfolio_value


def conditional_var(returns: pd.Series, confidence: float = 0.95,
                    horizon_days: int = 10, portfolio_value: float = 1.0) -> float:
    """Conditional VaR (Expected Shortfall) — mean loss in the tail beyond VaR.

    Coherent (subadditive) risk measure. Historical simulation when horizon > 1
    and enough data exists; otherwise sqrt(T) scaling.
    """
    if len(returns) < 10:
        return np.nan
    if horizon_days > 1 and len(returns) >= horizon_days + 10:
        multi_day = returns.rolling(horizon_days).sum().dropna()
        if len(multi_day) >= 10:
            cutoff = multi_day.quantile(1 - confidence)
            tail = multi_day[multi_day <= cutoff]
            return -tail.mean() * portfolio_value if len(tail) > 0 else np.nan
    cutoff = returns.quantile(1 - confidence)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return np.nan
    return -tail.mean() * np.sqrt(horizon_days) * portfolio_value


def beta_to_benchmark(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    if len(asset_returns) < 10 or len(bench_returns) < 10:
        return np.nan
    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan
    cov = aligned.cov().iloc[0, 1]
    var_bench = aligned.iloc[:, 1].var()
    return cov / var_bench if var_bench > 0 else np.nan


def tail_beta(asset_returns: pd.Series, bench_returns: pd.Series,
              threshold_sigma: float = -2.0) -> float:
    """Beta conditioned on market stress (benchmark down > threshold_sigma).

    Tail beta > 1 means the asset amplifies benchmark crashes.
    Default threshold -2 sigma ~ benchmark down ~2-3% in a day.
    """
    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 30:
        return np.nan
    bench = aligned.iloc[:, 1]
    cutoff = bench.mean() + threshold_sigma * bench.std()
    stress = aligned.loc[bench <= cutoff]
    if len(stress) < 5:
        return np.nan
    cov = stress.cov().iloc[0, 1]
    var_bench = stress.iloc[:, 1].var()
    return cov / var_bench if var_bench > 0 else np.nan

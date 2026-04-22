"""
metrics.py — Performance statistics for a strategy returns series.
"""

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """
    Compute comprehensive risk/return metrics from a strategy DataFrame
    that must contain at least 'strategy_returns' and 'positions' columns.

    Parameters
    ----------
    df               : output from strategy.compute_returns()
    periods_per_year : 252 (daily), 52 (weekly), 12 (monthly)
    """
    rets = df['strategy_returns'].dropna()

    empty = {
        'total_return'  : 0.0,
        'annual_return' : 0.0,
        'sharpe_ratio'  : 0.0,
        'sortino_ratio' : 0.0,
        'max_drawdown'  : 0.0,
        'calmar_ratio'  : 0.0,
        'win_rate'      : 0.0,
        'profit_factor' : 0.0,
        'num_trades'    : 0,
        'avg_trade_ret' : 0.0,
        'volatility'    : 0.0,
    }

    if rets.empty or rets.std() == 0 or len(rets) < 5:
        return empty

    # ── Returns ───────────────────────────────────────────────────────────────
    cum = (1 + rets).cumprod()
    total_return  = float(cum.iloc[-1] - 1) * 100
    n_years       = len(rets) / periods_per_year
    annual_return = float((cum.iloc[-1] ** (1 / max(n_years, 1/252)) - 1) * 100)
    volatility    = float(rets.std() * np.sqrt(periods_per_year) * 100)

    # ── Sharpe ────────────────────────────────────────────────────────────────
    rf_daily  = 0.02 / periods_per_year   # 2% annual risk-free rate
    excess    = rets - rf_daily
    sharpe    = float((excess.mean() / rets.std()) * np.sqrt(periods_per_year))

    # ── Sortino ───────────────────────────────────────────────────────────────
    downside  = rets[rets < rf_daily].std()
    sortino   = float((excess.mean() / downside) * np.sqrt(periods_per_year)) \
                if downside and downside > 0 else 0.0

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = float(drawdown.min() * 100)

    # ── Calmar ────────────────────────────────────────────────────────────────
    calmar    = float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0

    # ── Trade statistics ──────────────────────────────────────────────────────
    active = rets[rets != 0]          # only periods with an open position
    wins   = active[active > 0]
    losses = active[active < 0]

    win_rate      = float(len(wins) / len(active) * 100) if len(active) > 0 else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) \
                    if len(losses) > 0 and losses.sum() != 0 else 0.0
    avg_trade_ret = float(active.mean() * 100) if len(active) > 0 else 0.0

    # Number of distinct trades (position changes)
    pos = df.get('positions', pd.Series(dtype=float))
    num_trades = int((pos.diff().abs() > 0).sum())

    return {
        'total_return'  : round(total_return,  2),
        'annual_return' : round(annual_return, 2),
        'sharpe_ratio'  : round(sharpe,        3),
        'sortino_ratio' : round(sortino,        3),
        'max_drawdown'  : round(max_dd,         2),
        'calmar_ratio'  : round(calmar,         3),
        'win_rate'      : round(win_rate,       2),
        'profit_factor' : round(profit_factor,  3),
        'num_trades'    : num_trades,
        'avg_trade_ret' : round(avg_trade_ret,  4),
        'volatility'    : round(volatility,     2),
    }

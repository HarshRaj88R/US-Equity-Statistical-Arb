"""
strategy.py — Bollinger-Band mean-reversion signal generation.

NO LOOKAHEAD BIAS GUARANTEES:
  1. Rolling mean/std use only the past `lookback` periods (right-closed window).
  2. Signals are generated at the CLOSE of day t.
  3. Positions are shifted by 1 period before computing returns (executed at t+1 open).
  4. OLS parameters (alpha, beta) must be estimated on training data before calling here.
"""

import numpy as np
import pandas as pd


def generate_signals(spread: pd.Series, lookback: int,
                     entry_std: float, exit_std: float) -> pd.DataFrame:
    """
    Generate long/short positions using Bollinger Bands on the spread.

    Entry rules  (executed next period):
      Long  : spread drops below  (MA − entry_std × STD)
      Short : spread rises above  (MA + entry_std × STD)

    Exit rules  (executed next period):
      Long  exit : spread rises back to (MA − exit_std × STD)
      Short exit : spread falls back to (MA + exit_std × STD)

    Parameters
    ----------
    spread     : pre-computed residual series (Y − β·X − α)
    lookback   : rolling window length in periods
    entry_std  : Bollinger entry threshold in σ  (e.g. 2.0)
    exit_std   : Bollinger exit  threshold in σ  (e.g. 0.5)  ← tighter
    """
    df = pd.DataFrame({'spread': spread}, index=spread.index)

    # Rolling stats — min_periods=lookback prevents using a partial window
    df['ma']  = df['spread'].rolling(lookback, min_periods=lookback).mean()
    df['std'] = df['spread'].rolling(lookback, min_periods=lookback).std()

    # Entry bands
    df['upper_entry'] = df['ma'] + entry_std * df['std']
    df['lower_entry'] = df['ma'] - entry_std * df['std']

    # Exit bands (tighter, closer to mean)
    df['upper_exit'] = df['ma'] + exit_std * df['std']
    df['lower_exit'] = df['ma'] - exit_std * df['std']

    # ── Long leg ──────────────────────────────────────────────────────────────
    # Enter long: spread below lower entry band
    # Exit long : spread climbs back to lower exit band (mean - exit_std * σ)
    df['long_entry'] = df['spread'] < df['lower_entry']
    df['long_exit']  = df['spread'] >= df['lower_exit']

    df['pos_long'] = np.nan
    df.loc[df['long_entry'], 'pos_long'] = 1
    df.loc[df['long_exit'],  'pos_long'] = 0
    # Forward-fill: hold position until an explicit exit signal
    df['pos_long'] = df['pos_long'].ffill().fillna(0)

    # ── Short leg ─────────────────────────────────────────────────────────────
    # Enter short: spread above upper entry band
    # Exit short : spread falls back to upper exit band (mean + exit_std * σ)
    df['short_entry'] = df['spread'] > df['upper_entry']
    df['short_exit']  = df['spread'] <= df['upper_exit']

    df['pos_short'] = np.nan
    df.loc[df['short_entry'], 'pos_short'] = -1
    df.loc[df['short_exit'],  'pos_short'] = 0
    df['pos_short'] = df['pos_short'].ffill().fillna(0)

    # ── Combined position ─────────────────────────────────────────────────────
    # Clip to [-1, 1] to resolve simultaneous long+short conflicts
    df['positions'] = (df['pos_long'] + df['pos_short']).clip(-1, 1)

    return df


def compute_returns(signal_df: pd.DataFrame, beta: float,
                    price_x: pd.Series, price_y: pd.Series) -> pd.DataFrame:
    """
    Compute daily P&L from the spread position.

    spread P&L      = Δspread  = spread_t − spread_{t-1}
    capital deployed = |Y| + β·|X|  (notional of both legs)
    pct_change       = spread P&L / capital
    strategy_returns = position_{t-1} × pct_change_t   ← shift avoids lookahead

    The shift(1) ensures we only trade on the signal from the PREVIOUS period,
    matching real-world execution at the next open.
    """
    df = signal_df.copy()

    # Align price series to df index
    df['price_y'] = price_y.reindex(df.index)
    df['price_x'] = price_x.reindex(df.index)

    spread_delta  = df['spread'].diff()
    total_capital = df['price_y'].abs() + beta * df['price_x'].abs()

    df['pct_change']       = spread_delta / total_capital
    df['strategy_returns'] = df['positions'].shift(1) * df['pct_change']
    df['cumulative_returns'] = (1 + df['strategy_returns'].fillna(0)).cumprod()

    return df

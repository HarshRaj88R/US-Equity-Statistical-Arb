"""
backtest.py — Backtesting engine with train / forward-test split across timeframes.

NO LOOKAHEAD GUARANTEE:
  • OLS regression is fitted ONLY on the training window.
  • The same alpha & beta are then applied to the test window.
  • Bollinger-band statistics (rolling mean / std) use min_periods=lookback,
    so the first `lookback` rows of each period produce no signal.
  • Positions are shifted by 1 period before multiplying by returns.
"""

import pandas as pd
import numpy as np

from config import TIMEFRAMES, PERIODS_PER_YEAR, MIN_PERIODS
from data_loader import resample_pair, split_data
from cointegration import ols_regression, compute_spread, adf_test
from strategy import generate_signals, compute_returns
from metrics import compute_metrics


# ─── Single Period Backtest ───────────────────────────────────────────────────

def run_period(price_y: pd.Series, price_x: pd.Series,
               alpha: float, beta: float,
               lookback: int, entry_std: float, exit_std: float,
               periods_per_year: int) -> dict:
    """
    Apply pre-fitted OLS params to price_y / price_x, generate signals,
    and compute performance metrics for one period (train OR test).
    """
    spread     = compute_spread(price_y, price_x, alpha, beta)
    adf_result = adf_test(spread)
    signal_df  = generate_signals(spread, lookback, entry_std, exit_std)
    result_df  = compute_returns(signal_df, beta, price_x, price_y)
    perf       = compute_metrics(result_df, periods_per_year)

    return {
        'df'      : result_df,
        'spread'  : spread,
        'adf'     : adf_result,
        'metrics' : perf,
    }


# ─── Full Analysis Across All Timeframes ─────────────────────────────────────

def run_full_analysis(data: pd.DataFrame,
                      ticker_y: str, ticker_x: str,
                      lookback: int   = 30,
                      entry_std: float = 2.0,
                      exit_std: float  = 0.5,
                      train_ratio: float = 0.70) -> dict:
    """
    For every configured timeframe:
      1. Resample prices to that frequency.
      2. Time-split into train / test.
      3. Fit OLS on TRAIN only  → alpha, beta.
      4. Run strategy on train  → backtest result.
      5. Apply same params to test → forward-test result.

    Returns
    -------
    dict keyed by timeframe label, each value is:
      {
        'train' : {df, spread, adf, metrics},
        'test'  : {df, spread, adf, metrics},
        'alpha' : float,
        'beta'  : float,
        'model' : OLS model object,
        'split_date': Timestamp,  # first date of test period
      }
    or None if the timeframe has insufficient data.
    """
    price_y_all = data[ticker_y]
    price_x_all = data[ticker_x]

    results = {}

    for tf_label, tf_freq in TIMEFRAMES.items():
        ppy      = PERIODS_PER_YEAR[tf_label]
        min_rows = MIN_PERIODS[tf_label]

        # ── Resample ──────────────────────────────────────────────────────────
        py_all, px_all = resample_pair(price_y_all, price_x_all, tf_freq)

        if len(py_all) < min_rows * 2:
            results[tf_label] = None
            continue

        # ── Time-based split ──────────────────────────────────────────────────
        py_tr, px_tr, py_te, px_te = split_data(py_all, px_all, train_ratio)

        if len(py_tr) < min_rows or len(py_te) < 5:
            results[tf_label] = None
            continue

        # ── OLS on TRAIN only ─────────────────────────────────────────────────
        reg         = ols_regression(py_tr, px_tr)
        alpha, beta = reg['alpha'], reg['beta']

        # ── Backtest (train period) ────────────────────────────────────────────
        train_result = run_period(py_tr, px_tr, alpha, beta,
                                  lookback, entry_std, exit_std, ppy)

        # ── Forward test (test period, same OLS params) ────────────────────────
        test_result  = run_period(py_te, px_te, alpha, beta,
                                  lookback, entry_std, exit_std, ppy)

        results[tf_label] = {
            'train'      : train_result,
            'test'       : test_result,
            'alpha'      : alpha,
            'beta'       : beta,
            'r_squared'  : reg['rsquared'],
            'model'      : reg['model'],
            'split_date' : py_te.index[0] if len(py_te) > 0 else None,
            'train_start': py_tr.index[0],
            'train_end'  : py_tr.index[-1],
            'test_start' : py_te.index[0]  if len(py_te) > 0 else None,
            'test_end'   : py_te.index[-1] if len(py_te) > 0 else None,
        }

    return results


# ─── CLI Helper ──────────────────────────────────────────────────────────────

def print_results(results: dict, ticker_y: str, ticker_x: str) -> None:
    """Pretty-print backtest & forward-test metrics for all timeframes."""
    print(f"\n{'='*70}")
    print(f"  PAIR: {ticker_y} / {ticker_x}")
    print(f"{'='*70}")

    metric_keys = ['total_return', 'annual_return', 'sharpe_ratio',
                   'sortino_ratio', 'max_drawdown', 'num_trades', 'win_rate']

    for tf, res in results.items():
        if res is None:
            print(f"\n  [{tf}]  Insufficient data — skipped")
            continue

        print(f"\n  -- {tf.upper()} " + "-"*46)
        print(f"     OLS: alpha={res['alpha']:.4f}  beta={res['beta']:.4f}  "
              f"R2={res['r_squared']:.4f}")
        print(f"     Train: {res['train_start'].date()} -> {res['train_end'].date()}")
        print(f"     Test : {res['test_start'].date()} -> {res['test_end'].date()}")

        print(f"\n  {'Metric':<18} {'Backtest':>12} {'Forward Test':>14}")
        print(f"  {'-'*44}")
        for k in metric_keys:
            tr_val = res['train']['metrics'].get(k, 'N/A')
            te_val = res['test']['metrics'].get(k, 'N/A')
            unit   = '%' if 'return' in k or 'drawdown' in k or 'win_rate' in k else ''
            print(f"  {k:<18} {str(tr_val)+unit:>12} {str(te_val)+unit:>14}")

        # ADF summary
        adf_tr = res['train']['adf']
        adf_te = res['test']['adf']
        tr_v = "Cointegrated" if adf_tr['is_stationary'] else "Not Cointegrated"
        te_v = "Cointegrated" if adf_te['is_stationary'] else "Not Cointegrated"
        print(f"\n     ADF (train): p={adf_tr['p_value']}  [{tr_v}]")
        print(f"     ADF (test) : p={adf_te['p_value']}  [{te_v}]")

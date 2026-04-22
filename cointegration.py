"""
cointegration.py — OLS regression, ADF test, and pair screening.

No lookahead: OLS params are always estimated on training data only.
The ADF test is run on the resulting spread.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


# ─── OLS Regression ──────────────────────────────────────────────────────────

def ols_regression(price_y: pd.Series, price_x: pd.Series) -> dict:
    """
    Fit OLS:  Y = α + β·X + ε
    Returns a dict with model, alpha, beta, rsquared.
    """
    X = sm.add_constant(price_x.values)
    model = sm.OLS(price_y.values, X).fit()

    return {
        'model'    : model,
        'alpha'    : float(model.params[0]),
        'beta'     : float(model.params[1]),
        'rsquared' : float(model.rsquared),
        'pvalue_alpha': float(model.pvalues[0]),
        'pvalue_beta' : float(model.pvalues[1]),
    }


def compute_spread(price_y: pd.Series, price_x: pd.Series,
                   alpha: float, beta: float) -> pd.Series:
    """
    Compute spread = Y − β·X − α using pre-fitted OLS parameters.
    Call with training-set alpha/beta when evaluating a test period
    to avoid lookahead bias.
    """
    spread = price_y - beta * price_x - alpha
    spread.name = 'spread'
    return spread


# ─── ADF Test ────────────────────────────────────────────────────────────────

def adf_test(spread: pd.Series, significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity (mean-reversion).

    H₀: spread has a unit root (non-stationary / random walk)
    H₁: spread is stationary (mean-reverting) ← what we want

    A low p-value (< significance) → reject H₀ → spread is stationary
    → the pair is cointegrated → pairs trade is valid.
    """
    clean = spread.dropna()
    if len(clean) < 20:
        return {'error': 'insufficient data for ADF test'}

    result = adfuller(clean, autolag='AIC')
    stat, p_value, lags, nobs, crit_vals = result[:5]

    is_stationary = bool(p_value < significance)

    # Which critical value thresholds does the stat beat?
    passed_levels = [pct for pct, cv in crit_vals.items()
                     if stat < cv]

    return {
        'adf_statistic'  : round(float(stat), 4),
        'p_value'        : round(float(p_value), 6),
        'lags_used'      : int(lags),
        'n_observations' : int(nobs),
        'critical_values': {k: round(float(v), 4) for k, v in crit_vals.items()},
        'passed_levels'  : passed_levels,
        'is_stationary'  : is_stationary,
        'significance'   : significance,
        'verdict'        : '✅ Cointegrated' if is_stationary else '❌ Not Cointegrated',
    }


# ─── Pair Screener ───────────────────────────────────────────────────────────

def screen_pairs(data: pd.DataFrame, candidate_pairs: list[tuple],
                 significance: float = 0.05) -> pd.DataFrame:
    """
    For each candidate pair, fit OLS on the full history then run ADF.
    Returns a DataFrame sorted by p-value (most cointegrated first).

    Note: this uses the full data only for *screening*; when backtesting,
    OLS is re-fitted exclusively on the training period.
    """
    rows = []
    for ticker_y, ticker_x in candidate_pairs:
        if ticker_y not in data.columns or ticker_x not in data.columns:
            continue

        price_y = data[ticker_y].dropna()
        price_x = data[ticker_x].dropna()
        common  = price_y.index.intersection(price_x.index)
        if len(common) < 60:
            continue

        price_y = price_y.loc[common]
        price_x = price_x.loc[common]

        try:
            reg    = ols_regression(price_y, price_x)
            spread = compute_spread(price_y, price_x, reg['alpha'], reg['beta'])
            adf    = adf_test(spread, significance)

            if 'error' in adf:
                continue

            corr = float(np.corrcoef(price_y, price_x)[0, 1])

            rows.append({
                'pair'           : f"{ticker_y}/{ticker_x}",
                'ticker_y'       : ticker_y,
                'ticker_x'       : ticker_x,
                'p_value'        : adf['p_value'],
                'adf_statistic'  : adf['adf_statistic'],
                'cointegrated'   : adf['is_stationary'],
                'alpha'          : round(reg['alpha'], 4),
                'beta'           : round(reg['beta'], 4),
                'r_squared'      : round(reg['rsquared'], 4),
                'correlation'    : round(corr, 4),
                'n_obs'          : adf['n_observations'],
            })
        except Exception as e:
            print(f"  Warning: {ticker_y}/{ticker_x} skipped — {e}")
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('p_value').reset_index(drop=True)
    return df

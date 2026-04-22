"""
data_loader.py — Download and cache price data from Yahoo Finance.
"""

import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def download_universe(candidate_pairs: list[tuple], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for every unique ticker in candidate_pairs.
    Returns a DataFrame with one column per ticker, rows = trading days.
    """
    tickers = sorted(set(t for pair in candidate_pairs for t in pair))

    print(f"\n{'='*60}")
    print(f"  Downloading {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"  Period: {start} -> {end}")
    print(f"{'='*60}")

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)

    # yfinance returns multi-level columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw['Close']
    else:
        data = raw[['Close']]
        data.columns = tickers  # single ticker edge-case

    data.dropna(how='all', inplace=True)

    available = data.columns.tolist()
    print(f"  Downloaded {len(data)} rows  |  {len(available)} tickers available")

    return data


def get_pair_data(data: pd.DataFrame, ticker_y: str, ticker_x: str) -> pd.DataFrame:
    """
    Extract a clean two-column DataFrame for a specific pair.
    Drops rows where either price is NaN.
    """
    if ticker_y not in data.columns or ticker_x not in data.columns:
        raise KeyError(f"One or both tickers not in dataset: {ticker_y}, {ticker_x}")
    return data[[ticker_y, ticker_x]].dropna()


def resample_pair(price_y: pd.Series, price_x: pd.Series,
                  freq: str) -> tuple[pd.Series, pd.Series]:
    """
    Resample two price series to a lower frequency using period-end (last) prices.
    freq should be a pandas offset alias: 'D', 'W', 'ME', etc.
    Returns aligned series with no NaN rows.
    """
    if freq == 'D':
        return price_y, price_x

    # Resample each to period-end, then inner-join to keep only common dates
    py = price_y.resample(freq).last()
    px = price_x.resample(freq).last()
    combined = pd.concat([py, px], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]


def split_data(price_y: pd.Series, price_x: pd.Series,
               train_ratio: float) -> tuple:
    """
    Time-based (not random) split.
    Returns (py_train, px_train, py_test, px_test).
    """
    n = len(price_y)
    split = int(n * train_ratio)
    return (price_y.iloc[:split], price_x.iloc[:split],
            price_y.iloc[split:],  price_x.iloc[split:])

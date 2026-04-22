"""
config.py — Central configuration for the Pairs Trading System
Edit CANDIDATE_PAIRS, dates, and strategy parameters here.
"""

# ─── Universe of US stock/ETF pairs to screen ───────────────────────────────
CANDIDATE_PAIRS = [
    # Sector ETFs
    ('XLF', 'KBE'),    # Financials vs Banks ETF
    ('XLE', 'OIH'),    # Energy vs Oil Services
    ('XLK', 'QQQ'),    # Tech ETF vs NASDAQ
    ('XLV', 'IBB'),    # Health vs Biotech
    ('XLU', 'IDU'),    # Utilities ETFs
    # Country ETFs
    ('EWC', 'EWA'),    # Canada vs Australia
    ('EWG', 'EWQ'),    # Germany vs France
    ('EWJ', 'EWY'),    # Japan vs South Korea
    # Industry peers
    ('XOM', 'CVX'),    # Oil majors
    ('KO',  'PEP'),    # Beverages
    ('BAC', 'JPM'),    # Big banks
    ('WMT', 'TGT'),    # Retailers
    ('MCD', 'YUM'),    # Fast food
    ('HD',  'LOW'),    # Home improvement
    ('JNJ', 'PFE'),    # Pharma
    ('GS',  'MS'),     # Investment banks
    ('COST','WMT'),    # Wholesale vs retail
    ('MA',  'V'),      # Payment networks
    # Precious metals
    ('GLD', 'SLV'),    # Gold vs Silver
    ('GLD', 'GDX'),    # Gold ETF vs Miners
    # Broad market
    ('SPY', 'IVV'),    # Two S&P 500 ETFs
    ('VTI', 'ITOT'),   # Total market ETFs
    ('DIA', 'SPY'),    # Dow vs S&P 500
]

# ─── Data window ─────────────────────────────────────────────────────────────
START_DATE  = '2015-01-01'
END_DATE    = '2024-01-01'

# ─── Strategy parameters ─────────────────────────────────────────────────────
LOOKBACK    = 30       # rolling window length (in periods — days/weeks/months)
ENTRY_STD   = 2.0      # entry band (σ from mean)
EXIT_STD    = 0.5      # exit band  (σ from mean)  ← tighter exit
STD_DEV     = ENTRY_STD  # alias kept for backwards compatibility

# ─── Train / Forward-test split ───────────────────────────────────────────────
TRAIN_RATIO = 0.70     # first 70 % = backtest, last 30 % = forward test

# ─── ADF significance level ───────────────────────────────────────────────────
SIGNIFICANCE = 0.05

# ─── Resampling frequencies (pandas offset aliases) ──────────────────────────
TIMEFRAMES = {
    'Daily'   : 'D',
    'Weekly'  : 'W',
    'Monthly' : 'ME',   # 'ME' = month-end (pandas >= 2.2); use 'M' if older
}

# ─── Annualisation constants per timeframe ────────────────────────────────────
PERIODS_PER_YEAR = {
    'Daily'   : 252,
    'Weekly'  : 52,
    'Monthly' : 12,
}

# ─── Minimum number of periods required to run strategy ──────────────────────
MIN_PERIODS = {
    'Daily'   : 60,
    'Weekly'  : 20,
    'Monthly' : 10,
}

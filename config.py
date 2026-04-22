"""
config.py — Central configuration for the Pairs Trading System
Edit CANDIDATE_PAIRS, dates, and strategy parameters here.
"""

# ─── Universe of US stock/ETF pairs to screen ───────────────────────────────
CANDIDATE_PAIRS = [
    # --- Broad market / near-identical index trackers (strongest cointegration) ---
    ('SPY',  'IVV'),   # S&P 500 — two fund managers, same index
    ('SPY',  'VOO'),   # S&P 500 — Vanguard vs SPDR
    ('IVV',  'VOO'),   # S&P 500 — iShares vs Vanguard
    ('QQQ',  'QQQM'),  # Nasdaq-100 — original vs mini
    ('VTI',  'ITOT'),  # US total market — Vanguard vs iShares
    ('DIA',  'SPY'),   # Dow vs S&P 500
    ('IWM',  'VB'),    # Small-cap — iShares vs Vanguard

    # --- Sector ETF twins (same sector, different issuer) ---
    ('XLK',  'VGT'),   # Technology
    ('XLF',  'VFH'),   # Financials
    ('XLE',  'VDE'),   # Energy
    ('XLV',  'VHT'),   # Health Care
    ('XLU',  'VPU'),   # Utilities
    ('XLI',  'VIS'),   # Industrials
    ('XLB',  'VAW'),   # Materials
    ('XLP',  'VDC'),   # Consumer Staples
    ('XLY',  'VCR'),   # Consumer Discretionary
    ('XLRE', 'VNQ'),   # Real Estate

    # --- Sector vs sub-sector ---
    ('XLF',  'KBE'),   # Financials vs Banks
    ('XLK',  'QQQ'),   # Tech ETF vs Nasdaq
    ('XLV',  'IBB'),   # Health vs Biotech
    ('XLU',  'IDU'),   # Utilities ETFs
    ('XLE',  'OIH'),   # Energy vs Oil Services

    # --- Country ETFs ---
    ('EWC',  'EWA'),   # Canada vs Australia
    ('EWG',  'EWL'),   # Germany vs Switzerland
    ('EWJ',  'EWY'),   # Japan vs South Korea
    ('EWU',  'EWP'),   # UK vs Spain
    ('EEM',  'VWO'),   # Emerging markets — iShares vs Vanguard

    # --- Fixed income twins ---
    ('TLT',  'IEF'),   # Long vs intermediate US treasuries
    ('SHY',  'IEI'),   # Short vs 3-7yr US treasuries
    ('LQD',  'VCIT'),  # Investment-grade corp bonds
    ('HYG',  'JNK'),   # High-yield corp bonds — iShares vs SPDR
    ('BND',  'AGG'),   # Total US bond market

    # --- Gold / precious metals ---
    ('GLD',  'IAU'),   # Two gold ETFs (nearly identical)
    ('GLD',  'GDX'),   # Gold spot vs gold miners
    ('SLV',  'SIVR'),  # Two silver ETFs
    ('GLD',  'SLV'),   # Gold vs silver

    # --- Oil & energy commodities ---
    ('USO',  'BNO'),   # US vs Brent crude oil ETFs
    ('XOM',  'CVX'),   # Oil majors

    # --- Industry duopolies / close competitors ---
    ('MA',   'V'),     # Payment networks
    ('KO',   'PEP'),   # Beverages
    ('HD',   'LOW'),   # Home improvement
    ('BAC',  'JPM'),   # Big banks
    ('GS',   'MS'),    # Investment banks
    ('WMT',  'TGT'),   # Retailers
    ('MCD',  'YUM'),   # Fast food
    ('JNJ',  'PFE'),   # Large-cap pharma
    ('ABBV', 'MRK'),   # Large-cap pharma 2
    ('COST', 'WMT'),   # Warehouse vs traditional retail
    ('AMZN', 'EBAY'),  # E-commerce
    ('T',    'VZ'),    # Telecom duopoly
    ('NEE',  'DUK'),   # Utilities
    ('BRK-B','SPY'),   # Berkshire vs S&P 500
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
SIGNIFICANCE = 0.10   # 10% gives more pairs; tighten to 0.05 in the sidebar

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

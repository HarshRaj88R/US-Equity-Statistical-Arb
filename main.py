"""
main.py — CLI entry point. Run without the Streamlit dashboard.

Usage:
    python main.py                          # uses config.py defaults
    python main.py --pair EWC EWA           # specific pair
    python main.py --pair XOM CVX --lookback 20 --entry 2.0 --exit 0.5
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (CANDIDATE_PAIRS, START_DATE, END_DATE,
                    LOOKBACK, ENTRY_STD, EXIT_STD, TRAIN_RATIO,
                    SIGNIFICANCE)
from data_loader import download_universe
from cointegration import screen_pairs
from backtest import run_full_analysis, print_results


def parse_args():
    p = argparse.ArgumentParser(description="Pairs Trading — CLI")
    p.add_argument('--pair',     nargs=2, metavar=('Y', 'X'), default=None,
                   help="Pair to analyse, e.g. --pair EWC EWA")
    p.add_argument('--start',    default=START_DATE)
    p.add_argument('--end',      default=END_DATE)
    p.add_argument('--lookback', type=int,   default=LOOKBACK)
    p.add_argument('--entry',    type=float, default=ENTRY_STD)
    p.add_argument('--exit',     type=float, default=EXIT_STD,
                   dest='exit_std')
    p.add_argument('--train',    type=float, default=TRAIN_RATIO,
                   dest='train_ratio', help="Fraction of data for training (0-1)")
    p.add_argument('--screen',   action='store_true',
                   help="Screen all candidate pairs and print table")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Download ──────────────────────────────────────────────────────────────
    data = download_universe(CANDIDATE_PAIRS, args.start, args.end)

    # ── Screen mode ───────────────────────────────────────────────────────────
    if args.screen:
        print("\n  Screening all candidate pairs…")
        screened = screen_pairs(data, CANDIDATE_PAIRS, SIGNIFICANCE)
        if screened.empty:
            print("  No valid pairs found.")
            return
        print(f"\n{'-'*80}")
        print(f"  {'Pair':<14} {'p-value':>8} {'ADF Stat':>10} "
              f"{'beta':>7} {'R2':>7} {'Coint?':>10}")
        print(f"{'-'*80}")
        for _, r in screened.iterrows():
            tick = '[OK]' if r['cointegrated'] else '[--]'
            print(f"  {r['pair']:<14} {r['p_value']:>8.4f} {r['adf_statistic']:>10.4f} "
                  f"{r['beta']:>7.3f} {r['r_squared']:>7.4f}  {tick}")
        print(f"\n  Total: {len(screened)} pairs  |  "
              f"Cointegrated: {screened['cointegrated'].sum()}")
        return

    # ── Single pair mode ──────────────────────────────────────────────────────
    if args.pair:
        ticker_y, ticker_x = args.pair
    else:
        # Default: most cointegrated pair
        screened = screen_pairs(data, CANDIDATE_PAIRS, SIGNIFICANCE)
        if screened.empty or not screened['cointegrated'].any():
            print("  No cointegrated pairs found. Try --screen to see all pairs.")
            sys.exit(1)
        best     = screened[screened['cointegrated']].iloc[0]
        ticker_y = best['ticker_y']
        ticker_x = best['ticker_x']
        print(f"\n  Auto-selected most cointegrated pair: {ticker_y}/{ticker_x}  "
              f"(p={best['p_value']})")

    if ticker_y not in data.columns or ticker_x not in data.columns:
        print(f"  ERROR: {ticker_y} or {ticker_x} not available in downloaded data.")
        sys.exit(1)

    results = run_full_analysis(
        data, ticker_y, ticker_x,
        lookback=args.lookback,
        entry_std=args.entry,
        exit_std=args.exit_std,
        train_ratio=args.train_ratio,
    )

    print_results(results, ticker_y, ticker_x)


if __name__ == '__main__':
    main()

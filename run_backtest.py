# run_backtest.py
"""
Simple script to run backtesting with the ML models.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import argparse

from config.settings import WATCHLIST
from backtesting.comprehensive_backtester import run_ml_backtest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(description='Run ML Backtest')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest (default: all from watchlist)')
    parser.add_argument('--limit', type=int, help='Limit number of symbols (optional)')
    parser.add_argument('--all', action='store_true', help='Use all watchlist symbols (default behavior)')
    parser.add_argument('--days', type=int, default=730, help='Number of days to backtest (default: 730)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--position-size', type=float, default=0.02, help='Position size as fraction (default: 0.02)')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum positions (default: 10)')

    args = parser.parse_args()

    # Set symbols based on arguments
    if args.symbols:
        # Use explicitly provided symbols
        symbols = args.symbols
    elif args.limit:
        # Use limited number of symbols from watchlist
        symbols = WATCHLIST[:args.limit]
    else:
        # Default: use ALL symbols from watchlist
        symbols = WATCHLIST

    # Set dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    ML BACKTEST                           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Symbols: {len(symbols)} symbols                        ║
    ║  Period: {args.days} days                               ║
    ║  Capital: ${args.capital:,.0f}                          ║
    ║  Position Size: {args.position_size:.1%}                ║
    ║  Max Positions: {args.max_positions}                    ║
    ╚══════════════════════════════════════════════════════════╝

    Testing symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}
    """)

    # Create necessary directories
    Path('analysis/reports').mkdir(parents=True, exist_ok=True)

    # Run backtest
    print(f"\nStarting backtest for {len(symbols)} symbols...")
    print("This may take a while for large watchlists...\n")

    results = run_ml_backtest(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        train_test_split=0.7,
        temporal_gap_days=7,
        initial_capital=args.capital,
        position_size=args.position_size,
        max_positions=args.max_positions,
        stop_loss=0.05,
        take_profit=0.15,
        commission=0.001,
        slippage=0.001
    )

    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                  BACKTEST COMPLETE                       ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Total Return: {results.total_return:>8.2%}             ║
    ║  Sharpe Ratio: {results.sharpe_ratio:>8.2f}             ║
    ║  Max Drawdown: {results.max_drawdown:>8.2%}             ║
    ║  Win Rate: {results.win_rate:>8.1%}                     ║
    ║  Total Trades: {results.total_trades:>8d}               ║
    ║  Symbols Tested: {len(symbols):>6d}                     ║
    ╚══════════════════════════════════════════════════════════╝

    Results saved to:
    - data/backtest_results.json
    - analysis/reports/backtest_results.png
    """)


if __name__ == "__main__":
    main()
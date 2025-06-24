# run_analysis.py
"""Convenience script to run all analysis."""

import argparse
from analysis.multi_timeframe_predictions import create_enhanced_predictions
from analysis.view_predictions import view_all_predictions
from analysis.full_analysis import generate_full_analysis


def main():
    parser = argparse.ArgumentParser(description='Run trading analysis')
    parser.add_argument('--multi', action='store_true', help='Generate multi-timeframe predictions')
    parser.add_argument('--view', action='store_true', help='View current predictions')
    parser.add_argument('--full', action='store_true', help='Generate full analysis report')
    parser.add_argument('--all', action='store_true', help='Run all analyses')

    args = parser.parse_args()

    if args.all or args.multi:
        print("Generating multi-timeframe predictions...")
        create_enhanced_predictions()

    if args.all or args.view:
        print("\nViewing current predictions...")
        view_all_predictions()

    if args.all or args.full:
        print("\nGenerating full analysis report...")
        generate_full_analysis()

    if not any([args.multi, args.view, args.full, args.all]):
        print("No analysis specified. Use --help for options.")


if __name__ == "__main__":
    main()
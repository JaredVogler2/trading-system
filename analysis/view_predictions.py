# analysis/view_predictions.py
"""View and analyze prediction results with various sorting and filtering options."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


def view_all_predictions(csv_file: str = 'predictions.csv'):
    """
    Load and display all predictions with various views and statistics.

    Args:
        csv_file: Path to predictions CSV file
    """
    try:
        # Load the predictions
        df = pd.read_csv(csv_file)

        # Display settings for pandas
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)

        print("=" * 80)
        print("TRADING SYSTEM PREDICTIONS ANALYSIS")
        print("=" * 80)
        print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Symbols Analyzed: {len(df)}")

        # Basic statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Average Predicted Return: {df['predicted_return'].mean():.2%}")
        print(f"Median Predicted Return: {df['predicted_return'].median():.2%}")
        print(f"Standard Deviation: {df['predicted_return'].std():.2%}")
        print(f"Min Return: {df['predicted_return'].min():.2%} ({df.loc[df['predicted_return'].idxmin(), 'symbol']})")
        print(f"Max Return: {df['predicted_return'].max():.2%} ({df.loc[df['predicted_return'].idxmax(), 'symbol']})")
        print(f"Average Confidence: {df['confidence'].mean():.1%}")
        print(
            f"Predictions > 0: {len(df[df['predicted_return'] > 0])} ({len(df[df['predicted_return'] > 0]) / len(df):.1%})")
        print(
            f"Predictions < 0: {len(df[df['predicted_return'] < 0])} ({len(df[df['predicted_return'] < 0]) / len(df):.1%})")

        # Distribution breakdown
        print("\n" + "=" * 50)
        print("RETURN DISTRIBUTION")
        print("=" * 50)
        print(f"Strong Buy (>15%): {len(df[df['predicted_return'] > 0.15])} symbols")
        print(f"Buy (10-15%): {len(df[(df['predicted_return'] > 0.10) & (df['predicted_return'] <= 0.15)])} symbols")
        print(
            f"Moderate Buy (5-10%): {len(df[(df['predicted_return'] > 0.05) & (df['predicted_return'] <= 0.10)])} symbols")
        print(f"Mild Buy (2-5%): {len(df[(df['predicted_return'] > 0.02) & (df['predicted_return'] <= 0.05)])} symbols")
        print(
            f"Hold (-2% to 2%): {len(df[(df['predicted_return'] >= -0.02) & (df['predicted_return'] <= 0.02)])} symbols")
        print(
            f"Mild Sell (-5% to -2%): {len(df[(df['predicted_return'] < -0.02) & (df['predicted_return'] >= -0.05)])} symbols")
        print(f"Sell (< -5%): {len(df[df['predicted_return'] < -0.05])} symbols")

        # Confidence distribution
        print("\n" + "=" * 50)
        print("CONFIDENCE DISTRIBUTION")
        print("=" * 50)
        print(f"Very High (>80%): {len(df[df['confidence'] > 0.80])} symbols")
        print(f"High (70-80%): {len(df[(df['confidence'] > 0.70) & (df['confidence'] <= 0.80)])} symbols")
        print(f"Medium (60-70%): {len(df[(df['confidence'] > 0.60) & (df['confidence'] <= 0.70)])} symbols")
        print(f"Low (50-60%): {len(df[(df['confidence'] > 0.50) & (df['confidence'] <= 0.60)])} symbols")
        print(f"Very Low (<50%): {len(df[df['confidence'] <= 0.50])} symbols")

        # Top predictions by return
        print("\n" + "=" * 50)
        print("TOP 20 PREDICTIONS BY RETURN")
        print("=" * 50)
        top_returns = df.nlargest(20, 'predicted_return')[['symbol', 'predicted_return', 'confidence', 'current_price']]
        for idx, row in top_returns.iterrows():
            print(
                f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")

        # Top predictions by confidence
        print("\n" + "=" * 50)
        print("TOP 20 PREDICTIONS BY CONFIDENCE")
        print("=" * 50)
        top_confidence = df.nlargest(20, 'confidence')[['symbol', 'predicted_return', 'confidence', 'current_price']]
        for idx, row in top_confidence.iterrows():
            print(
                f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")

        # Best risk-adjusted opportunities (high confidence, good return)
        print("\n" + "=" * 50)
        print("BEST RISK-ADJUSTED OPPORTUNITIES")
        print("(Return > 5% and Confidence > 60%)")
        print("=" * 50)
        risk_adjusted = df[(df['predicted_return'] > 0.05) & (df['confidence'] > 0.60)].sort_values(
            by=['confidence', 'predicted_return'], ascending=[False, False]
        )
        if len(risk_adjusted) > 0:
            for idx, row in risk_adjusted.head(20).iterrows():
                risk_score = row['predicted_return'] * row['confidence']  # Simple risk-adjusted score
                print(
                    f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Score: {risk_score:6.3f} | Price: ${row['current_price']:8.2f}")
        else:
            print("No symbols meet these criteria")

        # Potential shorts
        print("\n" + "=" * 50)
        print("POTENTIAL SHORT OPPORTUNITIES")
        print("(Negative predicted returns)")
        print("=" * 50)
        shorts = df[df['predicted_return'] < 0].sort_values('predicted_return')
        if len(shorts) > 0:
            for idx, row in shorts.head(20).iterrows():
                print(
                    f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")
        else:
            print("No short opportunities identified")

        # Conservative plays (high confidence, moderate return)
        print("\n" + "=" * 50)
        print("CONSERVATIVE PLAYS")
        print("(Return 2-5% and Confidence > 70%)")
        print("=" * 50)
        conservative = df[
            (df['predicted_return'] > 0.02) & (df['predicted_return'] <= 0.05) & (df['confidence'] > 0.70)].sort_values(
            'confidence', ascending=False
        )
        if len(conservative) > 0:
            for idx, row in conservative.head(15).iterrows():
                print(
                    f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")
        else:
            print("No conservative plays meet these criteria")

        # Contrarian plays (low confidence but high return)
        print("\n" + "=" * 50)
        print("CONTRARIAN/HIGH-RISK PLAYS")
        print("(Return > 15% but Confidence < 50%)")
        print("=" * 50)
        contrarian = df[(df['predicted_return'] > 0.15) & (df['confidence'] < 0.50)].sort_values(
            'predicted_return', ascending=False
        )
        if len(contrarian) > 0:
            for idx, row in contrarian.head(10).iterrows():
                print(
                    f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")
        else:
            print("No contrarian plays identified")

        # Portfolio suggestions
        print("\n" + "=" * 80)
        print("PORTFOLIO CONSTRUCTION SUGGESTIONS")
        print("=" * 80)

        # Balanced portfolio
        print("\nBALANCED PORTFOLIO (Mixed risk/return):")
        # High confidence moderate return
        high_conf_mod = df[
            (df['confidence'] > 0.70) & (df['predicted_return'] > 0.03) & (df['predicted_return'] < 0.10)]
        # Medium confidence high return
        med_conf_high = df[(df['confidence'] > 0.50) & (df['confidence'] <= 0.70) & (df['predicted_return'] > 0.10)]

        balanced = pd.concat([
            high_conf_mod.head(5),
            med_conf_high.head(5)
        ]).sort_values('predicted_return', ascending=False)

        for idx, row in balanced.iterrows():
            print(
                f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")

        # Aggressive portfolio
        print("\nAGGRESSIVE PORTFOLIO (High return focus):")
        aggressive = df[df['predicted_return'] > 0.10].sort_values('predicted_return', ascending=False).head(10)
        for idx, row in aggressive.iterrows():
            print(
                f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")

        # Conservative portfolio
        print("\nCONSERVATIVE PORTFOLIO (High confidence focus):")
        conservative_port = df[(df['confidence'] > 0.75) & (df['predicted_return'] > 0.02)].sort_values(
            'confidence', ascending=False
        ).head(10)
        for idx, row in conservative_port.iterrows():
            print(
                f"{row['symbol']:6s} | Return: {row['predicted_return']:7.2%} | Confidence: {row['confidence']:5.1%} | Price: ${row['current_price']:8.2f}")

        # Save filtered views to separate CSVs
        print("\n" + "=" * 50)
        print("SAVING FILTERED VIEWS")
        print("=" * 50)

        # Save different views
        top_returns.to_csv('analysis/top_returns.csv', index=False)
        print("✓ Saved top_returns.csv")

        top_confidence.to_csv('analysis/top_confidence.csv', index=False)
        print("✓ Saved top_confidence.csv")

        if len(risk_adjusted) > 0:
            risk_adjusted.to_csv('analysis/risk_adjusted_opportunities.csv', index=False)
            print("✓ Saved risk_adjusted_opportunities.csv")

        if len(shorts) > 0:
            shorts.to_csv('analysis/short_opportunities.csv', index=False)
            print("✓ Saved short_opportunities.csv")

        # Return the dataframe for further processing
        return df

    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        print("Please run 'python main.py --predict' first to generate predictions.")
        return None
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None


def filter_predictions(df: pd.DataFrame,
                       min_return: float = None,
                       max_return: float = None,
                       min_confidence: float = None,
                       max_confidence: float = None,
                       symbols: list = None) -> pd.DataFrame:
    """
    Filter predictions based on criteria.

    Args:
        df: Predictions dataframe
        min_return: Minimum predicted return
        max_return: Maximum predicted return
        min_confidence: Minimum confidence
        max_confidence: Maximum confidence
        symbols: List of specific symbols to include

    Returns:
        Filtered dataframe
    """
    filtered = df.copy()

    if min_return is not None:
        filtered = filtered[filtered['predicted_return'] >= min_return]
    if max_return is not None:
        filtered = filtered[filtered['predicted_return'] <= max_return]
    if min_confidence is not None:
        filtered = filtered[filtered['confidence'] >= min_confidence]
    if max_confidence is not None:
        filtered = filtered[filtered['confidence'] <= max_confidence]
    if symbols is not None:
        filtered = filtered[filtered['symbol'].isin(symbols)]

    return filtered


if __name__ == "__main__":
    # Run the analysis
    df = view_all_predictions()

    # Example of custom filtering
    if df is not None:
        print("\n" + "=" * 50)
        print("CUSTOM FILTER EXAMPLE")
        print("(Return > 10% and Confidence > 65%)")
        print("=" * 50)

        custom = filter_predictions(df, min_return=0.10, min_confidence=0.65)
        print(f"Found {len(custom)} symbols matching criteria")
        if len(custom) > 0:
            print(custom[['symbol', 'predicted_return', 'confidence', 'current_price']].head(10))
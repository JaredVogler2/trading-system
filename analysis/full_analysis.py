# analysis/full_analysis.py
"""Generate comprehensive analysis reports with sector breakdowns and detailed statistics."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_sector_mapping() -> Dict[str, str]:
    """Create sector mapping for all symbols."""
    sector_map = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
        'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology', 'AMD': 'Technology',
        'INTC': 'Technology', 'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology',
        'NFLX': 'Technology', 'AVGO': 'Technology', 'CSCO': 'Technology', 'QCOM': 'Technology',
        'TXN': 'Technology', 'IBM': 'Technology', 'NOW': 'Technology', 'UBER': 'Technology',

        # Financial
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
        'MS': 'Financial', 'C': 'Financial', 'USB': 'Financial', 'PNC': 'Financial',
        'AXP': 'Financial', 'BLK': 'Financial', 'SCHW': 'Financial', 'COF': 'Financial',
        'SPGI': 'Financial', 'CME': 'Financial', 'ICE': 'Financial', 'V': 'Financial',
        'MA': 'Financial', 'PYPL': 'Financial', 'SQ': 'Financial', 'COIN': 'Financial',

        # Healthcare
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
        'TMO': 'Healthcare', 'ABT': 'Healthcare', 'CVS': 'Healthcare', 'MRK': 'Healthcare',
        'DHR': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare',
        'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'ZTS': 'Healthcare', 'BIIB': 'Healthcare',
        'ILMN': 'Healthcare', 'IDXX': 'Healthcare', 'ALGN': 'Healthcare', 'DXCM': 'Healthcare',

        # Consumer Discretionary
        'HD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
        'DIS': 'Consumer Discretionary', 'CMCSA': 'Consumer Discretionary', 'CHTR': 'Consumer Discretionary',
        'ROKU': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary',
        'RIVN': 'Consumer Discretionary', 'LCID': 'Consumer Discretionary', 'CCL': 'Consumer Discretionary',
        'RCL': 'Consumer Discretionary', 'WYNN': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary',
        'DKNG': 'Consumer Discretionary', 'PENN': 'Consumer Discretionary',

        # Consumer Staples
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',

        # Industrial
        'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
        'HON': 'Industrial', 'UPS': 'Industrial', 'RTX': 'Industrial', 'LMT': 'Industrial',
        'NOC': 'Industrial', 'DE': 'Industrial', 'EMR': 'Industrial', 'ETN': 'Industrial',
        'ITW': 'Industrial', 'PH': 'Industrial', 'GD': 'Industrial', 'FDX': 'Industrial',
        'NSC': 'Industrial', 'UNP': 'Industrial', 'CSX': 'Industrial', 'DAL': 'Industrial',

        # Energy
        'CVX': 'Energy', 'XOM': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',

        # Utilities
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        'AEP': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities', 'ED': 'Utilities',
        'WEC': 'Utilities', 'ES': 'Utilities',

        # Real Estate
        'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
        'PSA': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate', 'AVB': 'Real Estate',
        'EQR': 'Real Estate', 'SPG': 'Real Estate',

        # Communication Services
        'VZ': 'Communication', 'T': 'Communication', 'TMUS': 'Communication',
        'SNAP': 'Communication', 'PINS': 'Communication', 'TWTR': 'Communication',
        'ZM': 'Communication', 'DOCU': 'Communication',

        # Other
        'AAL': 'Other', 'UAL': 'Other', 'OKTA': 'Other'
    }

    return sector_map


def generate_full_analysis(csv_file: str = 'predictions.csv'):
    """Generate comprehensive analysis report with visualizations."""

    try:
        # Load predictions
        df = pd.read_csv(csv_file)

        # Add sector information
        sector_map = create_sector_mapping()
        df['sector'] = df['symbol'].map(sector_map).fillna('Other')

        # Create output directory
        output_dir = Path('analysis/reports')
        output_dir.mkdir(exist_ok=True, parents=True)

        # Generate text report
        generate_text_report(df, output_dir)

        # Generate Excel report
        generate_excel_report(df, output_dir)

        # Generate visualizations
        generate_visualizations(df, output_dir)

        # Generate sector analysis
        generate_sector_analysis(df, output_dir)

        # Generate portfolio recommendations
        generate_portfolio_recommendations(df, output_dir)

        print(f"\n✓ Full analysis complete! Check the 'analysis/reports' directory for:")
        print("  - full_analysis_report.txt")
        print("  - full_analysis.xlsx")
        print("  - Various PNG charts")
        print("  - sector_analysis.xlsx")
        print("  - portfolio_recommendations.txt")

        return df

    except Exception as e:
        print(f"Error generating analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_text_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive text report."""

    report_path = output_dir / 'full_analysis_report.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE TRADING SYSTEM ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Symbols Analyzed: {len(df)}\n")
        f.write(f"Prediction Horizon: 21 days\n\n")

        # Executive Summary
        f.write("=" * 50 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 50 + "\n")

        bullish = len(df[df['predicted_return'] > 0.02])
        bearish = len(df[df['predicted_return'] < -0.02])
        neutral = len(df) - bullish - bearish

        f.write(
            f"Market Outlook: {'BULLISH' if bullish > bearish else 'BEARISH' if bearish > bullish else 'NEUTRAL'}\n")
        f.write(f"Bullish Signals: {bullish} ({bullish / len(df):.1%})\n")
        f.write(f"Bearish Signals: {bearish} ({bearish / len(df):.1%})\n")
        f.write(f"Neutral Signals: {neutral} ({neutral / len(df):.1%})\n\n")

        # Overall Statistics
        f.write("=" * 50 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Predicted Return: {df['predicted_return'].mean():.2%}\n")
        f.write(f"Median Predicted Return: {df['predicted_return'].median():.2%}\n")
        f.write(f"Standard Deviation: {df['predicted_return'].std():.2%}\n")
        f.write(f"Skewness: {df['predicted_return'].skew():.2f}\n")
        f.write(f"Kurtosis: {df['predicted_return'].kurtosis():.2f}\n")
        f.write(
            f"Min Return: {df['predicted_return'].min():.2%} ({df.loc[df['predicted_return'].idxmin(), 'symbol']})\n")
        f.write(
            f"Max Return: {df['predicted_return'].max():.2%} ({df.loc[df['predicted_return'].idxmax(), 'symbol']})\n")
        f.write(f"Average Confidence: {df['confidence'].mean():.2%}\n")
        f.write(f"Confidence Range: {df['confidence'].min():.2%} - {df['confidence'].max():.2%}\n\n")

        # Percentile Analysis
        f.write("=" * 50 + "\n")
        f.write("RETURN PERCENTILES\n")
        f.write("=" * 50 + "\n")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            f.write(f"{p}th percentile: {df['predicted_return'].quantile(p / 100):.2%}\n")
        f.write("\n")

        # Sector Analysis
        f.write("=" * 50 + "\n")
        f.write("SECTOR ANALYSIS\n")
        f.write("=" * 50 + "\n")

        sector_stats = df.groupby('sector').agg({
            'predicted_return': ['mean', 'std', 'min', 'max', 'count'],
            'confidence': 'mean'
        }).round(4)

        # Sort by mean return
        sector_stats = sector_stats.sort_values(('predicted_return', 'mean'), ascending=False)

        f.write(
            f"{'Sector':<20} {'Avg Return':>12} {'Std Dev':>10} {'Min':>10} {'Max':>10} {'Count':>8} {'Avg Conf':>10}\n")
        f.write("-" * 82 + "\n")

        for sector in sector_stats.index:
            stats = sector_stats.loc[sector]
            # Fixed: Use .0f for count to format as integer
            f.write(f"{sector:<20} "
                    f"{stats[('predicted_return', 'mean')]:>11.2%} "
                    f"{stats[('predicted_return', 'std')]:>9.2%} "
                    f"{stats[('predicted_return', 'min')]:>9.2%} "
                    f"{stats[('predicted_return', 'max')]:>9.2%} "
                    f"{int(stats[('predicted_return', 'count')]):>8d} "
                    f"{stats[('confidence', 'mean')]:>9.1%}\n")

        # Top Opportunities by Category
        categories = [
            ("TOP 30 OVERALL OPPORTUNITIES", df.nlargest(30, 'predicted_return')),
            ("TOP 30 HIGH CONFIDENCE PLAYS (>70%)", df[df['confidence'] > 0.70].nlargest(30, 'predicted_return')),
            ("TOP 20 RISK-ADJUSTED PLAYS",
             df.assign(score=lambda x: x['predicted_return'] * x['confidence']).nlargest(20, 'score')),
            ("TOP 20 CONTRARIAN PLAYS (High Return, Low Confidence)",
             df[(df['predicted_return'] > 0.10) & (df['confidence'] < 0.60)].nlargest(20, 'predicted_return')),
            ("ALL SHORT OPPORTUNITIES", df[df['predicted_return'] < 0].sort_values('predicted_return'))
        ]

        for title, subset in categories:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"{title}\n")
            f.write(f"{'=' * 50}\n")

            if len(subset) > 0:
                f.write(
                    f"{'Rank':<6} {'Symbol':<8} {'Sector':<20} {'Return':>10} {'Conf':>8} {'Price':>10} {'Target':>10}\n")
                f.write("-" * 84 + "\n")

                for i, (idx, row) in enumerate(subset.iterrows(), 1):
                    target_price = row['current_price'] * (1 + row['predicted_return']) if pd.notna(
                        row['current_price']) else 0
                    price_str = f"${row['current_price']:>9.2f}" if pd.notna(row['current_price']) else "N/A"
                    target_str = f"${target_price:>9.2f}" if target_price > 0 else "N/A"

                    f.write(f"{i:<6d} {row['symbol']:<8} {row['sector']:<20} "
                            f"{row['predicted_return']:>9.2%} {row['confidence']:>7.1%} "
                            f"{price_str:>10} {target_str:>10}\n")
            else:
                f.write("No opportunities in this category.\n")

        # Risk Analysis
        f.write(f"\n{'=' * 50}\n")
        f.write("RISK ANALYSIS\n")
        f.write(f"{'=' * 50}\n")

        # Calculate portfolio metrics if all positions were taken
        avg_return = df['predicted_return'].mean()
        portfolio_std = df['predicted_return'].std() / np.sqrt(len(df))  # Assuming some diversification
        sharpe_ratio = (avg_return - 0.02 / 12) / (df['predicted_return'].std()) * np.sqrt(12)  # Annualized

        f.write(f"Portfolio Expected Return (equal weight): {avg_return:.2%}\n")
        f.write(f"Portfolio Standard Deviation: {portfolio_std:.2%}\n")
        f.write(f"Estimated Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Best 20 Positions Return: {df.nlargest(20, 'predicted_return')['predicted_return'].mean():.2%}\n")
        f.write(f"Worst 20 Positions Return: {df.nsmallest(20, 'predicted_return')['predicted_return'].mean():.2%}\n")

    print(f"✓ Text report saved to {report_path}")


def generate_excel_report(df: pd.DataFrame, output_dir: Path):
    """Generate comprehensive Excel report with multiple sheets."""

    excel_path = output_dir / 'full_analysis.xlsx'

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Sheet 1: All predictions sorted by return
        df_sorted = df.sort_values('predicted_return', ascending=False)
        df_sorted.to_excel(writer, sheet_name='All Predictions', index=False)

        # Sheet 2: Sector summary
        sector_summary = df.groupby('sector').agg({
            'predicted_return': ['mean', 'std', 'min', 'max', 'count'],
            'confidence': ['mean', 'min', 'max'],
            'current_price': 'sum'  # Total market cap proxy
        }).round(4)
        sector_summary.to_excel(writer, sheet_name='Sector Analysis')

        # Sheet 3: Top opportunities
        top_opps = df.nlargest(50, 'predicted_return')
        top_opps.to_excel(writer, sheet_name='Top 50 Opportunities', index=False)

        # Sheet 4: High confidence plays
        high_conf = df[df['confidence'] > 0.70].sort_values('predicted_return', ascending=False)
        high_conf.to_excel(writer, sheet_name='High Confidence', index=False)

        # Sheet 5: Risk-adjusted plays
        df['risk_adjusted_score'] = df['predicted_return'] * df['confidence']
        risk_adjusted = df.nlargest(50, 'risk_adjusted_score')
        risk_adjusted.to_excel(writer, sheet_name='Risk Adjusted', index=False)

        # Sheet 6: Short opportunities
        shorts = df[df['predicted_return'] < 0].sort_values('predicted_return')
        if len(shorts) > 0:
            shorts.to_excel(writer, sheet_name='Short Opportunities', index=False)

        # Sheet 7: Statistics
        stats_data = {
            'Metric': [
                'Total Symbols', 'Average Return', 'Median Return', 'Std Deviation',
                'Min Return', 'Max Return', 'Average Confidence', 'Bullish Count',
                'Bearish Count', 'Neutral Count', 'High Confidence Count (>70%)',
                'Strong Buy Count (>10%)', 'Strong Sell Count (<-5%)'
            ],
            'Value': [
                len(df),
                f"{df['predicted_return'].mean():.2%}",
                f"{df['predicted_return'].median():.2%}",
                f"{df['predicted_return'].std():.2%}",
                f"{df['predicted_return'].min():.2%}",
                f"{df['predicted_return'].max():.2%}",
                f"{df['confidence'].mean():.2%}",
                len(df[df['predicted_return'] > 0.02]),
                len(df[df['predicted_return'] < -0.02]),
                len(df[(df['predicted_return'] >= -0.02) & (df['predicted_return'] <= 0.02)]),
                len(df[df['confidence'] > 0.70]),
                len(df[df['predicted_return'] > 0.10]),
                len(df[df['predicted_return'] < -0.05])
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Summary Statistics', index=False)

        # Get workbook and add formats
        workbook = writer.book
        percent_format = workbook.add_format({'num_format': '0.00%'})
        dollar_format = workbook.add_format({'num_format': '$#,##0.00'})

        # Format columns in each sheet
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('D:E', 12, percent_format)  # Return and confidence columns
            worksheet.set_column('F:F', 12, dollar_format)  # Price column

    print(f"✓ Excel report saved to {excel_path}")


def generate_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate visualization charts."""

    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # 1. Return distribution histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df['predicted_return'] * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Return (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Returns')
    plt.grid(True, alpha=0.3)

    # 2. Confidence distribution
    plt.subplot(1, 2, 2)
    plt.hist(df['confidence'] * 100, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Scatter plot of return vs confidence
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['confidence'] * 100, df['predicted_return'] * 100,
                          c=df['predicted_return'] * 100, cmap='RdYlGn',
                          alpha=0.6, s=50)
    plt.xlabel('Confidence (%)')
    plt.ylabel('Predicted Return (%)')
    plt.title('Predicted Return vs Confidence')
    plt.colorbar(scatter, label='Return (%)')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)

    # Add annotations for top opportunities
    top_5 = df.nlargest(5, 'predicted_return')
    for _, row in top_5.iterrows():
        plt.annotate(row['symbol'],
                     (row['confidence'] * 100, row['predicted_return'] * 100),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.savefig(output_dir / 'return_vs_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Sector performance bar chart
    plt.figure(figsize=(12, 8))
    sector_avg = df.groupby('sector')['predicted_return'].mean().sort_values(ascending=True)
    colors = ['red' if x < 0 else 'green' for x in sector_avg.values]

    plt.barh(sector_avg.index, sector_avg.values * 100, color=colors, alpha=0.7)
    plt.xlabel('Average Predicted Return (%)')
    plt.title('Average Predicted Return by Sector')
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (sector, value) in enumerate(sector_avg.items()):
        plt.text(value * 100 + 0.1, i, f'{value:.1%}', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / 'sector_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Top 30 predictions
    plt.figure(figsize=(14, 10))
    top_30 = df.nlargest(30, 'predicted_return')

    plt.barh(range(len(top_30)), top_30['predicted_return'] * 100,
             color=plt.cm.RdYlGn(top_30['confidence']))
    plt.yticks(range(len(top_30)), top_30['symbol'])
    plt.xlabel('Predicted Return (%)')
    plt.title('Top 30 Predicted Returns')
    plt.grid(True, alpha=0.3, axis='x')

    # Add confidence as text
    for i, (_, row) in enumerate(top_30.iterrows()):
        plt.text(row['predicted_return'] * 100 + 0.5, i,
                 f"{row['confidence']:.0%}", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'top_30_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Visualizations saved")


def generate_sector_analysis(df: pd.DataFrame, output_dir: Path):
    """Generate detailed sector analysis."""

    sector_path = output_dir / 'sector_analysis.xlsx'

    with pd.ExcelWriter(sector_path, engine='xlsxwriter') as writer:
        # Detailed sector statistics
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector].sort_values('predicted_return', ascending=False)

            if len(sector_df) > 0:
                # Create summary for this sector
                summary = pd.DataFrame({
                    'Metric': ['Count', 'Avg Return', 'Median Return', 'Std Dev',
                               'Min Return', 'Max Return', 'Avg Confidence',
                               'Bullish %', 'Bearish %'],
                    'Value': [
                        len(sector_df),
                        f"{sector_df['predicted_return'].mean():.2%}",
                        f"{sector_df['predicted_return'].median():.2%}",
                        f"{sector_df['predicted_return'].std():.2%}",
                        f"{sector_df['predicted_return'].min():.2%}",
                        f"{sector_df['predicted_return'].max():.2%}",
                        f"{sector_df['confidence'].mean():.2%}",
                        f"{len(sector_df[sector_df['predicted_return'] > 0.02]) / len(sector_df):.1%}",
                        f"{len(sector_df[sector_df['predicted_return'] < -0.02]) / len(sector_df):.1%}"
                    ]
                })

                # Write sector sheet
                sheet_name = sector.replace('/', '_')[:31]  # Excel sheet name limit

                # Write summary at top
                summary.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)

                # Write detailed data below
                sector_df.to_excel(writer, sheet_name=sheet_name, startrow=len(summary) + 2, index=False)

    print(f"✓ Sector analysis saved to {sector_path}")


def generate_portfolio_recommendations(df: pd.DataFrame, output_dir: Path):
    """Generate portfolio construction recommendations."""

    rec_path = output_dir / 'portfolio_recommendations.txt'

    with open(rec_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PORTFOLIO CONSTRUCTION RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Conservative Portfolio
        f.write("=" * 50 + "\n")
        f.write("CONSERVATIVE PORTFOLIO\n")
        f.write("Target: Steady returns with high confidence\n")
        f.write("=" * 50 + "\n\n")

        conservative = df[(df['confidence'] > 0.75) & (df['predicted_return'] > 0.02) & (df['predicted_return'] < 0.10)]
        conservative = conservative.sort_values('confidence', ascending=False).head(15)

        f.write(f"{'Symbol':<8} {'Sector':<20} {'Return':>10} {'Conf':>8} {'Allocation':>12}\n")
        f.write("-" * 60 + "\n")

        allocation = 100 / len(conservative) if len(conservative) > 0 else 0
        for _, row in conservative.iterrows():
            f.write(f"{row['symbol']:<8} {row['sector']:<20} {row['predicted_return']:>9.2%} "
                    f"{row['confidence']:>7.1%} {allocation:>11.1f}%\n")

        if len(conservative) > 0:
            f.write(f"\nExpected Portfolio Return: {conservative['predicted_return'].mean():.2%}\n")
            f.write(f"Average Confidence: {conservative['confidence'].mean():.1%}\n")

        # Balanced Portfolio
        f.write("\n" + "=" * 50 + "\n")
        f.write("BALANCED PORTFOLIO\n")
        f.write("Target: Mix of growth and stability\n")
        f.write("=" * 50 + "\n\n")

        # Get stocks from different return/confidence buckets
        high_conf_moderate = df[
            (df['confidence'] > 0.70) & (df['predicted_return'] > 0.05) & (df['predicted_return'] < 0.15)].head(8)
        moderate_conf_high = df[
            (df['confidence'] > 0.50) & (df['confidence'] <= 0.70) & (df['predicted_return'] > 0.10)].head(7)

        balanced = pd.concat([high_conf_moderate, moderate_conf_high])
        balanced = balanced.sort_values('predicted_return', ascending=False)

        f.write(f"{'Symbol':<8} {'Sector':<20} {'Return':>10} {'Conf':>8} {'Allocation':>12}\n")
        f.write("-" * 60 + "\n")

        if len(balanced) > 0:
            # Higher allocation to high confidence stocks
            for _, row in balanced.iterrows():
                if row['confidence'] > 0.70:
                    allocation = 7.0
                else:
                    allocation = 6.0
                f.write(f"{row['symbol']:<8} {row['sector']:<20} {row['predicted_return']:>9.2%} "
                        f"{row['confidence']:>7.1%} {allocation:>11.1f}%\n")

            f.write(f"\nExpected Portfolio Return: {balanced['predicted_return'].mean():.2%}\n")
            f.write(f"Average Confidence: {balanced['confidence'].mean():.1%}\n")

        # Aggressive Portfolio
        f.write("\n" + "=" * 50 + "\n")
        f.write("AGGRESSIVE PORTFOLIO\n")
        f.write("Target: Maximum returns, higher risk tolerance\n")
        f.write("=" * 50 + "\n\n")

        aggressive = df.nlargest(20, 'predicted_return')

        f.write(f"{'Symbol':<8} {'Sector':<20} {'Return':>10} {'Conf':>8} {'Allocation':>12}\n")
        f.write("-" * 60 + "\n")

        # Equal weight for aggressive
        allocation = 100 / len(aggressive) if len(aggressive) > 0 else 0
        for _, row in aggressive.iterrows():
            f.write(f"{row['symbol']:<8} {row['sector']:<20} {row['predicted_return']:>9.2%} "
                    f"{row['confidence']:>7.1%} {allocation:>11.1f}%\n")

        if len(aggressive) > 0:
            f.write(f"\nExpected Portfolio Return: {aggressive['predicted_return'].mean():.2%}\n")
            f.write(f"Average Confidence: {aggressive['confidence'].mean():.1%}\n")

        # Sector-Diversified Portfolio
        f.write("\n" + "=" * 50 + "\n")
        f.write("SECTOR-DIVERSIFIED PORTFOLIO\n")
        f.write("Target: Balanced sector exposure\n")
        f.write("=" * 50 + "\n\n")

        # Get top 2 from each sector
        sector_picks = []
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            top_sector = sector_df[(sector_df['predicted_return'] > 0.02) & (sector_df['confidence'] > 0.50)].nlargest(
                2, 'predicted_return')
            sector_picks.append(top_sector)

        diversified = pd.concat(sector_picks).sort_values('predicted_return', ascending=False)

        f.write(f"{'Symbol':<8} {'Sector':<20} {'Return':>10} {'Conf':>8} {'Allocation':>12}\n")
        f.write("-" * 60 + "\n")

        if len(diversified) > 0:
            allocation = 100 / len(diversified)
            for _, row in diversified.iterrows():
                f.write(f"{row['symbol']:<8} {row['sector']:<20} {row['predicted_return']:>9.2%} "
                        f"{row['confidence']:>7.1%} {allocation:>11.1f}%\n")

            f.write(f"\nExpected Portfolio Return: {diversified['predicted_return'].mean():.2%}\n")
            f.write(f"Average Confidence: {diversified['confidence'].mean():.1%}\n")
            f.write(f"Sector Count: {diversified['sector'].nunique()}\n")

        # Risk Management Guidelines
        f.write("\n" + "=" * 50 + "\n")
        f.write("RISK MANAGEMENT GUIDELINES\n")
        f.write("=" * 50 + "\n\n")

        f.write("Position Sizing:\n")
        f.write("- Conservative: 2-3% per position\n")
        f.write("- Balanced: 3-5% per position\n")
        f.write("- Aggressive: 5-7% per position\n\n")

        f.write("Stop Loss Recommendations:\n")
        f.write("- High Confidence (>75%): 5-7% stop loss\n")
        f.write("- Medium Confidence (60-75%): 3-5% stop loss\n")
        f.write("- Low Confidence (<60%): 2-3% stop loss\n\n")

        f.write("Take Profit Targets:\n")
        f.write("- Conservative: 50-75% of predicted return\n")
        f.write("- Balanced: 75-100% of predicted return\n")
        f.write("- Aggressive: 100-150% of predicted return\n\n")

        f.write("Portfolio Monitoring:\n")
        f.write("- Daily: Check positions hitting stop loss or take profit\n")
        f.write("- Weekly: Review confidence changes in predictions\n")
        f.write("- Bi-weekly: Rebalance if any position exceeds 10% of portfolio\n")
        f.write("- Monthly: Full portfolio review and reoptimization\n")

    print(f"✓ Portfolio recommendations saved to {rec_path}")


if __name__ == "__main__":
    # Generate full analysis
    generate_full_analysis()
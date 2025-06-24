# create visualize_timeframes.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv('multi_timeframe_predictions.csv')

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Top 20 stocks progression
top20 = df.head(20)
x = range(len(top20))
width = 0.25

ax1 = axes[0, 0]
ax1.bar([i - width for i in x], top20['week1_predicted_return'] * 100, width, label='Week 1', alpha=0.8)
ax1.bar(x, top20['week2_predicted_return'] * 100, width, label='Week 2', alpha=0.8)
ax1.bar([i + width for i in x], top20['week3_predicted_return'] * 100, width, label='Week 3', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(top20['symbol'], rotation=45)
ax1.set_ylabel('Predicted Return (%)')
ax1.set_title('Top 20 Stocks - Return Progression')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Confidence decay
ax2 = axes[0, 1]
ax2.plot([1, 2, 3], [
    df['week1_confidence'].mean(),
    df['week2_confidence'].mean(),
    df['week3_confidence'].mean()
], marker='o', linewidth=2, markersize=10)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['Week 1', 'Week 2', 'Week 3'])
ax2.set_ylabel('Average Confidence')
ax2.set_title('Confidence Decay Over Time')
ax2.grid(True, alpha=0.3)

# 3. Return distribution by timeframe
ax3 = axes[1, 0]
data_to_plot = [
    df['week1_predicted_return'] * 100,
    df['week2_predicted_return'] * 100,
    df['week3_predicted_return'] * 100
]
ax3.boxplot(data_to_plot, labels=['Week 1', 'Week 2', 'Week 3'])
ax3.set_ylabel('Predicted Return (%)')
ax3.set_title('Return Distribution by Timeframe')
ax3.grid(True, alpha=0.3)

# 4. High confidence short-term plays
high_conf_short = df[df['week1_confidence'] > 0.8].nlargest(15, 'week1_predicted_return')
ax4 = axes[1, 1]
ax4.barh(high_conf_short['symbol'], high_conf_short['week1_predicted_return'] * 100)
ax4.set_xlabel('Week 1 Predicted Return (%)')
ax4.set_title('Best High-Confidence Week 1 Plays (>80% confidence)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_timeframe_analysis.png', dpi=300)
plt.show()

# Print trading recommendations
print("\n=== TRADING RECOMMENDATIONS ===")
print("\nBest Week 1 Plays (High Confidence Short-Term):")
week1_plays = df[df['week1_confidence'] > 0.8].nlargest(10, 'week1_predicted_return')
print(week1_plays[['symbol', 'week1_predicted_return', 'week1_confidence', 'current_price']])

print("\nBest Swing Trades (2-Week Horizon):")
swing_trades = df[df['week2_confidence'] > 0.75].nlargest(10, 'week2_predicted_return')
print(swing_trades[['symbol', 'week2_predicted_return', 'week2_confidence', 'current_price']])

print("\nBest Position Trades (3-Week Horizon):")
position_trades = df[df['week3_confidence'] > 0.7].nlargest(10, 'week3_predicted_return')
print(position_trades[['symbol', 'week3_predicted_return', 'week3_confidence', 'current_price']])
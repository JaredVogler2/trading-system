# features/base_features.py
"""
Base feature engineering module implementing all 27 base features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

from config.settings import BASE_FEATURES


class BaseFeatureCalculator:
    """Calculate all base features for stock data."""

    def __init__(self):
        """Initialize the feature calculator."""
        self.feature_names = BASE_FEATURES

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all base features for a stock.

        Args:
            df: DataFrame with OHLCV data and returns

        Returns:
            DataFrame with all features
        """
        # Create a copy to avoid modifying original
        features = df.copy()

        # Price return features
        features = self._calculate_price_returns(features)

        # Price position features
        features = self._calculate_price_position(features)

        # Volatility features
        features = self._calculate_volatility(features)

        # Price acceleration
        features = self._calculate_price_acceleration(features)

        # Moving average features
        features = self._calculate_moving_averages(features)

        # Technical indicators
        features = self._calculate_technical_indicators(features)

        # Volume features
        features = self._calculate_volume_features(features)

        # Support/Resistance features
        features = self._calculate_support_resistance(features)

        # Trend features
        features = self._calculate_trend_features(features)

        # Market regime features
        features = self._calculate_regime_features(features)

        # Trading signal features
        features = self._calculate_signal_features(features)

        # Seasonal features
        features = self._calculate_seasonal_features(features)

        # Risk metrics
        features = self._calculate_risk_metrics(features)

        # Cross-asset features (requires market data)
        features = self._calculate_cross_asset_features(features)

        # Select only the features we need
        return features[self.feature_names].fillna(0)

    def _calculate_price_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price return features."""
        # Returns at multiple timeframes
        df['returns_1d'] = df['close'].pct_change(1)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_20d'] = df['close'].pct_change(20)

        return df

    def _calculate_price_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price position within ranges."""
        # Price position within 10-day range
        rolling_max_10 = df['high'].rolling(window=10).max()
        rolling_min_10 = df['low'].rolling(window=10).min()
        df['price_position_10d'] = (df['close'] - rolling_min_10) / (rolling_max_10 - rolling_min_10 + 1e-10)

        # Price position within 20-day range
        rolling_max_20 = df['high'].rolling(window=20).max()
        rolling_min_20 = df['low'].rolling(window=20).min()
        df['price_position_20d'] = (df['close'] - rolling_min_20) / (rolling_max_20 - rolling_min_20 + 1e-10)

        return df

    def _calculate_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility measures."""
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()

        # 10-day volatility
        df['volatility_10d'] = df['returns'].rolling(window=10).std() * np.sqrt(252)

        # 20-day volatility
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

        return df

    def _calculate_price_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price acceleration (second derivative)."""
        # Price acceleration as change in returns
        df['price_acceleration'] = df['returns_1d'].diff()

        return df

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average features."""
        # Simple Moving Averages
        sma_5 = df['close'].rolling(window=5).mean()
        sma_20 = df['close'].rolling(window=20).mean()
        sma_12 = df['close'].rolling(window=12).mean()
        sma_26 = df['close'].rolling(window=26).mean()

        # SMA ratios
        df['sma_ratio_5_20'] = sma_5 / sma_20
        df['sma_ratio_12_26'] = sma_12 / sma_26

        # Exponential Moving Averages
        ema_5 = df['close'].ewm(span=5, adjust=False).mean()
        ema_20 = df['close'].ewm(span=20, adjust=False).mean()
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()

        # EMA ratios
        df['ema_ratio_5_20'] = ema_5 / ema_20
        df['ema_ratio_12_26'] = ema_12 / ema_26

        # MA alignment (trend strength indicator)
        # Check if short, medium, and long MAs are aligned
        ma_short = sma_5
        ma_medium = sma_12
        ma_long = sma_26

        df['ma_alignment'] = ((ma_short > ma_medium) & (ma_medium > ma_long)).astype(float) - \
                             ((ma_short < ma_medium) & (ma_medium < ma_long)).astype(float)

        # MA crossover detection
        df['ma_crossover'] = 0
        df.loc[(sma_5.shift(1) <= sma_20.shift(1)) & (sma_5 > sma_20), 'ma_crossover'] = 1  # Golden cross
        df.loc[(sma_5.shift(1) >= sma_20.shift(1)) & (sma_5 < sma_20), 'ma_crossover'] = -1  # Death cross

        return df

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # RSI
        df['rsi_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # RSI Divergence (simplified - price making new highs but RSI isn't)
        price_high_20 = df['high'].rolling(window=20).max()
        rsi_high_20 = df['rsi_14'].rolling(window=20).max()

        df['rsi_divergence'] = 0
        new_price_high = df['high'] >= price_high_20.shift(1)
        no_new_rsi_high = df['rsi_14'] < rsi_high_20.shift(1)
        df.loc[new_price_high & no_new_rsi_high & (df['rsi_14'] > 70), 'rsi_divergence'] = -1  # Bearish divergence

        new_price_low = df['low'] <= df['low'].rolling(window=20).min().shift(1)
        no_new_rsi_low = df['rsi_14'] > df['rsi_14'].rolling(window=20).min().shift(1)
        df.loc[new_price_low & no_new_rsi_low & (df['rsi_14'] < 30), 'rsi_divergence'] = 1  # Bullish divergence

        # MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd_line'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        df['stochastic_k'] = stoch.stoch()
        df['stochastic_d'] = stoch.stoch_signal()

        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            lbp=14
        ).williams_r()

        # CCI (Commodity Channel Index)
        df['cci_20'] = ta.trend.CCIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20
        ).cci()

        return df

    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features."""
        # Volume ratio (current vs average)
        avg_volume_20 = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (avg_volume_20 + 1e-10)

        # Volume trend
        volume_sma_5 = df['volume'].rolling(window=5).mean()
        volume_sma_20 = df['volume'].rolling(window=20).mean()
        df['volume_trend'] = (volume_sma_5 - volume_sma_20) / (volume_sma_20 + 1e-10)

        # Volume breakout (2 standard deviations above mean)
        volume_std = df['volume'].rolling(window=20).std()
        df['volume_breakout'] = (df['volume'] > (avg_volume_20 + 2 * volume_std)).astype(float)

        # Price-Volume Divergence
        # Calculate correlation between price and volume over rolling window
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()

        df['volume_divergence'] = price_change.rolling(window=20).corr(volume_change) * -1

        return df

    def _calculate_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        # Simple support/resistance based on recent lows/highs
        window = 20

        # Support level (recent low)
        df['support_level'] = df['low'].rolling(window=window).min()

        # Resistance level (recent high)
        df['resistance_level'] = df['high'].rolling(window=window).max()

        # Support strength (how many times price bounced off support)
        support_touches = (
                (df['low'] <= df['support_level'] * 1.02) &  # Within 2% of support
                (df['close'] > df['support_level'])  # But closed above
        ).rolling(window=window).sum()
        df['support_strength'] = support_touches / window

        # Resistance strength (how many times price was rejected at resistance)
        resistance_touches = (
                (df['high'] >= df['resistance_level'] * 0.98) &  # Within 2% of resistance
                (df['close'] < df['resistance_level'])  # But closed below
        ).rolling(window=window).sum()
        df['resistance_strength'] = resistance_touches / window

        # Distance from support/resistance (normalized)
        price_range = df['resistance_level'] - df['support_level'] + 1e-10
        df['support_distance'] = (df['close'] - df['support_level']) / price_range
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / price_range

        return df

    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-related features."""
        # Short-term trend (5-day)
        df['trend_short'] = np.sign(df['close'].rolling(window=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
        ))

        # Medium-term trend (20-day)
        df['trend_medium'] = np.sign(df['close'].rolling(window=20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        ))

        # Long-term trend (60-day)
        df['trend_long'] = np.sign(df['close'].rolling(window=60).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 60 else 0
        ))

        # Trend strength (R-squared of linear regression)
        def calculate_r_squared(x):
            if len(x) < 2:
                return 0
            try:
                slope, intercept = np.polyfit(range(len(x)), x, 1)
                y_pred = slope * np.arange(len(x)) + intercept
                ss_res = np.sum((x - y_pred) ** 2)
                ss_tot = np.sum((x - np.mean(x)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-10))
            except:
                return 0

        df['trend_strength'] = df['close'].rolling(window=20).apply(calculate_r_squared)

        # Trend acceleration (change in trend slope)
        trend_slope = df['close'].rolling(window=10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        df['trend_acceleration'] = trend_slope.diff()

        # Trend consistency (how often price is above/below moving average)
        ma_20 = df['close'].rolling(window=20).mean()
        df['trend_consistency'] = (df['close'] > ma_20).rolling(window=20).mean() * 2 - 1

        return df

    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features."""
        # Volatility regime (high/normal/low)
        vol_percentile = df['volatility_20d'].rolling(window=252).rank(pct=True)
        df['volatility_regime'] = pd.cut(
            vol_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=[-1, 0, 1],
            include_lowest=True
        ).astype(float).fillna(0)

        # Trend regime (trending/ranging)
        # Use ADX or trend strength as proxy
        df['trend_regime'] = (df['trend_strength'] > 0.5).astype(float)

        # Volume regime (high/normal/low volume)
        volume_percentile = df['volume'].rolling(window=252).rank(pct=True)
        df['volume_regime'] = pd.cut(
            volume_percentile,
            bins=[0, 0.33, 0.67, 1.0],
            labels=[-1, 0, 1],
            include_lowest=True
        ).astype(float).fillna(0)

        return df

    def _calculate_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signal features."""
        # Mean reversion signal
        # Z-score of price relative to moving average
        ma_20 = df['close'].rolling(window=20).mean()
        ma_std = df['close'].rolling(window=20).std()
        z_score = (df['close'] - ma_20) / (ma_std + 1e-10)

        # Mean reversion when price is extended
        df['mean_reversion_signal'] = 0
        df.loc[z_score < -2, 'mean_reversion_signal'] = 1  # Oversold
        df.loc[z_score > 2, 'mean_reversion_signal'] = -1  # Overbought

        # Momentum signal
        # Based on multiple timeframe momentum alignment
        mom_5 = df['returns_5d']
        mom_20 = df['returns_20d']

        df['momentum_signal'] = 0
        df.loc[(mom_5 > 0) & (mom_20 > 0) & (df['trend_strength'] > 0.5), 'momentum_signal'] = 1
        df.loc[(mom_5 < 0) & (mom_20 < 0) & (df['trend_strength'] > 0.5), 'momentum_signal'] = -1

        return df

    def _calculate_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonal effect features."""
        # Day of week effect (Monday = 0, Sunday = 6)
        df['day_of_week'] = df.index.dayofweek

        # Calculate average returns by day of week
        dow_returns = df.groupby('day_of_week')['returns_1d'].transform('mean')
        df['day_of_week_effect'] = df['returns_1d'] - dow_returns

        # Month effect
        df['month'] = df.index.month
        month_returns = df.groupby('month')['returns_1d'].transform('mean')
        df['month_effect'] = df['returns_1d'] - month_returns

        # Quarter effect
        df['quarter'] = df.index.quarter
        quarter_returns = df.groupby('quarter')['returns_1d'].transform('mean')
        df['quarter_effect'] = df['returns_1d'] - quarter_returns

        return df

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk-related metrics."""
        window = 20

        # Downside volatility (volatility of negative returns only)
        negative_returns = df['returns_1d'].copy()
        negative_returns[negative_returns > 0] = 0
        df['downside_volatility'] = negative_returns.rolling(window=window).std() * np.sqrt(252)

        # Maximum drawdown
        cumulative_returns = (1 + df['returns_1d']).cumprod()
        running_max = cumulative_returns.rolling(window=window, min_periods=1).max()
        drawdown = (cumulative_returns - running_max) / running_max
        df['max_drawdown'] = drawdown.rolling(window=window).min()

        # Value at Risk (95% confidence)
        df['var_95'] = df['returns_1d'].rolling(window=window).quantile(0.05)

        # Sharpe ratio components (for later calculation)
        # Store rolling mean and std of returns
        df['rolling_mean_return'] = df['returns_1d'].rolling(window=window).mean()
        df['rolling_std_return'] = df['returns_1d'].rolling(window=window).std()

        # Create a single sharpe_components feature (simplified)
        # This represents the risk-adjusted return potential
        df['sharpe_components'] = df['rolling_mean_return'] / (df['rolling_std_return'] + 1e-10)

        return df

    def _calculate_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset correlation features."""
        # These features require market data (SPY returns)
        # If market_returns column exists, use it

        if 'market_returns' in df.columns:
            # Market correlation
            df['market_correlation'] = df['returns_1d'].rolling(window=60).corr(df['market_returns'])

            # Beta (covariance with market / market variance)
            covariance = df['returns_1d'].rolling(window=60).cov(df['market_returns'])
            market_variance = df['market_returns'].rolling(window=60).var()
            df['beta'] = covariance / (market_variance + 1e-10)

            # Beta stability (rolling standard deviation of beta)
            df['beta_stability'] = 1 / (df['beta'].rolling(window=20).std() + 1)

            # Sector relative strength (simplified - using performance vs market)
            df['cumulative_return'] = (1 + df['returns_1d']).cumprod()
            df['market_cumulative_return'] = (1 + df['market_returns']).cumprod()
            df['sector_relative_strength'] = (
                    df['cumulative_return'] / df['market_cumulative_return']
            ).pct_change(20)
        else:
            # If no market data, fill with zeros
            df['market_correlation'] = 0
            df['sector_relative_strength'] = 0
            df['beta_stability'] = 1

        return df


def engineer_base_features(
        stock_data: Dict[str, pd.DataFrame],
        save_to_db: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Engineer base features for multiple stocks.

    Args:
        stock_data: Dictionary mapping symbols to DataFrames
        save_to_db: Whether to save features to database

    Returns:
        Dictionary mapping symbols to feature DataFrames
    """
    from tqdm import tqdm
    from data.database import DatabaseManager

    calculator = BaseFeatureCalculator()
    feature_data = {}

    if save_to_db:
        db = DatabaseManager()

    for symbol, df in tqdm(stock_data.items(), desc="Calculating features"):
        try:
            # Calculate all features
            features = calculator.calculate_all_features(df)

            # Remove any remaining NaN values
            features = features.fillna(0)

            # Store in dictionary
            feature_data[symbol] = features

            # Save to database if requested
            if save_to_db and not features.empty:
                for timestamp, row in features.iterrows():
                    db.save_features(
                        symbol=symbol,
                        timestamp=timestamp,
                        features=row.to_dict()
                    )

        except Exception as e:
            print(f"Error calculating features for {symbol}: {e}")
            continue

    return feature_data


if __name__ == "__main__":
    # Test the feature calculator
    from data.collector import fetch_data_for_training

    print("Testing base feature calculation...")

    # Fetch some test data
    test_symbols = ["AAPL", "MSFT"]
    stock_data = fetch_data_for_training(symbols=test_symbols, years_back=1)

    # Calculate features
    feature_data = engineer_base_features(stock_data)

    # Display results
    for symbol, features in feature_data.items():
        print(f"\n{symbol} Features:")
        print(f"Shape: {features.shape}")
        print(f"Columns: {features.columns.tolist()}")
        print("\nFirst few rows:")
        print(features.head())

        # Check for NaN values
        nan_counts = features.isna().sum()
        if nan_counts.any():
            print(f"\nWarning: NaN values found in {symbol}:")
            print(nan_counts[nan_counts > 0])
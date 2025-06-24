# features/interactions.py
"""
Feature interactions module implementing all 35 engineered feature interactions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

from config.settings import BASE_FEATURES


class FeatureInteractionCalculator:
    """Calculate complex feature interactions for enhanced signal generation."""

    def __init__(self):
        """Initialize the interaction calculator."""
        self.interaction_names = [
            # Volume-based interactions
            'trend_volume_alignment',
            'golden_cross_volume_confirmed',
            'death_cross_volume_confirmed',
            'breakout_volume_confirmation',

            # Support/Resistance interactions
            'support_bounce_confirmed',
            'resistance_break_confirmed',
            'support_break_volume',
            'resistance_bounce_volume',

            # Multi-timeframe interactions
            'multi_timeframe_trend_alignment',
            'timeframe_divergence_score',
            'short_medium_trend_confluence',
            'medium_long_trend_confluence',

            # Cross-timeframe confirmations
            'cross_timeframe_momentum',
            'cross_timeframe_reversal',

            # Oversold/Overbought interactions
            'oversold_bounce_setup',
            'overbought_reversal_setup',

            # Extreme value interactions
            'extreme_rsi_reversal',
            'extreme_volume_momentum',

            # Momentum interactions
            'momentum_exhaustion_signal',
            'momentum_acceleration_signal',

            # Mean reversion interactions
            'mean_reversion_quality',
            'mean_reversion_with_support',

            # Trend continuation
            'trend_continuation_quality',
            'trend_pullback_opportunity',

            # Risk-adjusted signals
            'risk_adjusted_momentum',
            'risk_adjusted_breakout',

            # Volatility patterns
            'volatility_breakout_signal',
            'volatility_contraction_breakout',

            # Divergence patterns
            'price_momentum_divergence',
            'volume_price_divergence_signal',

            # Consolidation patterns
            'consolidation_breakout_power',
            'range_expansion_signal',

            # Market relative signals
            'market_relative_strength_signal',
            'sector_rotation_signal',

            # Alpha generation
            'alpha_momentum_signal'
        ]

    def calculate_all_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all feature interactions.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with interaction features added
        """
        # Create a copy to avoid modifying original
        features = df.copy()

        # Initialize all interaction features with zeros
        for interaction in self.interaction_names:
            features[interaction] = 0.0

        try:
            # Volume-based interactions
            features = self._calculate_volume_interactions(features)

            # Support/Resistance interactions
            features = self._calculate_support_resistance_interactions(features)

            # Multi-timeframe interactions
            features = self._calculate_timeframe_interactions(features)

            # Extreme condition interactions
            features = self._calculate_extreme_interactions(features)

            # Momentum interactions
            features = self._calculate_momentum_interactions(features)

            # Mean reversion interactions
            features = self._calculate_mean_reversion_interactions(features)

            # Trend interactions
            features = self._calculate_trend_interactions(features)

            # Risk-adjusted interactions
            features = self._calculate_risk_adjusted_interactions(features)

            # Pattern-based interactions
            features = self._calculate_pattern_interactions(features)

            # Market-relative interactions
            features = self._calculate_market_relative_interactions(features)

        except Exception as e:
            # If any calculation fails, return features with zero interactions
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error calculating some interactions: {e}")

        return features

    def _safe_calculate(self, features: pd.DataFrame, calculation_func, feature_name: str):
        """Safely calculate a feature, returning zeros if it fails."""
        try:
            return calculation_func()
        except Exception:
            return pd.Series(0, index=features.index)

    def _calculate_volume_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based interaction features."""
        try:
            # Trend-volume alignment
            if 'trend_strength' in df.columns and 'volume_trend' in df.columns:
                df['trend_volume_alignment'] = df['trend_strength'] * (1 + df['volume_trend'])

            # Golden cross with volume confirmation
            if 'ma_crossover' in df.columns and 'volume_ratio' in df.columns:
                golden_cross = (df['ma_crossover'] == 1).astype(float)
                volume_surge = (df['volume_ratio'] > 1.5).astype(float)
                df['golden_cross_volume_confirmed'] = golden_cross * volume_surge

                # Death cross with volume confirmation
                death_cross = (df['ma_crossover'] == -1).astype(float)
                df['death_cross_volume_confirmed'] = death_cross * volume_surge

            # Breakout with volume confirmation
            if 'resistance_level' in df.columns and 'volume_breakout' in df.columns:
                # Use returns as proxy for price if close not available
                if 'returns_1d' in df.columns:
                    price_proxy = df['returns_1d'].cumsum()
                    at_resistance = (price_proxy >= price_proxy.rolling(20).quantile(0.95)).astype(float)
                    df['breakout_volume_confirmation'] = at_resistance * df['volume_breakout']

        except Exception as e:
            pass

        return df

    def _calculate_support_resistance_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support/resistance interaction features."""
        try:
            # Support bounce with confirmation
            if all(col in df.columns for col in ['support_distance', 'returns_1d', 'volume_ratio']):
                near_support = (df['support_distance'] < 0.02).astype(float)
                price_bounce = (df['returns_1d'] > 0).astype(float)
                df['support_bounce_confirmed'] = near_support * price_bounce * df['volume_ratio']

            # Resistance break with confirmation
            if all(col in df.columns for col in ['resistance_distance', 'momentum_signal', 'volume_breakout']):
                above_resistance = (df['resistance_distance'] < 0).astype(float)
                momentum_positive = (df['momentum_signal'] > 0).astype(float)
                df['resistance_break_confirmed'] = above_resistance * momentum_positive * df['volume_breakout']

            # Support break with volume (bearish)
            if all(col in df.columns for col in ['support_distance', 'volume_breakout']):
                below_support = (df['support_distance'] < -0.02).astype(float)
                df['support_break_volume'] = below_support * df['volume_breakout'] * -1

            # Resistance bounce with volume (bearish)
            if all(col in df.columns for col in ['resistance_distance', 'returns_1d', 'volume_ratio']):
                near_resistance = (df['resistance_distance'] < 0.02).astype(float)
                price_decline = (df['returns_1d'] < 0).astype(float)
                df['resistance_bounce_volume'] = near_resistance * price_decline * df['volume_ratio'] * -1

        except Exception as e:
            pass

        return df

    def _calculate_timeframe_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe interaction features."""
        try:
            # Multi-timeframe trend alignment
            if all(col in df.columns for col in ['trend_short', 'trend_medium', 'trend_long']):
                trend_sum = df['trend_short'] + df['trend_medium'] + df['trend_long']
                df['multi_timeframe_trend_alignment'] = trend_sum / 3.0

                # Timeframe divergence score
                trend_variance = np.var([df['trend_short'], df['trend_medium'], df['trend_long']], axis=0)
                df['timeframe_divergence_score'] = trend_variance

            # Trend confluence features
            if all(col in df.columns for col in ['trend_short', 'trend_medium', 'trend_strength']):
                short_medium_agree = (df['trend_short'] == df['trend_medium']).astype(float)
                df['short_medium_trend_confluence'] = short_medium_agree * df['trend_strength']

            if all(col in df.columns for col in ['trend_medium', 'trend_long', 'trend_strength']):
                medium_long_agree = (df['trend_medium'] == df['trend_long']).astype(float)
                df['medium_long_trend_confluence'] = medium_long_agree * df['trend_strength']

            # Cross-timeframe momentum
            if all(col in df.columns for col in ['returns_1d', 'returns_5d', 'returns_20d', 'momentum_signal']):
                returns_aligned = (
                        (df['returns_1d'] > 0) &
                        (df['returns_5d'] > 0) &
                        (df['returns_20d'] > 0)
                ).astype(float)
                df['cross_timeframe_momentum'] = returns_aligned * df['momentum_signal']

            # Cross-timeframe reversal signal
            if all(col in df.columns for col in ['trend_short', 'trend_long', 'rsi_14']):
                short_term_reversal = (
                        (df['trend_short'] * df['trend_long'] < 0) &  # Opposite directions
                        (abs(df['rsi_14'] - 50) > 20)  # RSI extreme
                ).astype(float)
                df['cross_timeframe_reversal'] = short_term_reversal

        except Exception as e:
            pass

        return df

    def _calculate_extreme_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate extreme condition interaction features."""
        try:
            # Oversold bounce setup
            if all(col in df.columns for col in ['rsi_14', 'stochastic_k', 'support_distance']):
                oversold_rsi = (df['rsi_14'] < 30).astype(float)
                oversold_stoch = (df['stochastic_k'] < 20).astype(float)
                at_support = (df['support_distance'] < 0.05).astype(float)
                df['oversold_bounce_setup'] = oversold_rsi * oversold_stoch * at_support

            # Overbought reversal setup
            if all(col in df.columns for col in ['rsi_14', 'stochastic_k', 'resistance_distance']):
                overbought_rsi = (df['rsi_14'] > 70).astype(float)
                overbought_stoch = (df['stochastic_k'] > 80).astype(float)
                at_resistance = (df['resistance_distance'] < 0.05).astype(float)
                df['overbought_reversal_setup'] = overbought_rsi * overbought_stoch * at_resistance * -1

            # Extreme RSI reversal
            if all(col in df.columns for col in ['rsi_14', 'rsi_divergence']):
                extreme_rsi = ((df['rsi_14'] < 20) | (df['rsi_14'] > 80)).astype(float)
                df['extreme_rsi_reversal'] = extreme_rsi * df['rsi_divergence']

            # Extreme volume with momentum
            if all(col in df.columns for col in ['volume_ratio', 'momentum_signal']):
                extreme_volume = (df['volume_ratio'] > 2).astype(float)
                df['extreme_volume_momentum'] = extreme_volume * df['momentum_signal']

        except Exception as e:
            pass

        return df

    def _calculate_momentum_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based interaction features."""
        try:
            # Momentum exhaustion signal
            if all(col in df.columns for col in ['momentum_signal', 'rsi_divergence', 'volume_trend']):
                strong_momentum = (abs(df['momentum_signal']) == 1).astype(float)
                divergence = (df['rsi_divergence'] != 0).astype(float)
                decreasing_volume = (df['volume_trend'] < 0).astype(float)
                df['momentum_exhaustion_signal'] = strong_momentum * divergence * decreasing_volume

            # Momentum acceleration signal
            if all(col in df.columns for col in ['trend_acceleration', 'volume_trend', 'trend_strength']):
                momentum_increasing = (df['trend_acceleration'] > 0).astype(float)
                volume_increasing = (df['volume_trend'] > 0).astype(float)
                df['momentum_acceleration_signal'] = momentum_increasing * volume_increasing * df['trend_strength']

        except Exception as e:
            pass

        return df

    def _calculate_mean_reversion_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean reversion interaction features."""
        try:
            # Mean reversion quality
            if all(col in df.columns for col in ['mean_reversion_signal', 'trend_strength']):
                # Use returns to calculate z-score if close prices not available
                if 'returns_20d' in df.columns:
                    z_score = (df['returns_20d'] - df['returns_20d'].rolling(20).mean()) / df['returns_20d'].rolling(
                        20).std()
                    extreme_zscore = (abs(z_score) > 2).astype(float)
                    low_trend_strength = (df['trend_strength'] < 0.3).astype(float)
                    df['mean_reversion_quality'] = extreme_zscore * low_trend_strength * df['mean_reversion_signal']

            # Mean reversion with support/resistance
            if all(col in df.columns for col in ['mean_reversion_signal', 'support_distance', 'resistance_distance']):
                at_key_level = (
                        (df['support_distance'] < 0.05) |
                        (df['resistance_distance'] < 0.05)
                ).astype(float)
                df['mean_reversion_with_support'] = df['mean_reversion_signal'] * at_key_level

        except Exception as e:
            pass

        return df

    def _calculate_trend_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based interaction features."""
        try:
            # Trend continuation quality
            if all(col in df.columns for col in ['trend_strength', 'momentum_signal', 'trend_medium', 'volume_ratio']):
                strong_trend = (df['trend_strength'] > 0.7).astype(float)
                aligned_momentum = (df['momentum_signal'] * df['trend_medium'] > 0).astype(float)
                df['trend_continuation_quality'] = strong_trend * aligned_momentum * df['volume_ratio']

            # Trend pullback opportunity
            if all(col in df.columns for col in ['returns_5d', 'trend_long', 'ma_alignment']):
                pullback = (df['returns_5d'] * df['trend_long'] < 0).astype(float)  # Against trend
                strong_long_trend = (abs(df['trend_long']) == 1).astype(float)
                not_broken = (df['ma_alignment'] != 0).astype(float)  # MAs still aligned
                df['trend_pullback_opportunity'] = pullback * strong_long_trend * not_broken

        except Exception as e:
            pass

        return df

    def _calculate_risk_adjusted_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk-adjusted interaction features."""
        try:
            # Risk-adjusted momentum
            if all(col in df.columns for col in ['momentum_signal', 'volatility_20d']):
                vol_weight = 1 / (df['volatility_20d'] + 0.1)  # Add small constant to avoid division by zero
                df['risk_adjusted_momentum'] = df['momentum_signal'] * vol_weight

            # Risk-adjusted breakout
            if all(col in df.columns for col in ['breakout_volume_confirmation', 'volatility_regime']):
                low_vol_regime = (df['volatility_regime'] == -1).astype(float)
                df['risk_adjusted_breakout'] = df['breakout_volume_confirmation'] * (1 + low_vol_regime)

        except Exception as e:
            pass

        return df

    def _calculate_pattern_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern-based interaction features."""
        try:
            # Volatility breakout signal
            if 'volatility_20d' in df.columns:
                vol_percentile = df['volatility_20d'].rolling(60).rank(pct=True)
                low_vol = (vol_percentile < 0.2).astype(float)
                vol_expansion = (df['volatility_20d'].diff() > 0).astype(float)
                df['volatility_breakout_signal'] = low_vol.shift(1) * vol_expansion

            # Volatility contraction breakout
            if all(col in df.columns for col in ['volatility_20d', 'returns_1d']):
                vol_contracting = (df['volatility_20d'].rolling(10).apply(lambda x: x[-1] < x[0])).astype(float)
                price_breakout = (abs(df['returns_1d']) > df['volatility_20d'] / np.sqrt(252) * 2).astype(float)
                df['volatility_contraction_breakout'] = vol_contracting.shift(1) * price_breakout

            # Price-momentum divergence
            if all(col in df.columns for col in ['returns_20d', 'macd_histogram']):
                price_up = (df['returns_20d'] > 0).astype(float)
                momentum_down = (df['macd_histogram'] < 0).astype(float)
                df['price_momentum_divergence'] = (price_up * momentum_down - (1 - price_up) * (1 - momentum_down))

            # Enhanced volume-price divergence
            if all(col in df.columns for col in ['volume_divergence', 'volume_breakout']):
                df['volume_price_divergence_signal'] = df['volume_divergence'] * df['volume_breakout']

            # Consolidation breakout power
            if all(col in df.columns for col in ['volatility_20d', 'breakout_volume_confirmation']):
                # Use volatility as proxy for range
                tight_range = (df['volatility_20d'] < df['volatility_20d'].rolling(50).mean() * 0.5).astype(float)
                df['consolidation_breakout_power'] = tight_range.shift(1) * df['breakout_volume_confirmation']

            # Range expansion signal
            if all(col in df.columns for col in ['volatility_20d', 'volume_breakout']):
                range_expansion = (df['volatility_20d'] > df['volatility_20d'].rolling(20).mean() * 1.5).astype(float)
                df['range_expansion_signal'] = range_expansion * df['volume_breakout']

        except Exception as e:
            pass

        return df

    def _calculate_market_relative_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market-relative interaction features."""
        try:
            # Market relative strength signal
            if 'sector_relative_strength' in df.columns and 'momentum_signal' in df.columns:
                outperforming = (df['sector_relative_strength'] > 0).astype(float)
                df['market_relative_strength_signal'] = outperforming * df['momentum_signal']

            # Sector rotation signal
            if 'sector_relative_strength' in df.columns and 'volume_ratio' in df.columns:
                rs_improving = (df['sector_relative_strength'].diff(5) > 0).astype(float)
                df['sector_rotation_signal'] = rs_improving * df['volume_ratio']

            # Alpha momentum signal
            if all(col in df.columns for col in ['market_correlation', 'momentum_signal', 'sector_relative_strength']):
                low_correlation = (abs(df['market_correlation']) < 0.5).astype(float)
                outperforming = (df['sector_relative_strength'] > 0).astype(float)
                df['alpha_momentum_signal'] = low_correlation * df['momentum_signal'] * outperforming
            else:
                # If market data not available, use simple momentum
                if 'momentum_signal' in df.columns:
                    df['alpha_momentum_signal'] = df['momentum_signal'] * 0.5

        except Exception as e:
            pass

        return df


def engineer_interaction_features(
        feature_data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Engineer interaction features for multiple stocks.

    Args:
        feature_data: Dictionary mapping symbols to DataFrames with base features

    Returns:
        Dictionary mapping symbols to DataFrames with all features
    """
    from tqdm import tqdm

    calculator = FeatureInteractionCalculator()
    enhanced_data = {}

    for symbol, df in tqdm(feature_data.items(), desc="Calculating interactions"):
        try:
            # Calculate all interactions
            enhanced_features = calculator.calculate_all_interactions(df)

            # Combine base features and interactions
            all_features = pd.concat([
                df[BASE_FEATURES],  # Original base features
                enhanced_features[calculator.interaction_names]  # New interactions
            ], axis=1)

            # Remove any remaining NaN values
            all_features = all_features.fillna(0)

            # Store in dictionary
            enhanced_data[symbol] = all_features

        except Exception as e:
            print(f"Error calculating interactions for {symbol}: {e}")
            # If interactions fail, at least return base features
            enhanced_data[symbol] = df[BASE_FEATURES].fillna(0)

    return enhanced_data


if __name__ == "__main__":
    # Test the interaction calculator
    from data.collector import fetch_data_for_training
    from features.base_features import engineer_base_features

    print("Testing feature interaction calculation...")

    # Fetch some test data
    test_symbols = ["AAPL"]
    stock_data = fetch_data_for_training(symbols=test_symbols, years_back=1)

    # Calculate base features first
    base_features = engineer_base_features(stock_data)

    # Calculate interactions
    all_features = engineer_interaction_features(base_features)

    # Display results
    for symbol, features in all_features.items():
        print(f"\n{symbol} Complete Features:")
        print(f"Shape: {features.shape}")
        print(f"Total features: {len(features.columns)}")
        print(f"\nBase features: {len(BASE_FEATURES)}")
        print(f"Interaction features: {len(features.columns) - len(BASE_FEATURES)}")

        # Show some interaction examples
        interaction_cols = [col for col in features.columns if col not in BASE_FEATURES]
        print(f"\nSample interaction features: {interaction_cols[:5]}")

        # Check for NaN values
        nan_counts = features.isna().sum()
        if nan_counts.any():
            print(f"\nWarning: NaN values found:")
            print(nan_counts[nan_counts > 0])
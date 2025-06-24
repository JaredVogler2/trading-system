# monitoring/mock_data_generators.py
"""
Mock data generators for testing when real data is unavailable.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random


class MockDataGenerator:
    """Generates realistic mock data for testing."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the mock data generator."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate_price_data(
            self,
            symbol: str,
            days: int = 365,
            start_price: float = 100.0,
            volatility: float = 0.02
    ) -> pd.DataFrame:
        """Generate mock price data."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate random walk
        returns = np.random.normal(0.0002, volatility, days)
        prices = start_price * (1 + returns).cumprod()

        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_vol = volatility * close
            high = close + np.random.uniform(0, daily_vol)
            low = close - np.random.uniform(0, daily_vol)
            open_price = close + np.random.uniform(-daily_vol / 2, daily_vol / 2)
            volume = np.random.randint(1000000, 10000000)

            data.append({
                'date': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df

    def generate_portfolio_history(
            self,
            initial_capital: float = 100000,
            days: int = 30,
            target_return: float = 0.15,
            volatility: float = 0.15
    ) -> pd.DataFrame:
        """Generate mock portfolio history."""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Generate returns with slight positive bias
        daily_return_mean = target_return / 252
        daily_vol = volatility / np.sqrt(252)
        daily_returns = np.random.normal(daily_return_mean, daily_vol, days)

        # Calculate portfolio values
        portfolio_values = initial_capital * (1 + daily_returns).cumprod()

        return pd.DataFrame({
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns,
            'cumulative_return': (portfolio_values / initial_capital) - 1
        }, index=dates)

    def generate_positions(
            self,
            num_positions: int = 5,
            symbols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate mock positions."""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'AMD']

        positions = []
        selected_symbols = random.sample(symbols, min(num_positions, len(symbols)))

        for symbol in selected_symbols:
            qty = random.randint(10, 100)
            entry_price = random.uniform(50, 500)
            current_price = entry_price * random.uniform(0.9, 1.2)

            positions.append({
                'symbol': symbol,
                'qty': qty,
                'avg_entry_price': entry_price,
                'current_price': current_price,
                'market_value': qty * current_price,
                'unrealized_pl': qty * (current_price - entry_price),
                'unrealized_plpc': (current_price - entry_price) / entry_price,
                'side': 'long'
            })

        return positions

    def generate_predictions(
            self,
            num_predictions: int = 50,
            symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Generate mock ML predictions."""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                       'SPY', 'QQQ', 'AMD', 'JPM', 'BAC', 'GS', 'MS', 'C']

        selected_symbols = random.sample(symbols, min(num_predictions, len(symbols)))

        predictions = []
        for symbol in selected_symbols:
            # Generate correlated return and confidence
            base_quality = np.random.beta(2, 2)  # Tends toward middle values

            predicted_return = np.random.normal(0.05 * base_quality, 0.03)
            predicted_return = np.clip(predicted_return, -0.15, 0.25)

            confidence = 0.5 + 0.4 * base_quality + np.random.normal(0, 0.05)
            confidence = np.clip(confidence, 0.3, 0.95)

            predictions.append({
                'symbol': symbol,
                'predicted_return': predicted_return,
                'confidence': confidence,
                'current_price': random.uniform(50, 500),
                'prediction_date': datetime.now()
            })

        return pd.DataFrame(predictions)

    def generate_trades(
            self,
            num_trades: int = 20,
            days_back: int = 30
    ) -> pd.DataFrame:
        """Generate mock trade history."""
        trades = []

        for i in range(num_trades):
            entry_date = datetime.now() - timedelta(days=random.randint(1, days_back))
            holding_days = random.randint(1, 21)
            exit_date = entry_date + timedelta(days=holding_days)

            # Generate P&L with realistic distribution
            win = random.random() < 0.55  # 55% win rate
            if win:
                pnl = random.uniform(100, 2000)
            else:
                pnl = -random.uniform(50, 1000)

            trades.append({
                'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']),
                'entry_date': entry_date,
                'exit_date': exit_date,
                'pnl': pnl,
                'return_pct': pnl / 10000,  # Approximate
                'holding_days': holding_days
            })

        return pd.DataFrame(trades)

    def generate_correlation_matrix(
            self,
            symbols: List[str],
            market_correlation: float = 0.6
    ) -> pd.DataFrame:
        """Generate mock correlation matrix."""
        n = len(symbols)

        # Start with market correlation
        corr_matrix = np.full((n, n), market_correlation)

        # Add some variation
        for i in range(n):
            for j in range(i + 1, n):
                # Add noise to correlation
                noise = np.random.uniform(-0.3, 0.3)
                corr_value = np.clip(market_correlation + noise, -1, 1)
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value

        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)

        return pd.DataFrame(corr_matrix, index=symbols, columns=symbols)


# Create a default instance
_default_generator = MockDataGenerator(seed=42)


def get_mock_generator() -> MockDataGenerator:
    """Get the default mock data generator instance."""
    return _default_generator
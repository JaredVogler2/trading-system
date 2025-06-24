# monitoring/real_data_manager.py
"""
Real data manager to replace all mock data with actual values.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Any
import yfinance as yf

from config.settings import DATA_DIR
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class RealDataManager:
    """Manages real data retrieval for the dashboard."""

    def __init__(self, trading_client=None, db_manager=None):
        self.trading_client = trading_client
        self.db = db_manager or DatabaseManager()
        self.cache = {}
        self.cache_expiry = {}

    def get_account_data(self):
        """Get real account data from Alpaca."""
        if not self.trading_client:
            # Try to load from shared state
            state_file = DATA_DIR / 'live_system_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    return state.get('alpaca_account', {})
            return {}

        try:
            account = self.trading_client.get_account()
            return {
                'account_number': account.account_number,
                'status': account.status,
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'unrealized_pl': float(account.unrealized_pl or 0),
                'realized_pl': float(account.unrealized_plpc or 0) * float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return {}

    def get_positions_data(self):
        """Get real positions from Alpaca or shared state."""
        # First try shared state
        state_file = DATA_DIR / 'live_system_state.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
                positions = state.get('positions', {})

                # Convert to list format
                positions_list = []
                for symbol, pos in positions.items():
                    positions_list.append({
                        'symbol': symbol,
                        'qty': pos.get('shares', 0),
                        'avg_entry_price': pos.get('entry_price', 0),
                        'current_price': pos.get('current_price', 0),
                        'market_value': pos.get('shares', 0) * pos.get('current_price', 0),
                        'unrealized_pl': pos.get('unrealized_pnl', 0),
                        'unrealized_plpc': pos.get('return_pct', 0),
                        'side': 'long'
                    })
                return positions_list

        # Try Alpaca API
        if self.trading_client:
            try:
                positions = self.trading_client.get_all_positions()
                return [{
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                } for pos in positions]
            except Exception as e:
                logger.error(f"Error getting positions: {e}")

        return []

    def get_portfolio_history(self):
        """Get real portfolio history from database."""
        try:
            # Get trades from database
            trades_df = self.db.get_recent_trades(days_back=365)

            if trades_df.empty:
                # No trades yet, create basic history from account value
                account_data = self.get_account_data()
                current_value = account_data.get('portfolio_value', 100000)

                # Create synthetic history
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                values = [current_value] * len(dates)

                return pd.DataFrame({
                    'daily_return': [0] * len(dates),
                    'cumulative_return': [0] * len(dates)
                }, index=dates)

            # Calculate portfolio value over time from trades
            # Group by date and calculate cumulative P&L
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
            daily_pnl = trades_df.groupby('date')['profit_loss'].sum()

            # Create full date range
            date_range = pd.date_range(
                start=daily_pnl.index.min(),
                end=datetime.now().date(),
                freq='D'
            )

            # Fill missing days with 0
            daily_pnl = daily_pnl.reindex(date_range, fill_value=0)

            # Calculate cumulative returns
            initial_value = 100000  # Starting capital
            cumulative_pnl = daily_pnl.cumsum()
            portfolio_values = initial_value + cumulative_pnl

            # Calculate returns
            daily_returns = portfolio_values.pct_change().fillna(0)
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            return pd.DataFrame({
                'portfolio_value': portfolio_values,
                'daily_return': daily_returns,
                'cumulative_return': cumulative_returns
            })

        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            # Return minimal valid dataframe
            return pd.DataFrame({
                'daily_return': [0],
                'cumulative_return': [0]
            }, index=[datetime.now()])

    def calculate_correlation_matrix(self, positions_list):
        """Calculate real correlation matrix from position symbols."""
        if not positions_list:
            return pd.DataFrame()

        symbols = [pos['symbol'] for pos in positions_list]

        try:
            # Get price data for correlation calculation
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)

            price_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    price_data[symbol] = hist['Close']

            if not price_data:
                return pd.DataFrame()

            # Calculate returns
            returns_df = pd.DataFrame(price_data).pct_change().dropna()

            # Calculate correlation
            return returns_df.corr()

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()

    def calculate_risk_metrics(self, positions_data):
        """Calculate real risk metrics from positions."""
        if not positions_data:
            return {
                'portfolio_heat': 0,
                'max_position': 0,
                'avg_correlation': 0,
                'concentration': 0
            }

        # Calculate total portfolio value
        total_value = sum(pos['market_value'] for pos in positions_data)

        if total_value == 0:
            return {
                'portfolio_heat': 0,
                'max_position': 0,
                'avg_correlation': 0,
                'concentration': 0
            }

        # Position weights
        weights = [pos['market_value'] / total_value for pos in positions_data]

        # Portfolio heat (assume 5% risk per position)
        portfolio_heat = sum(w * 0.05 for w in weights)

        # Max position
        max_position = max(weights) if weights else 0

        # Average correlation
        corr_matrix = self.calculate_correlation_matrix(positions_data)
        if not corr_matrix.empty and len(corr_matrix) > 1:
            # Get upper triangle of correlation matrix (excluding diagonal)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            avg_correlation = corr_matrix.where(mask).stack().mean()
        else:
            avg_correlation = 0

        # Concentration (Herfindahl index)
        concentration = sum(w ** 2 for w in weights)

        return {
            'portfolio_heat': portfolio_heat,
            'max_position': max_position,
            'avg_correlation': avg_correlation,
            'concentration': concentration
        }

    def get_latest_predictions(self):
        """Get real predictions from latest file."""
        try:
            # Find latest predictions file
            prediction_files = list(DATA_DIR.glob("predictions_*.csv"))
            if not prediction_files:
                return pd.DataFrame()

            latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
            return pd.read_csv(latest_file)

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return pd.DataFrame()

    def get_feature_importance(self):
        """Get real feature importance from model training logs."""
        try:
            # Look for feature importance file saved during training
            feature_importance_file = DATA_DIR / 'feature_importance.json'
            if feature_importance_file.exists():
                with open(feature_importance_file, 'r') as f:
                    return json.load(f)

            # If not available, return empty dict
            return {}

        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            return {}

    def get_model_performance_metrics(self):
        """Get real model performance from training logs."""
        try:
            # Look for model metrics file
            metrics_file = DATA_DIR / 'model_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    return {
                        'accuracy': metrics.get('directional_accuracy', 0) * 100,
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'mse': metrics.get('mse', 0),
                        'mae': metrics.get('mae', 0)
                    }

            # Check backtest results in database
            backtest_results = self.db.db.query(
                "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

            if backtest_results:
                return {
                    'accuracy': backtest_results.get('win_rate', 0) * 100,
                    'sharpe': backtest_results.get('sharpe_ratio', 0),
                    'total_return': backtest_results.get('total_return', 0),
                    'max_drawdown': backtest_results.get('max_drawdown', 0)
                }

            return {
                'accuracy': 0,
                'sharpe': 0,
                'total_return': 0,
                'max_drawdown': 0
            }

        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {'accuracy': 0, 'sharpe': 0}

    def calculate_performance_metrics(self, portfolio_history):
        """Calculate real performance metrics from portfolio history."""
        if portfolio_history.empty:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }

        # Total return
        total_return = portfolio_history['cumulative_return'].iloc[-1]

        # Daily returns
        daily_returns = portfolio_history['daily_return'].dropna()

        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            excess_returns = daily_returns - 0.02 / 252  # 2% risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate from trades
        try:
            trades_df = self.db.get_recent_trades()
            if not trades_df.empty and 'profit_loss' in trades_df.columns:
                win_rate = (trades_df['profit_loss'] > 0).mean()
            else:
                win_rate = 0
        except:
            win_rate = 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
# monitoring/data_fetchers_fixed_v2.py
"""Fixed data fetching with correct Alpaca API attributes."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import yfinance as yf
from typing import Dict, List, Optional

from config.settings import DATA_DIR
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles all data fetching operations."""

    def __init__(self, trading_client, data_client, mock_tracker, utilities):
        self.trading_client = trading_client
        self.data_client = data_client
        self.mock_tracker = mock_tracker
        self.utilities = utilities
        self.db = DatabaseManager()

    def get_account_data(self):
        """Get real-time account data from Alpaca."""
        # Check for live system state first
        state_file = DATA_DIR / 'live_system_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    if 'alpaca_account' in state and state['alpaca_account']:
                        logger.info("Using account data from live system state")
                        self.mock_tracker.clear_mock_source('account')
                        return state['alpaca_account']
            except Exception as e:
                logger.debug(f"Could not load account from state: {e}")

        # If no Alpaca client, return mock data
        if not self.trading_client:
            self.mock_tracker.add_mock_source('account', 'Alpaca not connected')
            return self._get_mock_account_data()

        # Don't use cache for account data - always fetch fresh
        try:
            logger.info("Fetching fresh account data from Alpaca...")
            account = self.trading_client.get_account()

            # Get the actual attributes from Alpaca account
            portfolio_value = float(account.portfolio_value)
            last_equity = float(account.last_equity) if hasattr(account, 'last_equity') else portfolio_value
            equity = float(account.equity) if hasattr(account, 'equity') else portfolio_value

            # Calculate P&L
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0

            data = {
                'account_number': account.account_number,
                'status': account.status,
                'portfolio_value': portfolio_value,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': equity,
                'last_equity': last_equity,
                'daily_pl': daily_pl,
                'daily_pl_pct': daily_pl_pct,
                # For compatibility, map to expected fields
                'unrealized_pl': daily_pl,  # Use daily P&L as unrealized
                'realized_pl': 0  # Alpaca doesn't provide this directly
            }

            logger.info(
                f"Successfully fetched account data: Portfolio=${data['portfolio_value']:,.2f}, Daily P&L=${data['daily_pl']:,.2f}")
            self.mock_tracker.clear_mock_source('account')
            return data

        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            self.mock_tracker.add_mock_source('account', f'API error: {str(e)}')
            return self._get_mock_account_data()

    def get_positions_data(self):
        """Get real-time positions from Alpaca."""
        # First check shared state from live system
        state_file = DATA_DIR / 'live_system_state.json'
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                    # Prefer raw Alpaca positions if available
                    if 'alpaca_positions' in state and state['alpaca_positions']:
                        logger.info(f"Using {len(state['alpaca_positions'])} Alpaca positions from shared state")
                        self.mock_tracker.clear_mock_source('positions')
                        return state['alpaca_positions']

                    # Fallback to processed positions
                    if 'positions' in state and state['positions']:
                        # Convert dict format to list format
                        positions_list = []
                        for symbol, pos in state['positions'].items():
                            positions_list.append({
                                'symbol': symbol,
                                'qty': pos.get('qty', pos.get('shares', 0)),
                                'avg_entry_price': pos.get('avg_entry_price', pos.get('entry_price', 0)),
                                'current_price': pos.get('current_price', 0),
                                'market_value': pos.get('market_value',
                                                        pos.get('qty', 0) * pos.get('current_price', 0)),
                                'unrealized_pl': pos.get('unrealized_pl', pos.get('unrealized_pnl', 0)),
                                'unrealized_plpc': pos.get('unrealized_plpc', pos.get('return_pct', 0)),
                                'side': pos.get('side', 'long')
                            })
                        logger.info(f"Using {len(positions_list)} positions from live system state")
                        self.mock_tracker.clear_mock_source('positions')
                        return positions_list
            except Exception as e:
                logger.debug(f"Could not load positions from state: {e}")

        # If no state file or error, try direct Alpaca connection
        if not self.trading_client:
            self.mock_tracker.add_mock_source('positions', 'Alpaca not connected')
            return self._get_mock_positions_data()

        try:
            logger.info("Fetching positions directly from Alpaca...")
            positions = self.trading_client.get_all_positions()
            positions_data = []

            for pos in positions:
                # Calculate current price from market value and quantity
                qty = float(pos.qty)
                market_value = float(pos.market_value)
                current_price = market_value / qty if qty > 0 else float(pos.avg_entry_price)

                position_data = {
                    'symbol': pos.symbol,
                    'qty': qty,
                    'market_value': market_value,
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': current_price,
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
                positions_data.append(position_data)

            logger.info(f"Successfully fetched {len(positions_data)} positions from Alpaca")
            self.mock_tracker.clear_mock_source('positions')
            return positions_data

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            self.mock_tracker.add_mock_source('positions', f'API error: {str(e)}')
            return self._get_mock_positions_data()

    def get_latest_predictions(self):
        """Get latest AI predictions."""
        try:
            # Look for prediction files with various naming patterns
            prediction_patterns = [
                "predictions_*.csv",  # Note the underscore
                "news_predictions_*.csv",
                "multi_timeframe_predictions.csv"
            ]

            all_prediction_files = []
            for pattern in prediction_patterns:
                all_prediction_files.extend(list(DATA_DIR.glob(pattern)))

            if all_prediction_files:
                # Get the most recent file
                latest_file = max(all_prediction_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading predictions from: {latest_file.name}")
                df = pd.read_csv(latest_file)

                # Ensure required columns exist
                required_cols = ['symbol', 'predicted_return', 'confidence']
                if all(col in df.columns for col in required_cols):
                    self.mock_tracker.clear_mock_source('predictions')
                    logger.info(f"Loaded {len(df)} predictions from {latest_file.name}")
                    return df
                else:
                    logger.warning(f"Prediction file missing required columns: {df.columns.tolist()}")

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")

        # Return mock predictions
        self.mock_tracker.add_mock_source('predictions', 'No predictions generated')
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA'],
            'predicted_return': [0.08, 0.12, 0.06, 0.15, 0.09],
            'confidence': [0.82, 0.76, 0.88, 0.71, 0.79],
            'current_price': [175.23, 338.45, 2721.33, 421.18, 248.91]
        })

    def _get_mock_account_data(self):
        """Generate mock account data for testing."""
        return {
            'account_number': 'MOCK123456',
            'status': 'ACTIVE (MOCK)',
            'portfolio_value': 105000.0,
            'buying_power': 95000.0,
            'cash': 50000.0,
            'unrealized_pl': 2500.0,
            'realized_pl': 1500.0,
            'daily_pl': 4000.0,
            'equity': 105000.0,
            'last_equity': 101000.0,
            '_is_mock': True
        }

    def _get_mock_positions_data(self):
        """Generate mock positions data."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        positions = []

        for symbol in symbols:
            qty = np.random.randint(10, 100)
            entry_price = np.random.uniform(100, 300)
            current_price = entry_price * np.random.uniform(0.95, 1.15)

            positions.append({
                'symbol': symbol,
                'qty': qty,
                'avg_entry_price': entry_price,
                'current_price': current_price,
                'market_value': qty * current_price,
                'unrealized_pl': qty * (current_price - entry_price),
                'unrealized_plpc': (current_price - entry_price) / entry_price,
                'side': 'long',
                '_is_mock': True
            })

        return positions
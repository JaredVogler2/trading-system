# monitoring/realtime_data_sync.py
"""
Real-time data synchronization module for the dashboard.
Ensures all data is current and eliminates any mock data usage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import threading
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_DIR, WATCHLIST
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


class RealTimeDataSync:
    """Handles real-time data synchronization for the dashboard."""

    def __init__(self, trading_client=None, data_client=None):
        """Initialize the data sync module."""
        self.trading_client = trading_client
        self.data_client = data_client
        self.db = DatabaseManager()

        # Cache settings
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = {
            'account': 5,  # 5 seconds for account data
            'positions': 5,  # 5 seconds for positions
            'orders': 10,  # 10 seconds for orders
            'predictions': 60,  # 1 minute for predictions
            'news': 300,  # 5 minutes for news
            'market_data': 30,  # 30 seconds for market data
            'backtest': 3600  # 1 hour for backtest results
        }

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Data update tracking
        self.last_updates = {}

    def get_account_data(self) -> Dict[str, Any]:
        """Get real-time account data with caching."""
        cache_key = 'account'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self.trading_client:
            logger.error("No trading client available")
            return {}

        try:
            account = self.trading_client.get_account()

            # Calculate all necessary fields
            portfolio_value = float(account.portfolio_value)
            equity = float(account.equity)
            last_equity = float(account.last_equity) if hasattr(account, 'last_equity') else equity

            # Calculate P&L
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0

            # Get additional fields
            long_market_value = float(account.long_market_value) if hasattr(account, 'long_market_value') else 0
            short_market_value = float(account.short_market_value) if hasattr(account, 'short_market_value') else 0

            data = {
                'account_number': account.account_number,
                'status': account.status,
                'portfolio_value': portfolio_value,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': equity,
                'last_equity': last_equity,
                'daily_pl': daily_pl,
                'daily_pl_pct': daily_pl_pct,
                'long_market_value': long_market_value,
                'short_market_value': short_market_value,
                'initial_margin': float(account.initial_margin) if hasattr(account, 'initial_margin') else 0,
                'maintenance_margin': float(account.maintenance_margin) if hasattr(account,
                                                                                   'maintenance_margin') else 0,
                'daytrade_count': int(account.daytrade_count) if hasattr(account, 'daytrade_count') else 0,
                'pattern_day_trader': bool(account.pattern_day_trader) if hasattr(account,
                                                                                  'pattern_day_trader') else False,
                'last_update': datetime.now().isoformat()
            }

            # Update cache
            self._update_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error getting account data: {e}")
            return self.cache.get(cache_key, {})

    def get_positions_data(self) -> List[Dict[str, Any]]:
        """Get real-time positions data with enrichment."""
        cache_key = 'positions'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self.trading_client:
            return []

        try:
            positions = self.trading_client.get_all_positions()

            # Convert to enriched format
            position_data = []

            for pos in positions:
                # Calculate derived fields
                qty = float(pos.qty)
                avg_entry_price = float(pos.avg_entry_price)
                market_value = float(pos.market_value)
                current_price = market_value / qty if qty > 0 else avg_entry_price
                unrealized_pl = float(pos.unrealized_pl)
                unrealized_plpc = float(pos.unrealized_plpc)

                # Get additional market data if available
                try:
                    ticker = yf.Ticker(pos.symbol)
                    info = ticker.info

                    position_dict = {
                        'symbol': pos.symbol,
                        'qty': qty,
                        'side': pos.side,
                        'avg_entry_price': avg_entry_price,
                        'current_price': current_price,
                        'market_value': market_value,
                        'cost_basis': qty * avg_entry_price,
                        'unrealized_pl': unrealized_pl,
                        'unrealized_plpc': unrealized_plpc,
                        'last_update': datetime.now().isoformat()
                    }

                position_data.append(position_dict)

            # Update cache
            self._update_cache(cache_key, position_data)
            return position_data

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return self.cache.get(cache_key, [])

    def get_orders_data(self, status='all', limit=100) -> List[Dict[str, Any]]:
        """Get real-time orders data."""
        cache_key = f'orders_{status}_{limit}'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        if not self.trading_client:
            return []

        try:
            orders = self.trading_client.get_orders(status=status, limit=limit)

            order_data = []
            for order in orders:
                order_dict = {
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty) if order.qty else 0,
                    'side': order.side.value if hasattr(order.side, 'value') else str(order.side),
                    'type': order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                    'time_in_force': order.time_in_force.value if hasattr(order.time_in_force, 'value') else str(
                        order.time_in_force),
                    'status': order.status.value if hasattr(order.status, 'value') else str(order.status),
                    'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                    'filled_at': order.filled_at.isoformat() if hasattr(order,
                                                                        'filled_at') and order.filled_at else None,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'filled_qty': float(order.filled_qty) if hasattr(order, 'filled_qty') and order.filled_qty else 0,
                    'limit_price': float(order.limit_price) if hasattr(order,
                                                                       'limit_price') and order.limit_price else None,
                    'stop_price': float(order.stop_price) if hasattr(order,
                                                                     'stop_price') and order.stop_price else None,
                    'last_update': datetime.now().isoformat()
                }
                order_data.append(order_dict)

            # Update cache
            self._update_cache(cache_key, order_data)
            return order_data

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return self.cache.get(cache_key, [])

    def get_portfolio_history(self, days=365) -> pd.DataFrame:
        """Get portfolio performance history."""
        cache_key = f'portfolio_history_{days}'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            # Get account activity and trades from database
            trades_df = self.db.get_recent_trades(days_back=days)

            if trades_df.empty:
                # No trade history, create synthetic data from current portfolio
                account_data = self.get_account_data()
                if account_data:
                    current_value = account_data.get('portfolio_value', 100000)
                    dates = pd.date_range(end=datetime.now(), periods=min(days, 30), freq='D')

                    # Create simple history
                    history_df = pd.DataFrame({
                        'date': dates,
                        'portfolio_value': [current_value] * len(dates),
                        'daily_return': [0] * len(dates),
                        'cumulative_return': [0] * len(dates)
                    })
                    history_df.set_index('date', inplace=True)

                    self._update_cache(cache_key, history_df)
                    return history_df

            # Calculate portfolio performance from trades
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date

            # Group by date
            daily_pnl = trades_df.groupby('date')['profit_loss'].sum()

            # Create date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Calculate portfolio value over time
            initial_value = 100000  # Base value
            portfolio_values = pd.Series(index=date_range, dtype=float)
            portfolio_values.iloc[0] = initial_value

            for i, date in enumerate(date_range[1:], 1):
                prev_value = portfolio_values.iloc[i - 1]
                daily_pl = daily_pnl.get(date.date(), 0)
                portfolio_values.iloc[i] = prev_value + daily_pl

            # Calculate returns
            daily_returns = portfolio_values.pct_change().fillna(0)
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            history_df = pd.DataFrame({
                'portfolio_value': portfolio_values,
                'daily_return': daily_returns,
                'cumulative_return': cumulative_returns
            })

            # Update cache
            self._update_cache(cache_key, history_df)
            return history_df

        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()

    def get_market_data(self, symbols: List[str], period='1d') -> Dict[str, pd.DataFrame]:
        """Get real-time market data for multiple symbols."""
        cache_key = f'market_data_{"-".join(sorted(symbols))}_{period}'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        market_data = {}

        try:
            # Use thread pool for parallel fetching
            def fetch_symbol_data(symbol):
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    if not data.empty:
                        return symbol, data
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                return symbol, pd.DataFrame()

            # Fetch in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(fetch_symbol_data, symbol) for symbol in symbols]

                for future in futures:
                    symbol, data = future.result()
                    if not data.empty:
                        market_data[symbol] = data

            # Update cache
            self._update_cache(cache_key, market_data)
            return market_data

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return self.cache.get(cache_key, {})

    def get_latest_predictions(self) -> Optional[pd.DataFrame]:
        """Get latest ML predictions from file system."""
        cache_key = 'predictions'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            # Look for prediction files
            prediction_patterns = [
                "predictions_*.csv",
                "predictions.csv",
                "multi_timeframe_predictions.csv"
            ]

            all_files = []
            for pattern in prediction_patterns:
                all_files.extend(list(DATA_DIR.glob(pattern)))

            if not all_files:
                return None

            # Get most recent file
            latest_file = max(all_files, key=lambda x: x.stat().st_mtime)

            # Load and validate
            df = pd.read_csv(latest_file)

            # Ensure required columns
            required_cols = ['symbol', 'predicted_return', 'confidence']
            if all(col in df.columns for col in required_cols):
                # Add metadata
                df['prediction_file'] = latest_file.name
                df['file_timestamp'] = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

                # Update cache
                self._update_cache(cache_key, df)
                return df

            return None

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None

    def get_latest_news_analysis(self) -> Optional[pd.DataFrame]:
        """Get latest news analysis from file system."""
        cache_key = 'news_analysis'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            # Look for news files
            news_files = list(DATA_DIR.glob("news_predictions_*.csv"))

            if not news_files:
                return None

            # Get most recent
            latest_file = max(news_files, key=lambda x: x.stat().st_mtime)

            # Load and enhance
            df = pd.read_csv(latest_file)

            # Add metadata
            df['news_file'] = latest_file.name
            df['file_timestamp'] = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

            # Update cache
            self._update_cache(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Error loading news analysis: {e}")
            return None

    def get_backtest_results(self) -> Optional[Dict[str, Any]]:
        """Get latest backtest results."""
        cache_key = 'backtest_results'

        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        try:
            # Check file system
            backtest_file = DATA_DIR / 'backtest_results.json'

            if not backtest_file.exists():
                # Check database
                results = self.db.db.query(
                    "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 1"
                ).fetchone()

                if results:
                    backtest_data = {
                        'total_return': results['total_return'],
                        'sharpe_ratio': results['sharpe_ratio'],
                        'max_drawdown': results['max_drawdown'],
                        'win_rate': results['win_rate'],
                        'total_trades': results['total_trades'],
                        'avg_win': results['avg_win'],
                        'avg_loss': results['avg_loss'],
                        'start_date': results['start_date'].isoformat(),
                        'end_date': results['end_date'].isoformat(),
                        'metrics_data': results['metrics_data']
                    }

                    self._update_cache(cache_key, backtest_data)
                    return backtest_data

                return None

            # Load from file
            with open(backtest_file, 'r') as f:
                backtest_data = json.load(f)

            # Update cache
            self._update_cache(cache_key, backtest_data)
            return backtest_data

        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            return None

    def get_model_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest model performance metrics."""
        cache_key = 'model_metrics'

        # Check cache
        if self._is_cache_valid(cache_key, duration=3600):  # 1 hour cache
            return self.cache[cache_key]

        try:
            # Check for metrics file
            metrics_file = DATA_DIR / 'model_metrics.json'

            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                # Update cache
                self._update_cache(cache_key, metrics)
                return metrics

            return None

        except Exception as e:
            logger.error(f"Error loading model metrics: {e}")
            return None

    def calculate_performance_metrics(self, portfolio_history: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if portfolio_history.empty:
            return {}

        try:
            daily_returns = portfolio_history['daily_return'].dropna()

            if len(daily_returns) < 2:
                return {}

            # Basic metrics
            total_return = portfolio_history['cumulative_return'].iloc[-1]

            # Annualized metrics
            trading_days = len(daily_returns)
            years = trading_days / 252

            if years > 0:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = 0

            # Volatility
            daily_vol = daily_returns.std()
            annual_vol = daily_vol * np.sqrt(252)

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = daily_returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

            # Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

            # Maximum drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

            # Win rate
            winning_days = len(daily_returns[daily_returns > 0])
            total_days = len(daily_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0

            # Value at Risk (95%)
            var_95 = daily_returns.quantile(0.05)

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'var_95': var_95,
                'trading_days': trading_days
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def sync_all_data(self):
        """Synchronize all data sources."""
        try:
            # Account data
            self.get_account_data()

            # Positions
            self.get_positions_data()

            # Recent orders
            self.get_orders_data(limit=50)

            # Portfolio history
            self.get_portfolio_history()

            # Latest predictions
            self.get_latest_predictions()

            # News analysis
            self.get_latest_news_analysis()

            # Update last sync time
            self.last_updates['full_sync'] = datetime.now()

            logger.info("Full data sync completed")

        except Exception as e:
            logger.error(f"Error in full data sync: {e}")

    # Cache management methods

    def _is_cache_valid(self, key: str, duration: Optional[int] = None) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache_expiry:
            return False

        if duration is None:
            duration = self.cache_duration.get(key.split('_')[0], 60)

        return (datetime.now() - self.cache_expiry[key]).total_seconds() < duration

    def _update_cache(self, key: str, data: Any):
        """Update cache with new data."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now()

    def clear_cache(self, key: Optional[str] = None):
        """Clear cache for specific key or all keys."""
        if key:
            self.cache.pop(key, None)
            self.cache_expiry.pop(key, None)
        else:
            self.cache.clear()
            self.cache_expiry.clear()

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status."""
        status = {}

        for key, expiry_time in self.cache_expiry.items():
            age = (datetime.now() - expiry_time).total_seconds()
            base_key = key.split('_')[0]
            max_age = self.cache_duration.get(base_key, 60)

            status[key] = {
                'age_seconds': age,
                'max_age_seconds': max_age,
                'is_valid': age < max_age,
                'expires_in': max(0, max_age - age)
            }

        return status


# Singleton instance for global access
_data_sync_instance = None


def get_data_sync() -> RealTimeDataSync:
    """Get or create the data sync singleton."""
    global _data_sync_instance

    if _data_sync_instance is None:
        # Try to create with API connections
        try:
            trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            _data_sync_instance = RealTimeDataSync(trading_client, data_client)
        except:
            # Create without API connections
            _data_sync_instance = RealTimeDataSync()

    return _data_sync_instance


# Background sync thread
def run_background_sync(interval=30):
    """Run data sync in background thread."""
    data_sync = get_data_sync()

    while True:
        try:
            data_sync.sync_all_data()
        except Exception as e:
            logger.error(f"Background sync error: {e}")

        time.sleep(interval)


# Start background sync when module is imported
sync_thread = threading.Thread(target=run_background_sync, daemon=True)
sync_thread.start()
entry_price,
'current_price': current_price,
'market_value': market_value,
'cost_basis': qty * avg_entry_price,
'unrealized_pl': unrealized_pl,
'unrealized_plpc': unrealized_plpc,
'change_today': info.get('regularMarketChangePercent', 0),
'volume': info.get('volume', 0),
'avg_volume': info.get('averageVolume', 0),
'market_cap': info.get('marketCap', 0),
'pe_ratio': info.get('trailingPE', 0),
'last_update': datetime.now().isoformat()
}
except:
# Fallback if yfinance fails
position_dict = {
    'symbol': pos.symbol,
    'qty': qty,
    'side': pos.side,
    'avg_entry_price': avg_
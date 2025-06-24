# backtesting/comprehensive_backtester.py
"""
Comprehensive backtesting system with proper temporal separation and realistic simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

from config.settings import BACKTEST_CONFIG, PREDICTION_CONFIG, WATCHLIST
from data.database import DatabaseManager
from features.pipeline import FeatureEngineeringPipeline
from models.neural_networks import make_ensemble_predictions

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    position_value: float = 0.0
    exit_reason: Optional[str] = None
    predicted_return: float = 0.0
    confidence: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def pnl(self) -> float:
        if not self.exit_price:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        if not self.exit_price or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def holding_days(self) -> int:
        if not self.exit_date:
            return 0
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResults:
    """Container for backtest results and metrics."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    positions_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Other metrics
    avg_holding_days: float = 0.0
    turnover_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'avg_holding_days': self.avg_holding_days,
            'turnover_rate': self.turnover_rate,
            'equity_curve': self.equity_curve.tolist() if not self.equity_curve.empty else [],
            'dates': [d.isoformat() for d in self.equity_curve.index] if not self.equity_curve.empty else []
        }


class ComprehensiveBacktester:
    """
    Advanced backtesting engine with realistic simulation and no data leakage.
    """

    def __init__(
            self,
            initial_capital: float = 100000,
            commission: float = 0.001,  # 0.1% per trade
            slippage: float = 0.001,  # 0.1% slippage
            position_size: float = 0.02,  # 2% per position
            max_positions: int = 10,
            stop_loss: float = 0.05,  # 5% stop loss
            take_profit: float = 0.15,  # 15% take profit
            confidence_threshold: float = 0.7,
            min_predicted_return: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.confidence_threshold = confidence_threshold
        self.min_predicted_return = min_predicted_return

        # State tracking
        self.cash = initial_capital
        self.positions: Dict[str, Trade] = {}
        self.trades: List[Trade] = []
        self.equity_history = []
        self.position_history = []

    def run_backtest(
            self,
            feature_data: Dict[str, pd.DataFrame],
            price_data: Dict[str, pd.DataFrame],
            model_trainers: List[Any],
            calibrator: Any,
            scalers: Dict[str, Any],
            train_end_date: datetime,
            test_start_date: datetime,
            test_end_date: datetime,
            prediction_horizon: int = 21
    ) -> BacktestResults:
        """
        Run comprehensive backtest with proper temporal separation.

        Args:
            feature_data: Features for each symbol
            price_data: Price data for each symbol
            model_trainers: Trained model ensemble
            calibrator: Confidence calibrator
            scalers: Feature scalers
            train_end_date: Last date of training data
            test_start_date: First date of test data (must be > train_end_date)
            test_end_date: Last date of test data
            prediction_horizon: Days ahead for predictions

        Returns:
            BacktestResults object with all metrics
        """

        # Validate temporal separation
        if test_start_date <= train_end_date:
            raise ValueError(f"Test start date {test_start_date} must be after train end date {train_end_date}")

        temporal_gap = (test_start_date - train_end_date).days
        logger.info(f"Temporal gap between train and test: {temporal_gap} days")

        # Get all trading dates in test period
        all_dates = set()
        for symbol, df in price_data.items():
            mask = (df.index >= test_start_date) & (df.index <= test_end_date)
            all_dates.update(df[mask].index.tolist())

        trading_dates = sorted(list(all_dates))
        logger.info(f"Backtesting from {trading_dates[0]} to {trading_dates[-1]} ({len(trading_dates)} days)")

        # Main backtest loop
        for current_date in trading_dates:
            # Update existing positions
            self._update_positions(current_date, price_data)

            # Check for exits
            self._check_exits(current_date, price_data)

            # Generate predictions for this date (using only past data)
            predictions = self._generate_predictions(
                current_date,
                feature_data,
                model_trainers,
                calibrator,
                scalers,
                prediction_horizon
            )

            # Execute new trades
            self._execute_trades(current_date, predictions, price_data)

            # Record equity
            self._record_equity(current_date, price_data)

        # Close all remaining positions at end
        self._close_all_positions(trading_dates[-1], price_data)

        # Calculate metrics
        results = self._calculate_metrics(trading_dates)

        return results

    def _update_positions(self, current_date: datetime, price_data: Dict[str, pd.DataFrame]):
        """Update current position prices."""
        for symbol, trade in self.positions.items():
            if symbol in price_data and current_date in price_data[symbol].index:
                current_price = price_data[symbol].loc[current_date, 'close']
                # Update position value
                trade.position_value = trade.shares * current_price

    def _check_exits(self, current_date: datetime, price_data: Dict[str, pd.DataFrame]):
        """Check and execute exit conditions."""
        positions_to_close = []

        for symbol, trade in self.positions.items():
            if symbol not in price_data or current_date not in price_data[symbol].index:
                continue

            current_price = price_data[symbol].loc[current_date, 'close']

            # Check stop loss
            if current_price <= trade.entry_price * (1 - self.stop_loss):
                positions_to_close.append((symbol, current_price, 'stop_loss'))

            # Check take profit
            elif current_price >= trade.entry_price * (1 + self.take_profit):
                positions_to_close.append((symbol, current_price, 'take_profit'))

            # Check time exit (holding period)
            elif (current_date - trade.entry_date).days >= PREDICTION_CONFIG['horizon_days']:
                positions_to_close.append((symbol, current_price, 'time_exit'))

        # Execute closes
        for symbol, exit_price, reason in positions_to_close:
            self._close_position(symbol, current_date, exit_price, reason)

    def _generate_predictions(
            self,
            current_date: datetime,
            feature_data: Dict[str, pd.DataFrame],
            model_trainers: List[Any],
            calibrator: Any,
            scalers: Dict[str, Any],
            prediction_horizon: int
    ) -> pd.DataFrame:
        """Generate predictions for current date using only historical data."""
        predictions_list = []

        for symbol, features in feature_data.items():
            # Only use data up to current date (no look-ahead)
            historical_features = features[features.index <= current_date]

            if len(historical_features) < 252:  # Need at least 1 year of history
                continue

            # Get features for current date
            if current_date not in historical_features.index:
                continue

            # Get feature names (exclude price columns)
            feature_cols = [col for col in features.columns
                            if col not in ['open', 'high', 'low', 'close', 'volume']]

            # Extract features
            current_features = historical_features.loc[current_date, feature_cols].values.reshape(1, -1)

            # Scale if scaler available
            if symbol in scalers:
                current_features = scalers[symbol].transform(current_features)

            # Make prediction
            pred, conf = make_ensemble_predictions(model_trainers, current_features, calibrator)

            predictions_list.append({
                'symbol': symbol,
                'predicted_return': pred[0],
                'confidence': conf[0],
                'prediction_date': current_date
            })

        return pd.DataFrame(predictions_list)

    def _execute_trades(
            self,
            current_date: datetime,
            predictions: pd.DataFrame,
            price_data: Dict[str, pd.DataFrame]
    ):
        """Execute trades based on predictions."""
        if predictions.empty:
            return

        # Filter for high-quality signals
        signals = predictions[
            (predictions['confidence'] >= self.confidence_threshold) &
            (predictions['predicted_return'] >= self.min_predicted_return)
            ].sort_values('predicted_return', ascending=False)

        # Execute trades up to position limit
        for _, signal in signals.iterrows():
            if len(self.positions) >= self.max_positions:
                break

            symbol = signal['symbol']

            # Skip if already have position
            if symbol in self.positions:
                continue

            # Skip if no price data
            if symbol not in price_data or current_date not in price_data[symbol].index:
                continue

            # Calculate position size
            portfolio_value = self._get_portfolio_value(price_data, current_date)
            position_value = portfolio_value * self.position_size

            # Get entry price with slippage
            entry_price = price_data[symbol].loc[current_date, 'close'] * (1 + self.slippage)

            # Calculate shares (whole shares only)
            shares = int(position_value / entry_price)
            if shares == 0:
                continue

            # Calculate actual position value and commission
            actual_position_value = shares * entry_price
            commission_cost = actual_position_value * self.commission
            total_cost = actual_position_value + commission_cost

            # Check if have enough cash
            if total_cost > self.cash:
                continue

            # Execute trade
            trade = Trade(
                symbol=symbol,
                entry_date=current_date,
                entry_price=entry_price,
                shares=shares,
                position_value=actual_position_value,
                predicted_return=signal['predicted_return'],
                confidence=signal['confidence']
            )

            self.positions[symbol] = trade
            self.trades.append(trade)
            self.cash -= total_cost

            logger.debug(f"BUY {shares} {symbol} @ ${entry_price:.2f} on {current_date}")

    def _close_position(
            self,
            symbol: str,
            exit_date: datetime,
            exit_price: float,
            reason: str
    ):
        """Close a position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        # Apply slippage to exit
        actual_exit_price = exit_price * (1 - self.slippage)

        # Calculate proceeds and commission
        gross_proceeds = trade.shares * actual_exit_price
        commission_cost = gross_proceeds * self.commission
        net_proceeds = gross_proceeds - commission_cost

        # Update trade
        trade.exit_date = exit_date
        trade.exit_price = actual_exit_price
        trade.exit_reason = reason

        # Update cash
        self.cash += net_proceeds

        # Remove from positions
        del self.positions[symbol]

        logger.debug(f"SELL {trade.shares} {symbol} @ ${actual_exit_price:.2f} on {exit_date} "
                     f"(P&L: ${trade.pnl:.2f}, {trade.return_pct:.2%})")

    def _close_all_positions(self, exit_date: datetime, price_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of backtest."""
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in price_data and exit_date in price_data[symbol].index:
                exit_price = price_data[symbol].loc[exit_date, 'close']
                self._close_position(symbol, exit_date, exit_price, 'backtest_end')

    def _get_portfolio_value(self, price_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total portfolio value."""
        positions_value = 0.0

        for symbol, trade in self.positions.items():
            if symbol in price_data and date in price_data[symbol].index:
                current_price = price_data[symbol].loc[date, 'close']
                positions_value += trade.shares * current_price

        return self.cash + positions_value

    def _record_equity(self, date: datetime, price_data: Dict[str, pd.DataFrame]):
        """Record equity and positions for this date."""
        portfolio_value = self._get_portfolio_value(price_data, date)

        self.equity_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions)
        })

        # Record position details
        position_snapshot = {
            'date': date,
            'positions': {}
        }

        for symbol, trade in self.positions.items():
            if symbol in price_data and date in price_data[symbol].index:
                current_price = price_data[symbol].loc[date, 'close']
                position_snapshot['positions'][symbol] = {
                    'shares': trade.shares,
                    'entry_price': trade.entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': (current_price - trade.entry_price) * trade.shares,
                    'unrealized_return': (current_price - trade.entry_price) / trade.entry_price
                }

        self.position_history.append(position_snapshot)

    def _calculate_metrics(self, trading_dates: List[datetime]) -> BacktestResults:
        """Calculate comprehensive performance metrics."""
        results = BacktestResults()

        # Store trades
        results.trades = self.trades

        # Create equity curve
        equity_df = pd.DataFrame(self.equity_history)
        if not equity_df.empty:
            equity_df.set_index('date', inplace=True)
            results.equity_curve = equity_df['portfolio_value']

            # Calculate daily returns
            results.daily_returns = results.equity_curve.pct_change().dropna()

            # Performance metrics
            results.total_return = (results.equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

            # Annualized return
            days = len(trading_dates)
            years = days / 252
            if years > 0:
                results.annualized_return = (1 + results.total_return) ** (1 / years) - 1

            # Volatility
            results.volatility = results.daily_returns.std() * np.sqrt(252)

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = results.daily_returns - risk_free_rate / 252
            if results.daily_returns.std() > 0:
                results.sharpe_ratio = np.sqrt(252) * excess_returns.mean() / results.daily_returns.std()

            # Sortino ratio
            downside_returns = results.daily_returns[results.daily_returns < 0]
            if len(downside_returns) > 0:
                results.downside_deviation = downside_returns.std() * np.sqrt(252)
                if results.downside_deviation > 0:
                    results.sortino_ratio = (results.annualized_return - risk_free_rate) / results.downside_deviation

            # Maximum drawdown
            cumulative = (1 + results.daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            results.max_drawdown = drawdown.min()

            # Drawdown duration
            drawdown_start = None
            max_duration = 0
            current_duration = 0

            for date, dd in drawdown.items():
                if dd < 0:
                    if drawdown_start is None:
                        drawdown_start = date
                    current_duration = (date - drawdown_start).days
                    max_duration = max(max_duration, current_duration)
                else:
                    drawdown_start = None
                    current_duration = 0

            results.max_drawdown_duration = max_duration

            # Calmar ratio
            if results.max_drawdown != 0:
                results.calmar_ratio = results.annualized_return / abs(results.max_drawdown)

            # VaR and CVaR
            results.var_95 = results.daily_returns.quantile(0.05)
            results.cvar_95 = results.daily_returns[results.daily_returns <= results.var_95].mean()

        # Trade statistics
        closed_trades = [t for t in self.trades if not t.is_open]
        results.total_trades = len(closed_trades)

        if closed_trades:
            # Win/loss statistics
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]

            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0

            # Average win/loss
            if winning_trades:
                results.avg_win = np.mean([t.pnl for t in winning_trades])
            if losing_trades:
                results.avg_loss = np.mean([t.pnl for t in losing_trades])

            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
            gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
            results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Expectancy
            results.expectancy = np.mean([t.pnl for t in closed_trades])

            # Average holding period
            holding_days = [t.holding_days for t in closed_trades]
            results.avg_holding_days = np.mean(holding_days) if holding_days else 0

            # Turnover rate (annualized)
            total_traded_value = sum(t.position_value for t in closed_trades)
            avg_portfolio_value = results.equity_curve.mean() if not results.equity_curve.empty else self.initial_capital
            if avg_portfolio_value > 0 and years > 0:
                results.turnover_rate = (total_traded_value / avg_portfolio_value) / years

        return results

    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """Plot comprehensive backtest results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Equity curve
        ax1 = axes[0, 0]
        results.equity_curve.plot(ax=ax1, color='blue', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = axes[0, 1]
        cumulative = (1 + results.daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        drawdown.plot(ax=ax2, color='red', linewidth=2)
        ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Returns distribution
        ax3 = axes[1, 0]
        results.daily_returns.hist(bins=50, ax=ax3, alpha=0.7, color='green')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.axvline(x=results.daily_returns.mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
        ax3.set_title('Daily Returns Distribution')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Monthly returns heatmap
        ax4 = axes[1, 1]
        if not results.daily_returns.empty:
            monthly_returns = results.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_pivot = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values * 100
            })
            monthly_returns_pivot = monthly_returns_pivot.pivot(index='Month', columns='Year', values='Return')

            sns.heatmap(monthly_returns_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('Monthly Returns Heatmap (%)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_report(self, results: BacktestResults) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("BACKTEST RESULTS REPORT")
        report.append("=" * 80)
        report.append("")

        # Configuration
        report.append("CONFIGURATION:")
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Position Size: {self.position_size:.1%}")
        report.append(f"Max Positions: {self.max_positions}")
        report.append(f"Stop Loss: {self.stop_loss:.1%}")
        report.append(f"Take Profit: {self.take_profit:.1%}")
        report.append(f"Commission: {self.commission:.2%}")
        report.append(f"Slippage: {self.slippage:.2%}")
        report.append("")

        # Performance Summary
        report.append("PERFORMANCE SUMMARY:")
        report.append(f"Total Return: {results.total_return:.2%}")
        report.append(f"Annualized Return: {results.annualized_return:.2%}")
        report.append(f"Volatility: {results.volatility:.2%}")
        report.append(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {results.sortino_ratio:.2f}")
        report.append(f"Max Drawdown: {results.max_drawdown:.2%}")
        report.append(f"Max Drawdown Duration: {results.max_drawdown_duration} days")
        report.append(f"Calmar Ratio: {results.calmar_ratio:.2f}")
        report.append("")

        # Trade Statistics
        report.append("TRADE STATISTICS:")
        report.append(f"Total Trades: {results.total_trades}")
        report.append(f"Winning Trades: {results.winning_trades}")
        report.append(f"Losing Trades: {results.losing_trades}")
        report.append(f"Win Rate: {results.win_rate:.1%}")
        report.append(f"Average Win: ${results.avg_win:,.2f}")
        report.append(f"Average Loss: ${results.avg_loss:,.2f}")
        report.append(f"Profit Factor: {results.profit_factor:.2f}")
        report.append(f"Expectancy: ${results.expectancy:,.2f}")
        report.append(f"Average Holding Days: {results.avg_holding_days:.1f}")
        report.append("")

        # Risk Metrics
        report.append("RISK METRICS:")
        report.append(f"Downside Deviation: {results.downside_deviation:.2%}")
        report.append(f"Value at Risk (95%): {results.var_95:.2%}")
        report.append(f"Conditional VaR (95%): {results.cvar_95:.2%}")
        report.append(f"Turnover Rate: {results.turnover_rate:.1f}x per year")
        report.append("")

        # Best and Worst Trades
        if results.trades:
            closed_trades = [t for t in results.trades if not t.is_open]
            if closed_trades:
                sorted_trades = sorted(closed_trades, key=lambda x: x.pnl, reverse=True)

                report.append("BEST TRADES:")
                for i, trade in enumerate(sorted_trades[:5]):
                    report.append(
                        f"{i + 1}. {trade.symbol}: ${trade.pnl:,.2f} ({trade.return_pct:.2%}) in {trade.holding_days} days")

                report.append("")
                report.append("WORST TRADES:")
                for i, trade in enumerate(sorted_trades[-5:]):
                    report.append(
                        f"{i + 1}. {trade.symbol}: ${trade.pnl:,.2f} ({trade.return_pct:.2%}) in {trade.holding_days} days")

        return "\n".join(report)


def run_ml_backtest(
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        train_test_split: float = 0.7,
        temporal_gap_days: int = 7,
        **kwargs
) -> BacktestResults:
    """
    Run a complete ML-based backtest.

    Args:
        symbols: List of symbols to trade
        start_date: Start date of backtest period
        end_date: End date of backtest period
        train_test_split: Fraction of data for training (0.7 = 70%)
        temporal_gap_days: Days between train and test periods
        **kwargs: Additional arguments for backtester

    Returns:
        BacktestResults object
    """
    from data.collector import DataCollector
    from features.base_features import engineer_base_features
    from features.interactions import engineer_interaction_features
    from models.neural_networks import create_ensemble_models, train_ensemble

    logger.info(f"Starting ML backtest for {len(symbols)} symbols from {start_date} to {end_date}")

    # Step 1: Collect data
    logger.info("Collecting historical data...")
    collector = DataCollector()
    raw_data = collector.fetch_historical_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date
    )

    # Validate data
    clean_data = collector.validate_data(raw_data)

    # Calculate returns
    for symbol in clean_data:
        clean_data[symbol] = collector.calculate_returns(clean_data[symbol])

    # Add market data
    price_data = clean_data.copy()  # Keep price data separate
    clean_data = collector.add_market_data(clean_data)

    # Step 2: Engineer features
    logger.info("Engineering features...")
    base_features = engineer_base_features(clean_data, save_to_db=False)
    feature_data = engineer_interaction_features(base_features)

    # Step 3: Prepare data with temporal split
    logger.info("Preparing train/test split...")
    pipeline = FeatureEngineeringPipeline(symbols)

    # Calculate split date
    total_days = (end_date - start_date).days
    train_days = int(total_days * train_test_split)
    train_end_date = start_date + timedelta(days=train_days)
    test_start_date = train_end_date + timedelta(days=temporal_gap_days)

    logger.info(f"Training period: {start_date} to {train_end_date}")
    logger.info(f"Testing period: {test_start_date} to {end_date}")

    # Prepare training data
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        feature_data,
        prediction_horizon=21,
        train_end_date=train_end_date,
        temporal_gap=temporal_gap_days
    )

    # Step 4: Train models
    logger.info("Training ML models...")
    input_dim = X_train.shape[1]
    models = create_ensemble_models(
        input_dim=input_dim,
        num_models=5
    )

    trainers, calibrator = train_ensemble(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        models,
        epochs=50  # Fewer epochs for backtesting
    )

    # Step 5: Run backtest
    logger.info("Running backtest simulation...")
    backtester = ComprehensiveBacktester(**kwargs)

    results = backtester.run_backtest(
        feature_data=feature_data,
        price_data=price_data,
        model_trainers=trainers,
        calibrator=calibrator,
        scalers=pipeline.scalers,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        test_end_date=end_date,
        prediction_horizon=21
    )

    # Step 6: Generate report
    report = backtester.generate_report(results)
    logger.info("\n" + report)

    # Step 7: Save results
    results_dict = results.to_dict()
    results_dict['start_date'] = start_date.isoformat()
    results_dict['end_date'] = end_date.isoformat()
    results_dict['train_end_date'] = train_end_date.isoformat()
    results_dict['test_start_date'] = test_start_date.isoformat()

    with open('data/backtest_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("Backtest results saved to data/backtest_results.json")

    # Plot results
    backtester.plot_results(results, save_path='analysis/reports/backtest_results.png')

    return results


if __name__ == "__main__":
    # Example usage
    from config.settings import WATCHLIST

    # Run backtest on top 20 symbols
    results = run_ml_backtest(
        symbols=WATCHLIST[:20],
        start_date=datetime.now() - timedelta(days=730),  # 2 years
        end_date=datetime.now(),
        train_test_split=0.7,
        temporal_gap_days=7,
        initial_capital=100000,
        position_size=0.02,
        max_positions=10
    )

    print(f"\nBacktest Complete!")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.1%}")
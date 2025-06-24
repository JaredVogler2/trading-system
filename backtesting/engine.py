# backtesting/engine.py
"""
Backtesting engine with proper temporal separation and no data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import BACKTEST_CONFIG, PREDICTION_CONFIG
from data.database import DatabaseManager
from models.neural_networks import make_ensemble_predictions

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    entry_date: datetime
    entry_price: float
    shares: int
    position_size: float
    predicted_return: float
    confidence: float
    stop_loss: float
    take_profit: float

    def current_value(self, current_price: float) -> float:
        """Calculate current position value."""
        return self.shares * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        return (current_price - self.entry_price) * self.shares

    def return_pct(self, current_price: float) -> float:
        """Calculate return percentage."""
        return (current_price - self.entry_price) / self.entry_price


class BacktestEngine:
    """
    Backtesting engine with proper temporal separation.

    Key features:
    - No look-ahead bias
    - Proper train/test separation
    - Realistic execution simulation
    - Transaction costs and slippage
    """

    def __init__(
            self,
            initial_capital: float = 100000,
            position_size: float = 0.02,
            max_positions: int = 10,
            stop_loss: float = 0.05,
            take_profit: float = 0.15,
            commission: float = 0.001,
            slippage: float = 0.001,
            temporal_gap: int = 7
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital
            position_size: Fraction of capital per position
            max_positions: Maximum concurrent positions
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            commission: Trading commission percentage
            slippage: Slippage percentage
            temporal_gap: Days between train and test data
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        self.slippage = slippage
        self.temporal_gap = temporal_gap

        # Portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []

        # Database connection
        self.db = DatabaseManager()

    def run_backtest(
            self,
            feature_data: Dict[str, pd.DataFrame],
            trainers: List[Any],
            calibrator: Any,
            scalers: Dict[str, Any],
            train_end_date: datetime,
            test_start_date: datetime,
            test_end_date: datetime,
            prediction_horizon: int = 21
    ) -> Dict[str, Any]:
        """
        Run backtest with proper temporal separation.

        Args:
            feature_data: Dictionary of feature DataFrames
            trainers: Trained model trainers
            calibrator: Confidence calibrator
            scalers: Feature scalers
            train_end_date: Last date of training data
            test_start_date: First date of test data (must be > train_end_date + temporal_gap)
            test_end_date: Last date of test data
            prediction_horizon: Days ahead for predictions

        Returns:
            Backtest results and metrics
        """
        # Validate temporal separation
        if test_start_date <= train_end_date + timedelta(days=self.temporal_gap):
            raise ValueError(
                f"Test start date must be at least {self.temporal_gap} days after train end date. "
                f"Train ends: {train_end_date}, Test starts: {test_start_date}"
            )

        logger.info(f"Starting backtest from {test_start_date} to {test_end_date}")
        logger.info(f"Temporal gap: {(test_start_date - train_end_date).days} days")

        # Get all trading dates in test period
        all_dates = []
        for symbol, df in feature_data.items():
            mask = (df.index >= test_start_date) & (df.index <= test_end_date)
            all_dates.extend(df[mask].index.tolist())

        trading_dates = sorted(list(set(all_dates)))

        # Simulate trading for each date
        for current_date in trading_dates:
            # Update existing positions
            self._update_positions(feature_data, current_date)

            # Generate predictions for this date
            predictions = self._generate_predictions(
                feature_data,
                trainers,
                calibrator,
                scalers,
                current_date,
                prediction_horizon
            )

            # Execute trades based on predictions
            self._execute_trades(predictions, feature_data, current_date)

            # Record portfolio state
            self._record_portfolio_state(feature_data, current_date)

        # Close all remaining positions at end
        self._close_all_positions(feature_data, test_end_date)

        # Calculate performance metrics
        results = self._calculate_metrics()

        return results

    def _update_positions(
            self,
            feature_data: Dict[str, pd.DataFrame],
            current_date: datetime
    ) -> None:
        """Update existing positions and check for exits."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            if symbol not in feature_data:
                continue

            # Get current price
            df = feature_data[symbol]
            if current_date not in df.index:
                continue

            current_price = df.loc[current_date, 'close']

            # Check stop loss
            if current_price <= position.stop_loss:
                positions_to_close.append((symbol, current_price, 'stop_loss'))

            # Check take profit
            elif current_price >= position.take_profit:
                positions_to_close.append((symbol, current_price, 'take_profit'))

            # Check if reached prediction horizon
            elif current_date >= position.entry_date + timedelta(days=PREDICTION_CONFIG['horizon_days']):
                positions_to_close.append((symbol, current_price, 'horizon_reached'))

        # Close positions
        for symbol, exit_price, exit_reason in positions_to_close:
            self._close_position(symbol, exit_price, current_date, exit_reason)

    def _generate_predictions(
            self,
            feature_data: Dict[str, pd.DataFrame],
            trainers: List[Any],
            calibrator: Any,
            scalers: Dict[str, Any],
            current_date: datetime,
            prediction_horizon: int
    ) -> pd.DataFrame:
        """
        Generate predictions for current date without look-ahead bias.

        CRITICAL: Only uses data available up to current_date.
        """
        predictions_list = []

        for symbol, df in feature_data.items():
            # Only use data up to current date (no look-ahead)
            historical_data = df[df.index <= current_date]

            if len(historical_data) < 252:  # Need at least 1 year of history
                continue

            # Get features for current date
            if current_date not in historical_data.index:
                continue

            # Extract features (using only historical data for calculations)
            feature_names = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            features = historical_data.loc[current_date, feature_names].values.reshape(1, -1)

            # Scale features
            if symbol in scalers:
                features = scalers[symbol].transform(features)

            # Make prediction
            pred, conf = make_ensemble_predictions(trainers, features, calibrator)

            predictions_list.append({
                'symbol': symbol,
                'prediction_date': current_date,
                'predicted_return': pred[0],
                'confidence': conf[0],
                'current_price': historical_data.loc[current_date, 'close']
            })

        return pd.DataFrame(predictions_list)

    def _execute_trades(
            self,
            predictions: pd.DataFrame,
            feature_data: Dict[str, pd.DataFrame],
            current_date: datetime
    ) -> None:
        """Execute trades based on predictions."""
        if predictions.empty:
            return

        # Filter predictions by confidence and return threshold
        viable_trades = predictions[
            (predictions['confidence'] >= PREDICTION_CONFIG['confidence_threshold']) &
            (predictions['predicted_return'] >= PREDICTION_CONFIG['min_predicted_return'])
            ].sort_values('predicted_return', ascending=False)

        # Execute trades up to position limit
        for _, pred in viable_trades.iterrows():
            if len(self.positions) >= self.max_positions:
                break

            if pred['symbol'] in self.positions:
                continue  # Already have position

            # Calculate position size
            position_value = self.cash * self.position_size

            if position_value < 100:  # Minimum position size
                continue

            # Calculate execution price with slippage
            entry_price = pred['current_price'] * (1 + self.slippage)

            # Calculate shares (whole shares only)
            shares = int(position_value / entry_price)
            if shares == 0:
                continue

            # Calculate actual position value and commission
            actual_position_value = shares * entry_price
            commission_cost = actual_position_value * self.commission
            total_cost = actual_position_value + commission_cost

            if total_cost > self.cash:
                continue  # Not enough cash

            # Create position
            position = Position(
                symbol=pred['symbol'],
                entry_date=current_date,
                entry_price=entry_price,
                shares=shares,
                position_size=actual_position_value,
                predicted_return=pred['predicted_return'],
                confidence=pred['confidence'],
                stop_loss=entry_price * (1 - self.stop_loss),
                take_profit=entry_price * (1 + self.take_profit)
            )

            # Update portfolio
            self.positions[pred['symbol']] = position
            self.cash -= total_cost

            # Log trade
            self.db.save_trade(
                symbol=pred['symbol'],
                trade_type='BUY',
                quantity=shares,
                price=entry_price,
                timestamp=current_date,
                metadata={
                    'predicted_return': pred['predicted_return'],
                    'confidence': pred['confidence'],
                    'commission': commission_cost
                }
            )

            logger.info(f"BUY {shares} shares of {pred['symbol']} at ${entry_price:.2f}")

    def _close_position(
            self,
            symbol: str,
            exit_price: float,
            exit_date: datetime,
            exit_reason: str
    ) -> None:
        """Close a position and record the trade."""
        position = self.positions[symbol]

        # Calculate exit price with slippage
        actual_exit_price = exit_price * (1 - self.slippage)

        # Calculate P&L
        gross_pnl = (actual_exit_price - position.entry_price) * position.shares
        exit_commission = actual_exit_price * position.shares * self.commission
        net_pnl = gross_pnl - exit_commission

        # Record closed trade
        trade_record = {
            'symbol': symbol,
            'entry_date': position.entry_date,
            'exit_date': exit_date,
            'entry_price': position.entry_price,
            'exit_price': actual_exit_price,
            'shares': position.shares,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_pct': (actual_exit_price - position.entry_price) / position.entry_price,
            'predicted_return': position.predicted_return,
            'confidence': position.confidence,
            'exit_reason': exit_reason,
            'holding_days': (exit_date - position.entry_date).days
        }

        self.closed_trades.append(trade_record)

        # Update cash
        self.cash += actual_exit_price * position.shares - exit_commission

        # Remove position
        del self.positions[symbol]

        # Log trade
        self.db.save_trade(
            symbol=symbol,
            trade_type='SELL',
            quantity=position.shares,
            price=actual_exit_price,
            timestamp=exit_date,
            profit_loss=net_pnl,
            metadata={
                'exit_reason': exit_reason,
                'holding_days': trade_record['holding_days']
            }
        )

        logger.info(f"SELL {position.shares} shares of {symbol} at ${actual_exit_price:.2f} "
                    f"(P&L: ${net_pnl:.2f})")

    def _close_all_positions(
            self,
            feature_data: Dict[str, pd.DataFrame],
            end_date: datetime
    ) -> None:
        """Close all remaining positions at end of backtest."""
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in feature_data and end_date in feature_data[symbol].index:
                exit_price = feature_data[symbol].loc[end_date, 'close']
                self._close_position(symbol, exit_price, end_date, 'backtest_end')

    def _record_portfolio_state(
            self,
            feature_data: Dict[str, pd.DataFrame],
            current_date: datetime
    ) -> None:
        """Record current portfolio state for analysis."""
        # Calculate portfolio value
        positions_value = 0
        for symbol, position in self.positions.items():
            if symbol in feature_data and current_date in feature_data[symbol].index:
                current_price = feature_data[symbol].loc[current_date, 'close']
                positions_value += position.shares * current_price

        total_value = self.cash + positions_value

        self.portfolio_history.append({
            'date': current_date,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'num_positions': len(self.positions),
            'return_pct': (total_value - self.initial_capital) / self.initial_capital
        })

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest metrics."""
        if not self.portfolio_history:
            return {'error': 'No portfolio history'}

        # Convert to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()

        # Total return
        total_return = (portfolio_df['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital

        # Annualized return
        num_days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / num_days) - 1

        # Sharpe ratio
        daily_rf_rate = 0.02 / 252  # 2% annual risk-free rate
        excess_returns = portfolio_df['daily_return'] - daily_rf_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Maximum drawdown
        cumulative_returns = (1 + portfolio_df['daily_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade analysis
        if self.closed_trades:
            trades_df = pd.DataFrame(self.closed_trades)
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] <= 0]

            win_rate = len(winning_trades) / len(trades_df)
            avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum()) if len(
                losing_trades) > 0 else float('inf')

            # Prediction accuracy
            trades_df['prediction_accurate'] = (
                    (trades_df['predicted_return'] > 0) == (trades_df['return_pct'] > 0)
            )
            prediction_accuracy = trades_df['prediction_accurate'].mean()

            # Average holding period
            avg_holding_days = trades_df['holding_days'].mean()
        else:
            win_rate = avg_win = avg_loss = profit_factor = prediction_accuracy = avg_holding_days = 0

        # Compile results
        results = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.closed_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'prediction_accuracy': prediction_accuracy,
            'avg_holding_days': avg_holding_days,
            'final_capital': portfolio_df['total_value'].iloc[-1],
            'portfolio_history': portfolio_df,
            'trades': pd.DataFrame(self.closed_trades) if self.closed_trades else pd.DataFrame()
        }

        # Save results to database
        self.db.save_backtest_results(
            symbol='PORTFOLIO',
            strategy_name='ensemble_neural_network',
            start_date=portfolio_df.index[0],
            end_date=portfolio_df.index[-1],
            metrics=results
        )

        return results

    def plot_results(self, results: Dict[str, Any]) -> None:
        """Plot backtest results."""
        portfolio_df = results['portfolio_history']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True)

        # Cumulative returns
        ax2 = axes[0, 1]
        cumulative_returns = (1 + portfolio_df['daily_return']).cumprod() - 1
        ax2.plot(portfolio_df.index, cumulative_returns * 100)
        ax2.set_title('Cumulative Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True)

        # Drawdown
        ax3 = axes[1, 0]
        cumulative_returns_calc = (1 + portfolio_df['daily_return']).cumprod()
        running_max = cumulative_returns_calc.expanding().max()
        drawdown = (cumulative_returns_calc - running_max) / running_max
        ax3.fill_between(portfolio_df.index, drawdown * 100, 0, alpha=0.3, color='red')
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True)

        # Trade distribution
        ax4 = axes[1, 1]
        if 'trades' in results and not results['trades'].empty:
            trades_df = results['trades']
            ax4.hist(trades_df['return_pct'] * 100, bins=30, alpha=0.7, color='blue')
            ax4.axvline(x=0, color='r', linestyle='--')
            ax4.set_title('Trade Return Distribution')
            ax4.set_xlabel('Return (%)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n=== Backtest Results ===")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Prediction Accuracy: {results['prediction_accuracy']:.2%}")
        print(f"Avg Holding Days: {results['avg_holding_days']:.1f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")


def run_walk_forward_analysis(
        feature_data: Dict[str, pd.DataFrame],
        initial_train_months: int = 24,
        test_months: int = 3,
        retrain_frequency: int = 3,  # Retrain every 3 months
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Run walk-forward analysis to simulate real-world model updates.

    This prevents data leakage by:
    1. Training only on past data
    2. Testing on future unseen data
    3. Retraining periodically as new data becomes available

    Args:
        feature_data: Feature data for all symbols
        initial_train_months: Initial training period
        test_months: Testing period length
        retrain_frequency: How often to retrain (months)
        start_date: Start of analysis
        end_date: End of analysis

    Returns:
        Combined results from all test periods
    """
    from features.pipeline import FeatureEngineeringPipeline
    from models.neural_networks import create_ensemble_models, train_ensemble

    logger.info("Starting walk-forward analysis")

    # Determine date range
    if not start_date:
        start_date = min(df.index.min() for df in feature_data.values())
    if not end_date:
        end_date = max(df.index.max() for df in feature_data.values())

    # Initialize
    pipeline = FeatureEngineeringPipeline()
    all_results = []
    combined_trades = []

    # Start with initial training period
    current_train_start = start_date
    current_train_end = start_date + timedelta(days=initial_train_months * 30)

    period = 1
    while current_train_end < end_date:
        logger.info(f"\n=== Walk-Forward Period {period} ===")
        logger.info(f"Training: {current_train_start} to {current_train_end}")

        # Define test period
        test_start = current_train_end + timedelta(days=BACKTEST_CONFIG['temporal_gap_days'])
        test_end = min(test_start + timedelta(days=test_months * 30), end_date)

        logger.info(f"Testing: {test_start} to {test_end}")

        # Prepare training data (only using data up to train_end)
        train_features = {}
        for symbol, df in feature_data.items():
            train_mask = (df.index >= current_train_start) & (df.index <= current_train_end)
            if train_mask.sum() > 252:  # At least 1 year
                train_features[symbol] = df[train_mask]

        # Split into train/validation
        X_train, X_val, _, y_train, y_val, _ = pipeline.prepare_training_data(
            train_features,
            train_end_date=current_train_end - timedelta(days=60)  # Save 2 months for validation
        )

        # Create and train models
        input_dim = X_train.shape[1]
        models = create_ensemble_models(input_dim, num_models=5)

        trainers, calibrator = train_ensemble(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            models,
            epochs=50  # Fewer epochs for walk-forward
        )

        # Run backtest on test period
        engine = BacktestEngine()
        results = engine.run_backtest(
            feature_data,
            trainers,
            calibrator,
            pipeline.scalers,
            current_train_end,
            test_start,
            test_end
        )

        # Store results
        results['period'] = period
        results['train_start'] = current_train_start
        results['train_end'] = current_train_end
        results['test_start'] = test_start
        results['test_end'] = test_end

        all_results.append(results)
        if 'trades' in results and not results['trades'].empty:
            combined_trades.append(results['trades'])

        # Move to next period
        if period % retrain_frequency == 0:
            # Retrain with more recent data
            current_train_start = current_train_end - timedelta(days=initial_train_months * 30)

        current_train_end = test_end
        period += 1

        # Break if not enough data for next test period
        if test_end + timedelta(days=test_months * 30) > end_date:
            break

    # Combine all results
    combined_results = {
        'periods': all_results,
        'num_periods': len(all_results),
        'avg_return': np.mean([r['total_return'] for r in all_results]),
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_results]),
        'total_trades': sum(r['total_trades'] for r in all_results),
        'all_trades': pd.concat(combined_trades) if combined_trades else pd.DataFrame()
    }

    # Plot combined results
    _plot_walk_forward_results(combined_results)

    return combined_results


def _plot_walk_forward_results(results: Dict[str, Any]) -> None:
    """Plot walk-forward analysis results."""
    periods = results['periods']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Returns by period
    ax1 = axes[0, 0]
    returns = [p['total_return'] * 100 for p in periods]
    ax1.bar(range(1, len(returns) + 1), returns)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Returns by Test Period')
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Return (%)')
    ax1.grid(True)

    # Sharpe ratios
    ax2 = axes[0, 1]
    sharpes = [p['sharpe_ratio'] for p in periods]
    ax2.plot(range(1, len(sharpes) + 1), sharpes, marker='o')
    ax2.axhline(y=1.0, color='g', linestyle='--', label='Sharpe = 1.0')
    ax2.set_title('Sharpe Ratio by Period')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True)

    # Win rates
    ax3 = axes[1, 0]
    win_rates = [p['win_rate'] * 100 for p in periods]
    ax3.plot(range(1, len(win_rates) + 1), win_rates, marker='o', color='green')
    ax3.axhline(y=50, color='r', linestyle='--')
    ax3.set_title('Win Rate by Period')
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.grid(True)

    # Cumulative performance
    ax4 = axes[1, 1]
    cumulative_capital = [results['periods'][0]['initial_capital']]
    for p in periods:
        cumulative_capital.append(p['final_capital'])

    ax4.plot(range(len(cumulative_capital)), cumulative_capital, marker='o')
    ax4.set_title('Cumulative Capital Growth')
    ax4.set_xlabel('Period')
    ax4.set_ylabel('Capital ($)')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n=== Walk-Forward Analysis Summary ===")
    print(f"Number of periods: {results['num_periods']}")
    print(f"Average return per period: {results['avg_return']:.2%}")
    print(f"Average Sharpe ratio: {results['avg_sharpe']:.2f}")
    print(f"Total trades: {results['total_trades']}")

    # Period details
    print("\nPeriod Details:")
    for i, p in enumerate(results['periods'], 1):
        print(f"Period {i}: Return={p['total_return']:.2%}, "
              f"Sharpe={p['sharpe_ratio']:.2f}, "
              f"Trades={p['total_trades']}")


if __name__ == "__main__":
    # Example usage
    print("Backtesting Engine Example")

    # This would be called from main.py with real data
    # Here we show the structure

    engine = BacktestEngine(
        initial_capital=100000,
        position_size=0.02,
        max_positions=10,
        stop_loss=0.05,
        take_profit=0.15
    )

    print(f"Initialized backtest engine with:")
    print(f"  Initial capital: ${engine.initial_capital:,}")
    print(f"  Position size: {engine.position_size:.1%}")
    print(f"  Max positions: {engine.max_positions}")
    print(f"  Stop loss: {engine.stop_loss:.1%}")
    print(f"  Take profit: {engine.take_profit:.1%}")
    print(f"  Temporal gap: {engine.temporal_gap} days")
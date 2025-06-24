# trading/live_trader.py
"""
Real-time paper trading implementation with Alpaca integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Trade, Quote

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, PREDICTION_CONFIG
from main import TradingSystem

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """Real-time position tracking."""
    symbol: str
    entry_price: float
    shares: int
    entry_time: datetime
    predicted_return: float
    confidence: float
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    def update_price(self, price: float):
        """Update current price and P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        """Calculate current return percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class RiskManager:
    """Advanced risk management with Kelly Criterion and correlation limits."""

    def __init__(self,
                 max_position_size: float = 0.05,
                 max_portfolio_heat: float = 0.10,
                 max_correlation: float = 0.70,
                 kelly_fraction: float = 0.25):
        """
        Initialize risk manager.

        Args:
            max_position_size: Maximum size for any single position (5%)
            max_portfolio_heat: Maximum portfolio risk (10%)
            max_correlation: Maximum average correlation for new positions
            kelly_fraction: Fraction of Kelly Criterion to use (25%)
        """
        self.max_position_size = max_position_size
        self.max_portfolio_heat = max_portfolio_heat
        self.max_correlation = max_correlation
        self.kelly_fraction = kelly_fraction
        self.correlation_matrix = pd.DataFrame()
        self.recent_trades = []

    def calculate_kelly_position_size(self,
                                      predicted_return: float,
                                      confidence: float,
                                      historical_accuracy: float = 0.55) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            predicted_return: Expected return from model
            confidence: Model confidence (0-1)
            historical_accuracy: Historical win rate of similar predictions

        Returns:
            Optimal position size as fraction of capital
        """
        # Adjust win probability based on confidence
        win_probability = historical_accuracy * confidence

        # Expected win/loss amounts
        expected_win = abs(predicted_return)
        expected_loss = 0.05  # Assume 5% stop loss

        # Calculate odds
        if expected_loss == 0:
            return self.max_position_size * 0.5  # Default to half max

        odds = expected_win / expected_loss

        # Kelly formula: (p * odds - q) / odds
        # where p = win probability, q = loss probability
        q = 1 - win_probability
        kelly_pct = (win_probability * odds - q) / odds

        # Apply Kelly fraction for conservative sizing
        conservative_kelly = kelly_pct * self.kelly_fraction

        # Ensure within bounds
        return np.clip(conservative_kelly, 0.01, self.max_position_size)

    def check_correlation_limits(self,
                                 symbol: str,
                                 current_positions: Dict[str, LivePosition]) -> bool:
        """
        Check if adding new position would exceed correlation limits.

        Args:
            symbol: Symbol to check
            current_positions: Current portfolio positions

        Returns:
            True if position is acceptable, False otherwise
        """
        if not current_positions or self.correlation_matrix.empty:
            return True

        if symbol not in self.correlation_matrix.index:
            # If we don't have correlation data, be conservative
            return len(current_positions) < 5

        # Calculate average correlation with existing positions
        position_symbols = list(current_positions.keys())

        # Filter to symbols we have data for
        valid_symbols = [s for s in position_symbols if s in self.correlation_matrix.index]

        if not valid_symbols:
            return True

        correlations = self.correlation_matrix.loc[symbol, valid_symbols]
        avg_correlation = correlations.mean()
        max_correlation = correlations.max()

        # Reject if average correlation too high or any single correlation extreme
        return avg_correlation < self.max_correlation and max_correlation < 0.90

    def calculate_portfolio_heat(self, positions: Dict[str, LivePosition]) -> float:
        """
        Calculate current portfolio heat (risk exposure).

        Args:
            positions: Current positions

        Returns:
            Portfolio heat as fraction of capital at risk
        """
        if not positions:
            return 0.0

        total_risk = 0.0
        for position in positions.values():
            # Risk = position size * distance to stop loss
            position_value = position.shares * position.current_price
            stop_distance = (position.current_price - position.stop_loss) / position.current_price
            position_risk = position_value * stop_distance
            total_risk += position_risk

        # This should be divided by total capital
        # For now, return a normalized estimate
        return min(total_risk / (sum(p.shares * p.current_price for p in positions.values()) + 1), 1.0)

    def update_correlation_matrix(self, returns_data: pd.DataFrame):
        """Update correlation matrix with latest returns data."""
        if len(returns_data) > 30:  # Need sufficient data
            self.correlation_matrix = returns_data.corr()

    def should_take_trade(self,
                          symbol: str,
                          predicted_return: float,
                          confidence: float,
                          current_positions: Dict[str, LivePosition],
                          portfolio_value: float) -> Tuple[bool, float, str]:
        """
        Comprehensive trade decision including position sizing.

        Returns:
            Tuple of (should_trade, position_size, reason)
        """
        # Check correlation limits
        if not self.check_correlation_limits(symbol, current_positions):
            return False, 0.0, "Exceeds correlation limits"

        # Check portfolio heat
        current_heat = self.calculate_portfolio_heat(current_positions)
        if current_heat > self.max_portfolio_heat:
            return False, 0.0, f"Portfolio heat too high: {current_heat:.1%}"

        # Calculate Kelly position size
        position_size = self.calculate_kelly_position_size(
            predicted_return,
            confidence
        )

        # Additional safety checks
        if position_size < 0.01:
            return False, 0.0, "Position size too small"

        if len(current_positions) >= 20:
            return False, 0.0, "Maximum positions reached"

        return True, position_size, "Trade approved"


class LiveTradingSystem:
    """Main live trading system with real-time execution."""

    def __init__(self,
                 trading_system: TradingSystem,
                 paper: bool = True):
        """
        Initialize live trading system.

        Args:
            trading_system: Trained trading system instance
            paper: Use paper trading (True) or live trading (False)
        """
        self.system = trading_system
        self.paper = paper

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper=paper
        )

        self.data_stream = StockDataStream(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY
        )

        # Risk manager
        self.risk_manager = RiskManager()

        # Position tracking
        self.positions: Dict[str, LivePosition] = {}
        self.pending_orders = {}

        # Performance tracking
        self.trades_today = []
        self.daily_pnl = 0.0

        # Configuration
        self.min_confidence = PREDICTION_CONFIG.get('confidence_threshold', 0.7)
        self.min_return = PREDICTION_CONFIG.get('min_predicted_return', 0.02)
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15

        logger.info(f"Live trading system initialized ({'Paper' if paper else 'Live'} mode)")

    async def start_trading(self):
        """Start the live trading system."""
        logger.info("Starting live trading system...")

        # Subscribe to market data for current positions
        await self._subscribe_to_market_data()

        # Start main trading loop
        await asyncio.gather(
            self._trading_loop(),
            self._position_monitor_loop(),
            self._risk_monitor_loop()
        )

    async def _trading_loop(self):
        """Main trading loop - generates and executes trades."""
        while True:
            try:
                # Check if market is open
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed. Waiting...")
                    await asyncio.sleep(60)
                    continue

                # Generate predictions
                logger.info("Generating predictions...")
                predictions = self.system.generate_predictions()

                # Filter for high-quality trades
                trades = predictions[
                    (predictions['confidence'] >= self.min_confidence) &
                    (predictions['predicted_return'] >= self.min_return)
                    ].sort_values('predicted_return', ascending=False)

                logger.info(f"Found {len(trades)} potential trades")

                # Get account info
                account = self.trading_client.get_account()
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)

                # Execute trades
                for _, trade in trades.iterrows():
                    if trade['symbol'] in self.positions:
                        continue  # Already have position

                    # Risk management checks
                    should_trade, position_size, reason = self.risk_manager.should_take_trade(
                        trade['symbol'],
                        trade['predicted_return'],
                        trade['confidence'],
                        self.positions,
                        portfolio_value
                    )

                    if not should_trade:
                        logger.info(f"Skipping {trade['symbol']}: {reason}")
                        continue

                    # Calculate order size
                    position_value = portfolio_value * position_size
                    shares = int(position_value / trade['current_price'])

                    if shares < 1:
                        continue

                    # Place order
                    await self._place_order(
                        symbol=trade['symbol'],
                        shares=shares,
                        predicted_return=trade['predicted_return'],
                        confidence=trade['confidence'],
                        current_price=trade['current_price']
                    )

                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)

    async def _position_monitor_loop(self):
        """Monitor existing positions for exit signals."""
        while True:
            try:
                for symbol, position in list(self.positions.items()):
                    # Check stop loss
                    if position.current_price <= position.stop_loss:
                        logger.info(f"Stop loss triggered for {symbol}")
                        await self._close_position(symbol, "stop_loss")

                    # Check take profit
                    elif position.current_price >= position.take_profit:
                        logger.info(f"Take profit triggered for {symbol}")
                        await self._close_position(symbol, "take_profit")

                    # Check time-based exit (if position held too long)
                    elif (datetime.now() - position.entry_time).days >= PREDICTION_CONFIG['horizon_days']:
                        logger.info(f"Time exit for {symbol}")
                        await self._close_position(symbol, "time_exit")

                    # Check for reversal signals
                    elif position.confidence < 0.5:  # Confidence has degraded
                        logger.info(f"Confidence degraded for {symbol}")
                        await self._close_position(symbol, "low_confidence")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(10)

    async def _risk_monitor_loop(self):
        """Monitor portfolio-level risk metrics."""
        while True:
            try:
                # Calculate portfolio heat
                portfolio_heat = self.risk_manager.calculate_portfolio_heat(self.positions)

                if portfolio_heat > self.risk_manager.max_portfolio_heat:
                    logger.warning(f"Portfolio heat critical: {portfolio_heat:.1%}")
                    # Could implement portfolio-wide stop or position reduction

                # Update daily P&L
                self._update_daily_pnl()

                # Check drawdown
                if self.daily_pnl < -0.02 * self._get_portfolio_value():  # 2% daily loss limit
                    logger.warning("Daily loss limit approaching!")
                    # Could halt new trades for the day

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                await asyncio.sleep(30)

    async def _place_order(self,
                           symbol: str,
                           shares: int,
                           predicted_return: float,
                           confidence: float,
                           current_price: float):
        """Place a market order with stops."""
        try:
            # Calculate stop loss and take profit levels
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)

            # Place market order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(f"Order placed: BUY {shares} {symbol} @ market")

            # Track position (will update with actual fill price)
            self.positions[symbol] = LivePosition(
                symbol=symbol,
                entry_price=current_price,  # Will update with actual fill
                shares=shares,
                entry_time=datetime.now(),
                predicted_return=predicted_return,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=current_price
            )

            # Subscribe to real-time data for this symbol
            await self._subscribe_symbol(symbol)

        except Exception as e:
            logger.error(f"Order placement error for {symbol}: {e}")

    async def _close_position(self, symbol: str, reason: str):
        """Close a position."""
        try:
            position = self.positions.get(symbol)
            if not position:
                return

            # Place market sell order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=position.shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            # Calculate P&L
            pnl = position.unrealized_pnl
            return_pct = position.return_pct

            logger.info(f"Position closed: SELL {position.shares} {symbol} "
                        f"| P&L: ${pnl:.2f} ({return_pct:.2%}) | Reason: {reason}")

            # Record trade
            self.trades_today.append({
                'symbol': symbol,
                'entry_time': position.entry_time,
                'exit_time': datetime.now(),
                'entry_price': position.entry_price,
                'exit_price': position.current_price,
                'shares': position.shares,
                'pnl': pnl,
                'return_pct': return_pct,
                'reason': reason,
                'predicted_return': position.predicted_return,
                'confidence': position.confidence
            })

            # Update daily P&L
            self.daily_pnl += pnl

            # Remove position
            del self.positions[symbol]

            # Unsubscribe from market data
            await self._unsubscribe_symbol(symbol)

        except Exception as e:
            logger.error(f"Position close error for {symbol}: {e}")

    async def _subscribe_to_market_data(self):
        """Subscribe to market data streams."""
        # Subscribe to bars for all positions
        symbols = list(self.positions.keys())
        if symbols:
            async def on_bar(bar: Bar):
                if bar.symbol in self.positions:
                    self.positions[bar.symbol].update_price(bar.close)

            self.data_stream.subscribe_bars(on_bar, *symbols)

    async def _subscribe_symbol(self, symbol: str):
        """Subscribe to market data for a symbol."""

        async def on_bar(bar: Bar):
            if bar.symbol in self.positions:
                self.positions[bar.symbol].update_price(bar.close)

        self.data_stream.subscribe_bars(on_bar, symbol)

    async def _unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from market data for a symbol."""
        self.data_stream.unsubscribe_bars(symbol)

    def _update_daily_pnl(self):
        """Update daily P&L from all positions."""
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized_pnl = sum(trade['pnl'] for trade in self.trades_today)
        self.daily_pnl = unrealized_pnl + realized_pnl

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        try:
            account = self.trading_client.get_account()
            return float(account.portfolio_value)
        except:
            return 100000.0  # Default

    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            'mode': 'Paper' if self.paper else 'Live',
            'positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'trades_today': len(self.trades_today),
            'portfolio_heat': self.risk_manager.calculate_portfolio_heat(self.positions),
            'active_symbols': list(self.positions.keys())
        }


# Standalone function to run live trading
async def run_live_trading(symbols: List[str] = None, paper: bool = True):
    """
    Run the live trading system.

    Args:
        symbols: List of symbols to trade
        paper: Use paper trading (True) or live trading (False)
    """
    # Initialize the main trading system
    system = TradingSystem(symbols=symbols)

    # Load models (assumes already trained)
    feature_data = system.collect_and_prepare_data()
    system._load_models(feature_data)

    # Create live trading system
    live_system = LiveTradingSystem(system, paper=paper)

    # Start trading
    await live_system.start_trading()


if __name__ == "__main__":
    # Run paper trading
    asyncio.run(run_live_trading(paper=True))
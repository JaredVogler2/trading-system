# monitoring/enhanced_dashboard.py
"""
Enhanced Trading Dashboard with Real Data Integration
No mock data - everything driven by actual portfolio, ML predictions, and market data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import logging
import yfinance as yf
from pathlib import Path
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import pytz

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, DATA_DIR, MODEL_DIR
from data.database import DatabaseManager
from backtesting.engine import BacktestEngine

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Trading System Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    /* Main theme */
    .main {
        padding: 0rem 1rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #444;
        height: 100%;
    }

    /* Status indicators */
    .status-online {
        color: #00ff88;
        font-weight: bold;
    }

    .status-offline {
        color: #ff4444;
        font-weight: bold;
    }

    .status-warning {
        color: #ffaa00;
        font-weight: bold;
    }

    /* Trading signals */
    .signal-buy {
        background-color: rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        padding: 10px;
        margin: 5px 0;
    }

    .signal-sell {
        background-color: rgba(255, 68, 68, 0.1);
        border-left: 4px solid #ff4444;
        padding: 10px;
        margin: 5px 0;
    }

    /* News sentiment */
    .sentiment-positive {
        color: #00ff88;
    }

    .sentiment-negative {
        color: #ff4444;
    }

    .sentiment-neutral {
        color: #ffaa00;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,123,255,0.1) 100%);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Connectivity status */
    .connection-status {
        position: fixed;
        top: 3.5rem;
        right: 1rem;
        z-index: 1000;
        background: rgba(0,0,0,0.8);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Main dashboard class with real data integration."""

    def __init__(self):
        """Initialize dashboard with API connections."""
        self.db = DatabaseManager()

        # Initialize API connections
        self.trading_client = None
        self.data_client = None
        self.alpaca_connected = False
        self.openai_connected = False

        # Try to connect to Alpaca
        try:
            self.trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            self.data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            # Test connection
            account = self.trading_client.get_account()
            self.alpaca_connected = True
            logger.info("Alpaca connection successful")
        except Exception as e:
            logger.error(f"Alpaca connection failed: {e}")
            self.alpaca_connected = False

        # Check OpenAI connection
        import os
        self.openai_connected = bool(os.getenv("OPENAI_API_KEY") and
                                     os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here")

        # Market hours tracking
        self.tz = pytz.timezone('US/Eastern')

    def run(self):
        """Run the main dashboard application."""
        # Header with navigation
        st.markdown("""
        <h1 style='text-align: center; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); 
        padding: 1.5rem; border-radius: 10px; color: white;'>
        🤖 AI Trading System Dashboard
        </h1>
        """, unsafe_allow_html=True)

        # Show connectivity status
        self._display_connectivity_status()

        # Sidebar navigation
        with st.sidebar:
            st.markdown("## 📊 Navigation")

            page = st.radio("Select Page", [
                "📈 Portfolio Overview",
                "💼 Positions & Trades",
                "🧠 ML Predictions",
                "📰 News Analysis",
                "📊 Backtesting Results",
                "📉 Performance Analytics",
                "❓ Help & Documentation"
            ])

            # Market status
            st.markdown("---")
            self._display_market_status()

            # Quick stats
            if self.alpaca_connected:
                st.markdown("---")
                self._display_quick_stats()

        # Main content area
        if page == "📈 Portfolio Overview":
            self._page_portfolio_overview()
        elif page == "💼 Positions & Trades":
            self._page_positions_trades()
        elif page == "🧠 ML Predictions":
            self._page_ml_predictions()
        elif page == "📰 News Analysis":
            self._page_news_analysis()
        elif page == "📊 Backtesting Results":
            self._page_backtesting()
        elif page == "📉 Performance Analytics":
            self._page_performance_analytics()
        elif page == "❓ Help & Documentation":
            self._page_help_documentation()

        # Auto-refresh
        if st.sidebar.checkbox("Auto-refresh (30s)", value=True):
            st.markdown(
                """<script>
                setTimeout(function() {
                    window.location.reload();
                }, 30000);
                </script>""",
                unsafe_allow_html=True
            )

    def _display_connectivity_status(self):
        """Display API connectivity status."""
        alpaca_status = "🟢 Connected" if self.alpaca_connected else "🔴 Disconnected"
        openai_status = "🟢 Connected" if self.openai_connected else "🔴 Not configured"

        st.markdown(
            f"""<div class='connection-status'>
            Alpaca: {alpaca_status} | OpenAI: {openai_status}
            </div>""",
            unsafe_allow_html=True
        )

    def _display_market_status(self):
        """Display current market status."""
        now = datetime.now(self.tz)
        market_open = self._is_market_open()

        if market_open:
            # Calculate time to close
            close_time = now.replace(hour=16, minute=0, second=0)
            time_to_close = close_time - now
            hours, remainder = divmod(time_to_close.seconds, 3600)
            minutes = remainder // 60

            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: rgba(0,255,136,0.1); border-radius: 8px;'>
            <h4 style='color: #00ff88; margin: 0;'>🟢 Market Open</h4>
            <p style='margin: 0.5rem 0 0 0;'>Closes in {hours}h {minutes}m</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Calculate next open
            next_open = self._get_next_market_open()
            time_to_open = next_open - now
            hours = time_to_open.total_seconds() // 3600

            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: rgba(255,68,68,0.1); border-radius: 8px;'>
            <h4 style='color: #ff4444; margin: 0;'>🔴 Market Closed</h4>
            <p style='margin: 0.5rem 0 0 0;'>Opens in {int(hours)}h</p>
            </div>
            """, unsafe_allow_html=True)

    def _display_quick_stats(self):
        """Display quick account statistics."""
        if not self.alpaca_connected:
            return

        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()

            st.metric("Portfolio Value", f"${float(account.portfolio_value):,.2f}")
            st.metric("Positions", len(positions))
            st.metric("Buying Power", f"${float(account.buying_power):,.2f}")

        except Exception as e:
            logger.error(f"Error getting quick stats: {e}")

    def _page_portfolio_overview(self):
        """Portfolio overview page with real data."""
        st.markdown('<div class="section-header"><h2>📈 Portfolio Overview</h2></div>',
                    unsafe_allow_html=True)

        if not self.alpaca_connected:
            st.error("❌ Alpaca connection required. Please check your API credentials.")
            return

        try:
            # Get account data
            account = self.trading_client.get_account()

            # Calculate daily P&L
            portfolio_value = float(account.portfolio_value)
            equity = float(account.equity)
            last_equity = float(account.last_equity) if hasattr(account, 'last_equity') else equity
            daily_pl = equity - last_equity
            daily_pl_pct = (daily_pl / last_equity * 100) if last_equity > 0 else 0

            # Display main metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Portfolio Value",
                    f"${portfolio_value:,.2f}",
                    f"${daily_pl:+,.2f} ({daily_pl_pct:+.2f}%)"
                )

            with col2:
                st.metric("Cash", f"${float(account.cash):,.2f}")

            with col3:
                st.metric("Buying Power", f"${float(account.buying_power):,.2f}")

            with col4:
                positions = self.trading_client.get_all_positions()
                total_pl = sum(float(p.unrealized_pl) for p in positions)
                st.metric("Unrealized P&L", f"${total_pl:+,.2f}")

            # Portfolio vs SPY comparison chart
            st.markdown("### Portfolio Performance vs SPY Benchmark")
            self._display_portfolio_vs_spy_chart()

            # Position distribution
            if positions:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Position Distribution")
                    self._display_position_distribution(positions)

                with col2:
                    st.markdown("### Position Performance")
                    self._display_position_performance(positions)

            # Recent activity summary
            st.markdown("### Recent Trading Activity")
            self._display_recent_activity()

        except Exception as e:
            st.error(f"Error loading portfolio data: {e}")
            logger.error(f"Portfolio overview error: {e}")

    def _display_portfolio_vs_spy_chart(self):
        """Display portfolio performance vs SPY benchmark."""
        try:
            # Get portfolio history from database
            trades_df = self.db.get_recent_trades(days_back=365)

            # Get SPY data
            spy = yf.Ticker("SPY")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            spy_data = spy.history(start=start_date, end=end_date)

            if not spy_data.empty:
                # Calculate returns
                spy_data['daily_return'] = spy_data['Close'].pct_change()
                spy_data['cumulative_return'] = (1 + spy_data['daily_return']).cumprod() - 1

                # Create figure
                fig = go.Figure()

                # Add SPY line
                fig.add_trace(go.Scatter(
                    x=spy_data.index,
                    y=spy_data['cumulative_return'] * 100,
                    mode='lines',
                    name='SPY Benchmark',
                    line=dict(color='#ffaa00', width=2)
                ))

                # Add portfolio performance if we have trade history
                if not trades_df.empty:
                    # Calculate portfolio returns from trades
                    # This is simplified - in production you'd track actual portfolio value over time
                    portfolio_returns = self._calculate_portfolio_returns(trades_df)

                    if portfolio_returns is not None:
                        fig.add_trace(go.Scatter(
                            x=portfolio_returns.index,
                            y=portfolio_returns['cumulative_return'] * 100,
                            mode='lines',
                            name='Portfolio',
                            line=dict(color='#00ff88', width=3)
                        ))

                fig.update_layout(
                    title="Portfolio vs SPY Performance (1 Year)",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Performance metrics comparison
                if not trades_df.empty:
                    col1, col2, col3 = st.columns(3)

                    spy_annual_return = spy_data['cumulative_return'].iloc[-1]
                    spy_volatility = spy_data['daily_return'].std() * np.sqrt(252)
                    spy_sharpe = spy_annual_return / spy_volatility if spy_volatility > 0 else 0

                    with col1:
                        st.metric("SPY Annual Return", f"{spy_annual_return:.2%}")
                    with col2:
                        st.metric("SPY Volatility", f"{spy_volatility:.2%}")
                    with col3:
                        st.metric("SPY Sharpe Ratio", f"{spy_sharpe:.2f}")

        except Exception as e:
            st.error(f"Error creating performance chart: {e}")
            logger.error(f"Performance chart error: {e}")

    def _display_position_distribution(self, positions):
        """Display position distribution pie chart."""
        if not positions:
            st.info("No active positions")
            return

        # Prepare data
        symbols = []
        values = []

        for pos in positions:
            symbols.append(pos.symbol)
            values.append(float(pos.market_value))

        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.3,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(t=30, b=30, l=30, r=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_position_performance(self, positions):
        """Display position performance bar chart."""
        if not positions:
            return

        # Prepare data
        data = []
        for pos in positions:
            unrealized_plpc = float(pos.unrealized_plpc) * 100
            data.append({
                'symbol': pos.symbol,
                'return': unrealized_plpc,
                'color': '#00ff88' if unrealized_plpc > 0 else '#ff4444'
            })

        df = pd.DataFrame(data)

        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=df['symbol'],
            y=df['return'],
            marker_color=df['color']
        )])

        fig.update_layout(
            title="Position Returns (%)",
            xaxis_title="Symbol",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=300,
            margin=dict(t=50, b=30, l=30, r=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_recent_activity(self):
        """Display recent trading activity."""
        try:
            # Get recent orders
            orders = self.trading_client.get_orders(status='all', limit=10)

            if orders:
                order_data = []
                for order in orders:
                    order_data.append({
                        'Time': order.submitted_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'Symbol': order.symbol,
                        'Side': order.side.value,
                        'Qty': float(order.qty) if order.qty else 0,
                        'Status': order.status.value,
                        'Filled Price': f"${float(order.filled_avg_price):.2f}" if order.filled_avg_price else "-"
                    })

                df = pd.DataFrame(order_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent trading activity")

        except Exception as e:
            st.error(f"Error loading recent activity: {e}")

    def _page_positions_trades(self):
        """Positions and trades detail page."""
        st.markdown('<div class="section-header"><h2>💼 Positions & Trades</h2></div>',
                    unsafe_allow_html=True)

        if not self.alpaca_connected:
            st.error("❌ Alpaca connection required")
            return

        # Current positions
        st.markdown("### Current Positions")
        self._display_current_positions()

        # Recent trades with ML context
        st.markdown("### Recent Trades with ML Predictions")
        self._display_trades_with_ml_context()

        # Trade performance analysis
        st.markdown("### Trade Performance Analysis")
        self._display_trade_performance_analysis()

    def _display_current_positions(self):
        """Display detailed current positions."""
        try:
            positions = self.trading_client.get_all_positions()

            if not positions:
                st.info("No active positions")
                return

            # Load ML predictions for context
            predictions_df = self._load_latest_predictions()

            position_data = []
            for pos in positions:
                # Get ML prediction for this symbol if available
                ml_prediction = None
                ml_confidence = None
                if predictions_df is not None and pos.symbol in predictions_df['symbol'].values:
                    pred_row = predictions_df[predictions_df['symbol'] == pos.symbol].iloc[0]
                    ml_prediction = pred_row.get('predicted_return', None)
                    ml_confidence = pred_row.get('confidence', None)

                position_data.append({
                    'Symbol': pos.symbol,
                    'Shares': int(pos.qty),
                    'Avg Cost': f"${float(pos.avg_entry_price):.2f}",
                    'Current Price': f"${float(pos.current_price):.2f}",
                    'Market Value': f"${float(pos.market_value):,.2f}",
                    'Unrealized P&L': f"${float(pos.unrealized_pl):+,.2f}",
                    'Return %': f"{float(pos.unrealized_plpc) * 100:+.2f}%",
                    'ML Prediction': f"{ml_prediction:.2%}" if ml_prediction else "-",
                    'ML Confidence': f"{ml_confidence:.1%}" if ml_confidence else "-"
                })

            df = pd.DataFrame(position_data)

            # Style the dataframe
            def color_pnl(val):
                if isinstance(val, str) and val.startswith('$'):
                    return 'color: #00ff88' if '+' in val else 'color: #ff4444'
                return ''

            styled_df = df.style.applymap(color_pnl, subset=['Unrealized P&L', 'Return %'])
            st.dataframe(styled_df, use_container_width=True)

            # Position summary metrics
            total_value = sum(float(p.market_value) for p in positions)
            total_pl = sum(float(p.unrealized_pl) for p in positions)
            total_return = total_pl / (total_value - total_pl) * 100 if (total_value - total_pl) > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Market Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Unrealized P&L", f"${total_pl:+,.2f}")
            with col3:
                st.metric("Total Return", f"{total_return:+.2f}%")

        except Exception as e:
            st.error(f"Error loading positions: {e}")

    def _display_trades_with_ml_context(self):
        """Display recent trades with their ML prediction context."""
        try:
            # Get recent filled orders
            orders = self.trading_client.get_orders(status='filled', limit=20)

            if not orders:
                st.info("No recent filled orders")
                return

            # Get trade context from database
            trade_data = []
            for order in orders:
                # Look for ML prediction context
                # In a real system, you'd store this when placing the trade
                trade_data.append({
                    'Date': order.filled_at.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': order.symbol,
                    'Action': order.side.value,
                    'Shares': int(order.filled_qty) if order.filled_qty else 0,
                    'Price': f"${float(order.filled_avg_price):.2f}",
                    'Value': f"${float(order.filled_qty) * float(order.filled_avg_price):,.2f}"
                })

            df = pd.DataFrame(trade_data)
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading trades: {e}")

    def _display_trade_performance_analysis(self):
        """Analyze trade performance."""
        try:
            # Get closed positions from database
            trades_df = self.db.get_recent_trades(days_back=30)

            if trades_df.empty:
                st.info("No closed trades in the last 30 days")
                return

            # Calculate metrics
            winning_trades = trades_df[trades_df['profit_loss'] > 0]
            losing_trades = trades_df[trades_df['profit_loss'] < 0]

            win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
            avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['profit_loss'].sum() / losing_trades['profit_loss'].sum()) if len(
                losing_trades) > 0 and losing_trades['profit_loss'].sum() != 0 else 0

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col2:
                st.metric("Average Win", f"${avg_win:,.2f}")
            with col3:
                st.metric("Average Loss", f"${avg_loss:,.2f}")
            with col4:
                st.metric("Profit Factor", f"{profit_factor:.2f}")

            # Trade distribution
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=trades_df['profit_loss'],
                nbinsx=30,
                name='P&L Distribution',
                marker_color='#00ff88'
            ))

            fig.update_layout(
                title="Trade P&L Distribution (30 Days)",
                xaxis_title="Profit/Loss ($)",
                yaxis_title="Count",
                template="plotly_dark",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing trades: {e}")

    def _page_ml_predictions(self):
        """ML predictions page."""
        st.markdown('<div class="section-header"><h2>🧠 Machine Learning Predictions</h2></div>',
                    unsafe_allow_html=True)

        # Load latest predictions
        predictions_df = self._load_latest_predictions()

        if predictions_df is None or predictions_df.empty:
            st.warning("No ML predictions available. Run the prediction system to generate predictions.")
            return

        # Model performance metrics
        st.markdown("### Model Performance Metrics")
        self._display_model_metrics()

        # Current predictions
        st.markdown("### Current ML Predictions")
        self._display_ml_predictions(predictions_df)

        # Prediction accuracy tracking
        st.markdown("### Prediction Accuracy Tracking")
        self._display_prediction_accuracy()

        # Feature importance
        st.markdown("### Feature Importance Analysis")
        self._display_feature_importance()

    def _display_model_metrics(self):
        """Display ML model performance metrics."""
        try:
            # Look for saved model metrics
            metrics_file = DATA_DIR / 'model_metrics.json'

            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    accuracy = metrics.get('directional_accuracy', 0) * 100
                    st.metric("Directional Accuracy", f"{accuracy:.1f}%")

                with col2:
                    sharpe = metrics.get('sharpe_ratio', 0)
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                with col3:
                    mse = metrics.get('mse', 0)
                    st.metric("MSE", f"{mse:.4f}")

                with col4:
                    confidence = metrics.get('avg_confidence', 0) * 100
                    st.metric("Avg Confidence", f"{confidence:.1f}%")
            else:
                st.info("Model metrics not available. Train models to generate metrics.")

        except Exception as e:
            st.error(f"Error loading model metrics: {e}")

    def _display_ml_predictions(self, predictions_df):
        """Display current ML predictions."""
        # Add signal strength
        predictions_df['signal_strength'] = predictions_df['predicted_return'] * predictions_df['confidence']

        # Separate buy and sell signals
        buy_signals = predictions_df[predictions_df['predicted_return'] > 0.02].sort_values('signal_strength',
                                                                                            ascending=False)
        sell_signals = predictions_df[predictions_df['predicted_return'] < -0.02].sort_values('signal_strength')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Buy Signals")
            if not buy_signals.empty:
                for _, row in buy_signals.head(10).iterrows():
                    st.markdown(f"""
                    <div class='signal-buy'>
                    <strong>{row['symbol']}</strong><br>
                    Return: {row['predicted_return']:.2%} | Confidence: {row['confidence']:.1%}<br>
                    Signal Strength: {row['signal_strength']:.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No buy signals")

        with col2:
            st.markdown("#### 🔴 Sell Signals")
            if not sell_signals.empty:
                for _, row in sell_signals.head(10).iterrows():
                    st.markdown(f"""
                    <div class='signal-sell'>
                    <strong>{row['symbol']}</strong><br>
                    Return: {row['predicted_return']:.2%} | Confidence: {row['confidence']:.1%}<br>
                    Signal Strength: {abs(row['signal_strength']):.3f}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No sell signals")

        # Prediction distribution
        st.markdown("### Prediction Distribution")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=predictions_df['predicted_return'] * 100,
            nbinsx=50,
            name='Predicted Returns',
            marker_color='#00ff88'
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig.add_vline(x=2, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_vline(x=-2, line_dash="dash", line_color="red", opacity=0.5)

        fig.update_layout(
            title="Distribution of Predicted Returns",
            xaxis_title="Predicted Return (%)",
            yaxis_title="Count",
            template="plotly_dark",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    def _display_prediction_accuracy(self):
        """Track and display prediction accuracy over time."""
        st.info(
            "Prediction accuracy tracking requires storing predictions with actual outcomes. This feature will be populated as trades complete.")

    def _display_feature_importance(self):
        """Display feature importance from the models."""
        try:
            # Look for saved feature importance
            importance_file = DATA_DIR / 'feature_importance.json'

            if importance_file.exists():
                with open(importance_file, 'r') as f:
                    importance_data = json.load(f)

                # Convert to dataframe and plot
                df = pd.DataFrame(list(importance_data.items()), columns=['Feature', 'Importance'])
                df = df.sort_values('Importance', ascending=True).tail(20)

                fig = go.Figure(go.Bar(
                    x=df['Importance'],
                    y=df['Feature'],
                    orientation='h',
                    marker_color='#00ff88'
                ))

                fig.update_layout(
                    title="Top 20 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    template="plotly_dark",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance data not available")

        except Exception as e:
            st.error(f"Error loading feature importance: {e}")

    def _page_news_analysis(self):
        """News analysis page."""
        st.markdown('<div class="section-header"><h2>📰 News Analysis & Sentiment</h2></div>',
                    unsafe_allow_html=True)

        if not self.openai_connected:
            st.warning("⚠️ News analysis requires OpenAI API connection. Please configure your OpenAI API key.")
            return

        # Load latest news analysis
        news_data = self._load_latest_news_analysis()

        if news_data is None:
            st.info("No recent news analysis available. The system analyzes news periodically when running.")
            return

        # Overall market sentiment
        st.markdown("### Market Sentiment Overview")
        self._display_market_sentiment(news_data)

        # Individual stock impacts
        st.markdown("### Stock-Specific News Impact")
        self._display_stock_news_impacts(news_data)

        # Sentiment timeline
        st.markdown("### Sentiment Timeline")
        self._display_sentiment_timeline()

    def _display_market_sentiment(self, news_data):
        """Display overall market sentiment gauge."""
        if isinstance(news_data, pd.DataFrame) and not news_data.empty:
            # Calculate overall sentiment
            avg_sentiment = news_data['impact_score'].mean()

            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_sentiment,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Sentiment"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "#00ff88" if avg_sentiment > 0 else "#ff4444"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "#ff0000"},
                        {'range': [-0.5, -0.2], 'color': "#ff6666"},
                        {'range': [-0.2, 0.2], 'color': "#ffaa00"},
                        {'range': [0.2, 0.5], 'color': "#66ff66"},
                        {'range': [0.5, 1], 'color': "#00ff00"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))

            fig.update_layout(
                template="plotly_dark",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Sentiment breakdown
            bullish = len(news_data[news_data['impact_score'] > 0.2])
            bearish = len(news_data[news_data['impact_score'] < -0.2])
            neutral = len(news_data) - bullish - bearish

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullish Signals", bullish, delta=None, delta_color="normal")
            with col2:
                st.metric("Neutral Signals", neutral, delta=None, delta_color="normal")
            with col3:
                st.metric("Bearish Signals", bearish, delta=None, delta_color="normal")

    def _display_stock_news_impacts(self, news_data):
        """Display news impacts by stock."""
        if isinstance(news_data, pd.DataFrame) and not news_data.empty:
            # Sort by absolute impact
            news_data['abs_impact'] = news_data['impact_score'].abs()
            top_impacts = news_data.sort_values('abs_impact', ascending=False).head(20)

            for _, row in top_impacts.iterrows():
                sentiment_class = 'sentiment-positive' if row['impact_score'] > 0 else 'sentiment-negative'
                direction = "LONG" if row['impact_score'] > 0 else "SHORT"

                st.markdown(f"""
                <div style='padding: 10px; margin: 5px 0; border-left: 4px solid {'#00ff88' if row['impact_score'] > 0 else '#ff4444'}; background: rgba(255,255,255,0.05);'>
                <strong>{row['symbol']}</strong> - <span class='{sentiment_class}'>{direction}</span><br>
                Impact: {row['impact_score']:.2f} | Confidence: {row['confidence']:.1%}<br>
                <small>{row.get('reasoning', 'No details available')[:200]}...</small>
                </div>
                """, unsafe_allow_html=True)

    def _display_sentiment_timeline(self):
        """Display sentiment over time."""
        st.info(
            "Sentiment timeline tracks how market sentiment changes throughout the day. This feature requires continuous news monitoring.")

    def _page_backtesting(self):
        """Backtesting results page."""
        st.markdown('<div class="section-header"><h2>📊 Backtesting Results</h2></div>',
                    unsafe_allow_html=True)

        # The display_backtest_page expects a TabDisplay instance, not TradingDashboard
        # We need to either:
        # 1. Modify this method to handle backtesting directly, or
        # 2. Create a compatible object to pass

        # Option 1: Handle backtesting directly here
        from monitoring.backtest_runner import BacktestRunner
        from config.settings import WATCHLIST, DATA_DIR, BACKTEST_CONFIG

        # Check for existing results
        results_file = DATA_DIR / 'backtest_results.json'

        if results_file.exists():
            # Load and display existing results
            try:
                with open(results_file, 'r') as f:
                    saved_results = json.load(f)

                # Display results
                self._display_backtest_results(saved_results)

                # Option to run new backtest
                if st.button("Run New Backtest", type="secondary"):
                    st.session_state['run_new_backtest'] = True
                    st.rerun()

            except Exception as e:
                st.error(f"Error loading backtest results: {e}")
                st.session_state['run_new_backtest'] = True
        else:
            st.session_state['run_new_backtest'] = True

        # Run new backtest interface
        if st.session_state.get('run_new_backtest', False):
            self._run_new_backtest_interface()

    def _run_new_backtest_interface(self):
        """Interface for running a new backtest."""
        st.markdown("### Configure Backtest Parameters")

        col1, col2 = st.columns(2)

        with col1:
            # Date range
            st.subheader("Date Range")

            # Default to last 2 years
            default_end = datetime.now()
            default_start = default_end - timedelta(days=730)

            start_date = st.date_input(
                "Start Date",
                value=default_start,
                max_value=default_end - timedelta(days=90)
            )

            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=start_date + timedelta(days=90)
            )

            # Backtest type
            backtest_type = st.radio(
                "Backtest Type",
                ["Standard (Single Split)", "Walk-Forward Analysis"],
                help="Walk-forward simulates real-world retraining"
            )

        with col2:
            # Symbol selection
            st.subheader("Symbol Selection")

            symbol_option = st.radio(
                "Symbols to Test",
                ["Top 20 Symbols", "All Symbols", "Custom Selection"]
            )

            if symbol_option == "Custom Selection":
                from config.settings import WATCHLIST
                selected_symbols = st.multiselect(
                    "Select Symbols",
                    options=WATCHLIST,
                    default=WATCHLIST[:10]
                )
            elif symbol_option == "Top 20 Symbols":
                from config.settings import WATCHLIST
                selected_symbols = WATCHLIST[:20]
            else:
                from config.settings import WATCHLIST
                selected_symbols = WATCHLIST

            st.info(f"Selected {len(selected_symbols)} symbols")

        # Run backtest button
        if st.button("Run Backtest", type="primary", disabled=not selected_symbols):
            # Create progress container
            progress_container = st.container()

            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Progress callback
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)

                # Initialize runner
                from monitoring.backtest_runner import BacktestRunner
                runner = BacktestRunner()

                try:
                    # Run appropriate backtest
                    if backtest_type == "Standard (Single Split)":
                        results = runner.run_standard_backtest(
                            symbols=selected_symbols,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            end_date=datetime.combine(end_date, datetime.min.time()),
                            progress_callback=update_progress
                        )
                    else:
                        results = runner.run_walk_forward_backtest(
                            symbols=selected_symbols,
                            start_date=datetime.combine(start_date, datetime.min.time()),
                            end_date=datetime.combine(end_date, datetime.min.time()),
                            progress_callback=update_progress
                        )

                    # Clear progress and show results
                    progress_container.empty()
                    st.success("Backtest completed successfully!")

                    # Clear the run new backtest flag
                    st.session_state['run_new_backtest'] = False

                    # Rerun to show results
                    st.rerun()

                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
                    logger.error(f"Backtest error: {e}", exc_info=True)

    def _display_backtest_results(self, results: Dict[str, Any]):
        """Display backtest results."""
        # Key metrics
        st.markdown("### Backtest Performance Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{results.get('total_return', 0):.2%}",
                help="Total portfolio return over backtest period"
            )

        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{results.get('sharpe_ratio', 0):.2f}",
                help="Risk-adjusted return metric"
            )

        with col3:
            st.metric(
                "Max Drawdown",
                f"{results.get('max_drawdown', 0):.2%}",
                help="Largest peak-to-trough decline"
            )

        with col4:
            st.metric(
                "Win Rate",
                f"{results.get('win_rate', 0):.1%}",
                help="Percentage of profitable trades"
            )

        # Equity curve
        if 'equity_curve' in results and results['equity_curve']:
            st.markdown("### Equity Curve")

            fig = go.Figure()

            # Get dates if available
            if 'dates' in results:
                dates = [datetime.fromisoformat(d) for d in results['dates']]
            else:
                dates = list(range(len(results['equity_curve'])))

            # Add equity curve
            fig.add_trace(go.Scatter(
                x=dates,
                y=results['equity_curve'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2)
            ))

            # Add initial capital line
            from config.settings import BACKTEST_CONFIG
            initial_capital = BACKTEST_CONFIG.get('initial_capital', 100000)
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital"
            )

            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        # Trade analysis
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Trade Statistics")

            trade_stats = {
                "Total Trades": results.get('total_trades', 0),
                "Average Win": f"${results.get('avg_win', 0):,.2f}",
                "Average Loss": f"${results.get('avg_loss', 0):,.2f}",
                "Profit Factor": f"{results.get('profit_factor', 0):.2f}",
                "Avg Holding Days": f"{results.get('avg_holding_days', 0):.1f}",
                "Prediction Accuracy": f"{results.get('prediction_accuracy', 0):.1%}"
            }

            for metric, value in trade_stats.items():
                st.metric(metric, value)

        with col2:
            st.markdown("### Risk Metrics")

            risk_metrics = {
                "Annualized Return": f"{results.get('annualized_return', 0):.2%}",
                "Max Consecutive Losses": results.get('max_consecutive_losses', 'N/A'),
                "Recovery Factor": f"{abs(results.get('total_return', 0) / results.get('max_drawdown', -0.01)):.2f}",
                "Final Capital": f"${results.get('final_capital', 0):,.2f}"
            }

            for metric, value in risk_metrics.items():
                st.metric(metric, value)

    def _display_equity_curve(self, results):
        """Display backtest equity curve."""
        if 'equity_curve' in results:
            equity_data = results['equity_curve']

            fig = go.Figure()

            # Add equity curve
            fig.add_trace(go.Scatter(
                x=list(range(len(equity_data))),
                y=equity_data,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2)
            ))

            # Add drawdown
            if 'drawdown' in results:
                fig.add_trace(go.Scatter(
                    x=list(range(len(results['drawdown']))),
                    y=[d * 100 for d in results['drawdown']],
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='#ff4444', width=1),
                    yaxis='y2'
                ))

            fig.update_layout(
                title="Backtest Equity Curve",
                xaxis_title="Trading Days",
                yaxis_title="Portfolio Value ($)",
                yaxis2=dict(
                    title="Drawdown (%)",
                    overlaying='y',
                    side='right'
                ),
                template="plotly_dark",
                height=400,
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

    def _display_trade_analysis(self, results):
        """Display backtest trade analysis."""
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Trades", results.get('total_trades', 0))
            st.metric("Avg Win", f"${results.get('avg_win', 0):,.2f}")
            st.metric("Avg Loss", f"${results.get('avg_loss', 0):,.2f}")

        with col2:
            st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
            st.metric("Avg Days Held", f"{results.get('avg_holding_days', 0):.1f}")
            accuracy = results.get('prediction_accuracy', 0) * 100
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")

    def _display_risk_analysis(self, results):
        """Display risk analysis from backtest."""
        # Monthly returns
        if 'monthly_returns' in results:
            monthly_data = results['monthly_returns']

            fig = go.Figure(go.Bar(
                x=list(range(len(monthly_data))),
                y=[r * 100 for r in monthly_data],
                marker_color=['#00ff88' if r > 0 else '#ff4444' for r in monthly_data]
            ))

            fig.update_layout(
                title="Monthly Returns",
                xaxis_title="Month",
                yaxis_title="Return (%)",
                template="plotly_dark",
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        # Risk metrics table
        risk_metrics = {
            "Annualized Return": f"{results.get('annualized_return', 0):.2%}",
            "Annualized Volatility": f"{results.get('annualized_vol', 0):.2%}",
            "Downside Deviation": f"{results.get('downside_deviation', 0):.2%}",
            "Sortino Ratio": f"{results.get('sortino_ratio', 0):.2f}",
            "Calmar Ratio": f"{results.get('calmar_ratio', 0):.2f}",
            "VaR (95%)": f"${results.get('var_95', 0):,.2f}"
        }

        risk_df = pd.DataFrame(list(risk_metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(risk_df, use_container_width=True)

    def _page_performance_analytics(self):
        """Detailed performance analytics page."""
        st.markdown('<div class="section-header"><h2>📉 Performance Analytics</h2></div>',
                    unsafe_allow_html=True)

        if not self.alpaca_connected:
            st.error("❌ Alpaca connection required for performance analytics")
            return

        # Time period selector
        period = st.selectbox("Select Time Period", ["1W", "1M", "3M", "6M", "1Y", "All Time"])

        # Performance metrics
        st.markdown("### Performance Metrics")
        self._display_performance_metrics(period)

        # Returns analysis
        st.markdown("### Returns Analysis")
        self._display_returns_analysis(period)

        # Risk-adjusted returns
        st.markdown("### Risk-Adjusted Performance")
        self._display_risk_adjusted_performance(period)

    def _display_performance_metrics(self, period):
        """Display performance metrics for selected period."""
        # This would calculate metrics based on actual trading history
        st.info("Performance metrics will be calculated from your trading history")

    def _display_returns_analysis(self, period):
        """Display returns analysis."""
        # Placeholder for returns distribution, daily returns chart, etc.
        st.info("Returns analysis will show daily, weekly, and monthly return patterns")

    def _display_risk_adjusted_performance(self, period):
        """Display risk-adjusted performance metrics."""
        st.info("Risk-adjusted metrics including Sharpe, Sortino, and Information ratios")

    def _page_help_documentation(self):
        """Help and documentation page."""
        st.markdown('<div class="section-header"><h2>❓ Help & Documentation</h2></div>',
                    unsafe_allow_html=True)

        # System overview
        st.markdown("""
        ### System Overview

        This AI Trading System combines advanced machine learning with real-time market data to identify trading opportunities.

        #### Key Components:

        1. **Machine Learning Ensemble**
           - Uses 5 different neural network architectures
           - Deep Neural Networks, LSTM, Attention Networks, CNN, and Base Networks
           - Ensemble approach reduces overfitting and improves prediction reliability
           - Trained on 97 engineered features (62 base + 35 interactions)

        2. **Feature Engineering**
           - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
           - Price patterns and momentum signals
           - Volume analysis and market microstructure
           - Cross-asset correlations and sector analysis
           - Risk metrics and volatility regimes

        3. **News Sentiment Analysis**
           - Uses OpenAI GPT-4 for natural language processing
           - Analyzes news from multiple sources (Yahoo, Bloomberg, Reuters, etc.)
           - Predicts market impact and assigns confidence scores
           - Provides actionable long/short signals based on sentiment

        4. **Risk Management**
           - Kelly Criterion for optimal position sizing
           - Correlation limits to avoid concentration risk
           - Portfolio heat monitoring (max 10% at risk)
           - Stop loss (5%) and take profit (15%) levels
           - Maximum position limits and daily loss limits

        5. **Execution System**
           - Integrates with Alpaca for paper/live trading
           - Real-time position monitoring
           - Automatic order execution based on ML signals
           - Slippage and commission modeling
        """)

        # How to use
        st.markdown("""
        ### How to Use This Dashboard

        1. **Portfolio Overview**: Monitor your overall performance, positions, and compare against SPY benchmark
        2. **Positions & Trades**: View detailed position information with ML context
        3. **ML Predictions**: See current buy/sell signals from the ensemble model
        4. **News Analysis**: Review sentiment analysis and market impact predictions
        5. **Backtesting**: Evaluate strategy performance on historical data
        6. **Performance Analytics**: Deep dive into returns, risk metrics, and patterns
        """)

        # Trading signals interpretation
        st.markdown("""
        ### Interpreting Trading Signals

        **Buy Signals (Green)**:
        - Predicted Return > 2% over 21-day horizon
        - Confidence > 70% from ensemble model
        - Positive risk/reward based on Kelly Criterion

        **Sell Signals (Red)**:
        - Predicted Return < -2% over 21-day horizon
        - Or existing position reaching stop loss/take profit
        - Or confidence degradation in held positions

        **Signal Strength**:
        - Calculated as Predicted Return × Confidence
        - Higher values indicate stronger conviction
        - Used for position sizing and prioritization
        """)

        # Risk warnings
        st.markdown("""
        ### ⚠️ Important Risk Disclaimers

        - **No Guarantee**: Past performance does not guarantee future results
        - **Market Risk**: All trading involves risk of loss
        - **Model Risk**: ML models can fail or produce incorrect predictions
        - **Execution Risk**: Slippage and fees can impact returns
        - **Technology Risk**: System failures or data issues can occur

        **This system is for educational and research purposes. Always do your own research and consult with financial advisors.**
        """)

        # Technical details
        with st.expander("Technical Implementation Details"):
            st.markdown("""
            **Data Pipeline**:
            - Alpaca API for real-time market data
            - yfinance for historical data and benchmarks
            - PostgreSQL/SQLite for data persistence
            - Redis for caching (if configured)

            **ML Pipeline**:
            - PyTorch for neural network implementation
            - Scikit-learn for preprocessing and metrics
            - Feature scaling with RobustScaler
            - Temporal train/validation/test splits

            **Infrastructure**:
            - Streamlit for dashboard
            - Asyncio for concurrent operations
            - Scheduled tasks for data updates
            - Logging and monitoring systems
            """)

    # Helper methods

    def _is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now(self.tz)

        # Check if weekend
        if now.weekday() >= 5:
            return False

        # Check time (9:30 AM - 4:00 PM ET)
        market_open = now.time() >= datetime.strptime("09:30", "%H:%M").time()
        market_close = now.time() <= datetime.strptime("16:00", "%H:%M").time()

        return market_open and market_close

    def _get_next_market_open(self):
        """Get next market open time."""
        now = datetime.now(self.tz)

        # If it's during the week but after hours
        if now.weekday() < 5 and now.time() > datetime.strptime("16:00", "%H:%M").time():
            # Next day at 9:30 AM
            next_open = now.replace(hour=9, minute=30, second=0) + timedelta(days=1)
        # If it's before market open
        elif now.weekday() < 5 and now.time() < datetime.strptime("09:30", "%H:%M").time():
            # Today at 9:30 AM
            next_open = now.replace(hour=9, minute=30, second=0)
        else:
            # Find next Monday
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now.replace(hour=9, minute=30, second=0) + timedelta(days=days_until_monday)

        return next_open

    def _load_latest_predictions(self):
        """Load the most recent ML predictions."""
        try:
            # Look for prediction files
            prediction_files = list(DATA_DIR.glob("predictions_*.csv"))
            if not prediction_files:
                # Try alternate name
                prediction_files = list(DATA_DIR.glob("predictions.csv"))

            if prediction_files:
                # Get most recent file
                if len(prediction_files) == 1:
                    latest_file = prediction_files[0]
                else:
                    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)

                df = pd.read_csv(latest_file)
                return df

            return None

        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None

    def _load_latest_news_analysis(self):
        """Load the most recent news analysis."""
        try:
            # Look for news prediction files
            news_files = list(DATA_DIR.glob("news_predictions_*.csv"))

            if news_files:
                latest_file = max(news_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_csv(latest_file)
                return df

            return None

        except Exception as e:
            logger.error(f"Error loading news analysis: {e}")
            return None

    def _calculate_portfolio_returns(self, trades_df):
        """Calculate portfolio returns from trade history."""
        try:
            # This is a simplified calculation
            # In production, you'd track actual portfolio value over time
            if trades_df.empty:
                return None

            # Group by date and calculate cumulative P&L
            trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
            daily_pnl = trades_df.groupby('date')['profit_loss'].sum()

            # Calculate cumulative returns
            initial_capital = 100000  # Assumption
            cumulative_pnl = daily_pnl.cumsum()
            portfolio_value = initial_capital + cumulative_pnl
            daily_returns = portfolio_value.pct_change().fillna(0)
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            return pd.DataFrame({
                'daily_return': daily_returns,
                'cumulative_return': cumulative_returns
            })

        except Exception as e:
            logger.error(f"Error calculating portfolio returns: {e}")
            return None

    def _run_backtest(self):
        """Run backtest and save results."""
        try:
            # This would run your actual backtest
            # For now, show a message
            st.info(
                "Backtest functionality requires loading trained models and historical data. This would typically take 5-15 minutes.")

            # In production, you would:
            # 1. Load the trained models
            # 2. Get historical data
            # 3. Run BacktestEngine
            # 4. Save results

        except Exception as e:
            st.error(f"Error running backtest: {e}")


# Main entry point
def main():
    """Run the enhanced dashboard."""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
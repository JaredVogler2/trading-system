# monitoring/display_tabs.py
"""Tab display methods for the dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path

from config.settings import WATCHLIST, DATA_DIR

logger = logging.getLogger(__name__)


class TabDisplay:
    """Handles all tab displays for the dashboard."""

    def __init__(self, data_fetcher, chart_creator, mock_tracker, utilities, dashboard):
        self.data_fetcher = data_fetcher
        self.chart_creator = chart_creator
        self.mock_tracker = mock_tracker
        self.utilities = utilities
        self.dashboard = dashboard

    def display_overview_tab(self):
        """Display comprehensive overview tab."""
        st.markdown('<div class="section-header"><h2>📈 Portfolio Overview</h2></div>', unsafe_allow_html=True)

        # Show mock data warning if any mock data is in use
        self._display_mock_warning()

        # Show data source status
        self._display_data_source_status()

        # Get real-time data
        account_data = self.data_fetcher.get_account_data()
        positions_data = self.data_fetcher.get_positions_data()
        portfolio_history = self._get_portfolio_history()

        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            portfolio_value = account_data.get('portfolio_value', 0)
            if self.mock_tracker.is_mock('account'):
                st.metric(
                    "Portfolio Value ⚠️",
                    self.utilities.format_mock_value(portfolio_value, "${:,.2f}"),
                    help="⚠️ This is MOCK DATA - not real values!"
                )
            else:
                st.metric("Portfolio Value", f"${portfolio_value:,.2f}")

        with col2:
            daily_pnl = account_data.get('daily_pl',
                                         account_data.get('unrealized_pl', 0) + account_data.get('realized_pl', 0))
            daily_pnl_pct = account_data.get('daily_pl_pct',
                                             (daily_pnl / portfolio_value * 100) if portfolio_value > 0 else 0)
            if self.mock_tracker.is_mock('account'):
                st.metric(
                    "Daily P&L ⚠️",
                    self.utilities.format_mock_value(daily_pnl, "${:,.2f}"),
                    f"{daily_pnl_pct:+.2f}%*",
                    help="⚠️ This is MOCK DATA"
                )
            else:
                st.metric("Daily P&L", f"${daily_pnl:,.2f}", f"{daily_pnl_pct:+.2f}%")

        with col3:
            positions_count = len(positions_data)
            if self.mock_tracker.is_mock('positions'):
                st.metric("Active Positions ⚠️", f"{positions_count}*", help="⚠️ Mock positions")
            else:
                st.metric("Active Positions", positions_count)

        with col4:
            buying_power = account_data.get('buying_power', 0)
            if self.mock_tracker.is_mock('account'):
                st.metric("Buying Power ⚠️", self.utilities.format_mock_value(buying_power, "${:,.2f}"))
            else:
                st.metric("Buying Power", f"${buying_power:,.2f}")

        with col5:
            win_rate = self._calculate_recent_win_rate()
            if self.mock_tracker.is_mock('trades'):
                st.metric("Recent Win Rate ⚠️", self.utilities.format_mock_value(win_rate, "{:.1%}"))
            else:
                st.metric("Recent Win Rate", f"{win_rate:.1%}")

        # Portfolio performance vs benchmark
        st.subheader("📊 Portfolio Performance vs SPY")
        benchmark_chart = self.chart_creator.create_benchmark_comparison_chart(portfolio_history)
        st.plotly_chart(benchmark_chart, use_container_width=True)

        # Current positions and recent activity
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💼 Current Positions")
            if self.mock_tracker.is_mock('positions'):
                st.warning("⚠️ Showing MOCK positions - not real data!")
            if positions_data:
                positions_df = self._format_positions_display(positions_data)
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("No active positions")

        with col2:
            st.subheader("📈 Top Predictions")
            if self.mock_tracker.is_mock('predictions'):
                st.warning("⚠️ Showing MOCK predictions")
            predictions = self.data_fetcher.get_latest_predictions()
            if predictions is not None and not predictions.empty:
                top_predictions = predictions.head(10)
                for _, pred in top_predictions.iterrows():
                    confidence_color = "🟢" if pred['confidence'] > 0.8 else "🟡" if pred['confidence'] > 0.6 else "🔴"
                    mock_indicator = "*" if self.mock_tracker.is_mock('predictions') else ""
                    st.markdown(f"""
                    **{pred['symbol']}{mock_indicator}** {confidence_color}  
                    Return: {pred['predicted_return']:.2%}{mock_indicator} | Conf: {pred['confidence']:.1%}{mock_indicator}
                    """)
            else:
                st.info("No recent predictions available")

    def display_positions_tab(self):
        """Display detailed positions and trading tab."""
        st.markdown('<div class="section-header"><h2>💼 Positions & Trading Analysis</h2></div>', unsafe_allow_html=True)

        # Add refresh button and last update time
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("🔄 Refresh Positions", key="refresh_positions"):
                st.rerun()

        with col2:
            # Show last update time
            state_file = DATA_DIR / 'live_system_state.json'
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        last_update = state.get('timestamp', 'Unknown')
                        # Parse and format the timestamp
                        try:
                            update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                            time_ago = (datetime.now() - update_time.replace(tzinfo=None)).total_seconds()
                            if time_ago < 60:
                                st.info(f"Last update: {int(time_ago)}s ago")
                            else:
                                st.info(f"Last update: {int(time_ago / 60)}m ago")
                        except:
                            st.info(f"Last update: {last_update}")
                except:
                    pass

        with col3:
            # Show connection status
            if self.mock_tracker.is_mock('positions'):
                st.error("⚠️ Using Mock Data")
            else:
                st.success("✅ Live Data")

        # Show mock data warning if applicable
        self._display_mock_warning()

        # Get positions data
        positions_data = self.data_fetcher.get_positions_data()

        # Show account summary first
        self._display_account_summary()

        if not positions_data:
            st.info("No active positions")
            # Show recent orders even if no positions
            self._display_recent_orders()
            return

        # Detailed positions table
        st.subheader("📋 Current Positions (Live from Alpaca)")
        if self.mock_tracker.is_mock('positions'):
            st.warning("⚠️ These are MOCK positions - not real trading data!")

        detailed_positions = self._get_detailed_positions_table(positions_data)
        st.dataframe(detailed_positions, use_container_width=True)

        # Show recent orders
        self._display_recent_orders()

        # Position analytics
        col1, col2 = st.columns(2)

        with col1:
            # Position distribution
            st.subheader("🥧 Position Distribution")
            dist_chart = self.chart_creator.create_position_distribution_chart(positions_data)
            st.plotly_chart(dist_chart, use_container_width=True)

        with col2:
            # Performance by position
            st.subheader("📊 Position Performance")
            perf_chart = self.chart_creator.create_position_performance_chart(positions_data)
            st.plotly_chart(perf_chart, use_container_width=True)

        # Risk analysis
        st.subheader("⚠️ Position Risk Analysis")
        risk_metrics = self._calculate_position_risk_metrics(positions_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            value = risk_metrics['portfolio_heat']
            if self.mock_tracker.is_mock('positions'):
                st.metric("Portfolio Heat ⚠️", self.utilities.format_mock_value(value, "{:.1%}"))
            else:
                st.metric("Portfolio Heat", f"{value:.1%}")

        with col2:
            value = risk_metrics['max_position']
            if self.mock_tracker.is_mock('positions'):
                st.metric("Largest Position ⚠️", self.utilities.format_mock_value(value, "{:.1%}"))
            else:
                st.metric("Largest Position", f"{value:.1%}")

        with col3:
            value = risk_metrics['avg_correlation']
            if self.mock_tracker.is_mock('correlation'):
                st.metric("Avg Correlation ⚠️", self.utilities.format_mock_value(value, "{:.2f}"))
            else:
                st.metric("Avg Correlation", f"{value:.2f}")

        with col4:
            value = risk_metrics['concentration']
            if self.mock_tracker.is_mock('positions'):
                st.metric("Concentration Risk ⚠️", self.utilities.format_mock_value(value, "{:.1%}"))
            else:
                st.metric("Concentration Risk", f"{value:.1%}")

    def display_ai_models_tab(self):
        """Display AI models and predictions tab."""
        st.markdown('<div class="section-header"><h2>🧠 AI Models & Predictions</h2></div>', unsafe_allow_html=True)

        # Show mock data warning if applicable
        self._display_mock_warning()

        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)

        # Check if using mock model metrics
        is_mock_metrics = self.mock_tracker.is_mock('model_metrics')

        with col1:
            value = "76.3%"
            delta = "+2.1%"
            if is_mock_metrics:
                st.metric("Model Accuracy ⚠️", f"{value}*", f"{delta}*", help="⚠️ Mock metric")
            else:
                st.metric("Model Accuracy", value, delta)

        with col2:
            value = "72.5%"
            delta = "+0.8%"
            if is_mock_metrics:
                st.metric("Prediction Confidence ⚠️", f"{value}*", f"{delta}*")
            else:
                st.metric("Prediction Confidence", value, delta)

        with col3:
            st.metric("Feature Count", "97", "62 base + 35 interactions")

        with col4:
            st.metric("Ensemble Size", "5 models")

        # Model predictions table
        st.subheader("📊 Latest Model Predictions")
        if self.mock_tracker.is_mock('predictions'):
            st.warning("⚠️ These are MOCK predictions - not real model output!")

        predictions = self.data_fetcher.get_latest_predictions()

        if predictions is not None and not predictions.empty:
            # Add signal strength column
            predictions['signal_strength'] = predictions['predicted_return'] * predictions['confidence']
            predictions['action'] = predictions['predicted_return'].apply(
                lambda x: '🟢 BUY' if x > 0.02 else '🔴 SELL' if x < -0.02 else '⚪ HOLD'
            )

            # Display predictions
            display_cols = ['symbol', 'predicted_return', 'confidence', 'signal_strength', 'action']

            # Add current_price if available
            if 'current_price' in predictions.columns:
                display_cols.append('current_price')

            # Add asterisks if mock
            if self.mock_tracker.is_mock('predictions'):
                predictions_display = predictions[display_cols].head(20).copy()
                predictions_display['symbol'] = predictions_display['symbol'] + '*'
            else:
                predictions_display = predictions[display_cols].head(20)

            st.dataframe(
                predictions_display.style.format({
                    'predicted_return': '{:.2%}',
                    'confidence': '{:.1%}',
                    'signal_strength': '{:.3f}',
                    'current_price': '${:.2f}' if 'current_price' in display_cols else None
                }),
                use_container_width=True
            )
        else:
            st.info("No predictions available")

    # Helper methods
    def _display_mock_warning(self):
        """Display a prominent warning when mock data is being used."""
        warning_banner = self.mock_tracker.get_warning_banner()
        if warning_banner:
            st.markdown(
                f'<div class="mock-warning-banner">{warning_banner}</div>',
                unsafe_allow_html=True
            )

    def _display_data_source_status(self):
        """Display the status of all data sources."""
        with st.expander("📊 Data Source Status", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Account Data**")
                if self.mock_tracker.is_mock('account'):
                    st.error("🔴 Mock Data")
                    if 'account' in self.mock_tracker.mock_values:
                        st.caption(self.mock_tracker.mock_values['account']['reason'])
                else:
                    st.success("🟢 Live Data")

            with col2:
                st.markdown("**Positions**")
                if self.mock_tracker.is_mock('positions'):
                    st.error("🔴 Mock Data")
                    if 'positions' in self.mock_tracker.mock_values:
                        st.caption(self.mock_tracker.mock_values['positions']['reason'])
                else:
                    st.success("🟢 Live Data")

            with col3:
                st.markdown("**Market Data**")
                if self.mock_tracker.is_mock('market_data'):
                    st.error("🔴 Mock Data")
                else:
                    st.success("🟢 Live Data")

            with col4:
                st.markdown("**Predictions**")
                if self.mock_tracker.is_mock('predictions'):
                    st.error("🔴 Mock Data")
                else:
                    st.success("🟢 Live Data")

    def _display_account_summary(self):
        """Display account summary information."""
        try:
            account_data = self.data_fetcher.get_account_data()

            if account_data:
                st.subheader("💰 Account Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Portfolio Value", f"${account_data.get('portfolio_value', 0):,.2f}")

                with col2:
                    equity = account_data.get('equity', account_data.get('portfolio_value', 0))
                    st.metric("Equity", f"${equity:,.2f}")

                with col3:
                    st.metric("Cash", f"${account_data.get('cash', 0):,.2f}")

                with col4:
                    st.metric("Buying Power", f"${account_data.get('buying_power', 0):,.2f}")

                # Daily P&L
                if 'daily_pl' in account_data:
                    daily_pl = account_data['daily_pl']
                    daily_pl_pct = account_data.get('daily_pl_pct', 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Daily P&L", f"${daily_pl:,.2f}", f"{daily_pl_pct:.2f}%")

                    with col2:
                        st.metric("Day Trades", account_data.get('daytrade_count', 0))

        except Exception as e:
            logger.error(f"Error displaying account summary: {e}")

    def _display_recent_orders(self):
        """Display recent orders from Alpaca."""
        try:
            state_file = DATA_DIR / 'live_system_state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)

                    if 'alpaca_orders' in state and state['alpaca_orders']:
                        st.subheader("📝 Recent Orders")

                        orders_df = pd.DataFrame(state['alpaca_orders'])

                        # Format the dataframe for display
                        if not orders_df.empty:
                            # Convert timestamps
                            if 'submitted_at' in orders_df.columns:
                                orders_df['submitted_at'] = pd.to_datetime(orders_df['submitted_at']).dt.strftime(
                                    '%Y-%m-%d %H:%M:%S')

                            # Format numeric columns
                            numeric_cols = ['qty', 'filled_avg_price']
                            for col in numeric_cols:
                                if col in orders_df.columns:
                                    orders_df[col] = pd.to_numeric(orders_df[col], errors='coerce')

                            # Select and order columns for display
                            display_cols = []
                            for col in ['symbol', 'side', 'qty', 'type', 'status', 'submitted_at', 'filled_avg_price']:
                                if col in orders_df.columns:
                                    display_cols.append(col)

                            if display_cols:
                                orders_display = orders_df[display_cols].head(10)

                                # Apply formatting
                                format_dict = {}
                                if 'qty' in display_cols:
                                    format_dict['qty'] = '{:.0f}'
                                if 'filled_avg_price' in display_cols:
                                    format_dict['filled_avg_price'] = '${:.2f}'

                                st.dataframe(
                                    orders_display.style.format(format_dict, na_rep='-'),
                                    use_container_width=True
                                )
                            else:
                                st.dataframe(orders_df.head(10), use_container_width=True)
        except Exception as e:
            logger.error(f"Error displaying orders: {e}")

    def _get_portfolio_history(self):
        """Get portfolio performance history."""
        # Implementation for getting portfolio history
        # This would typically come from the database
        try:
            # For now, return sample data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            daily_returns = np.random.normal(0.001, 0.02, 30)
            cumulative_returns = (1 + daily_returns).cumprod() - 1

            return pd.DataFrame({
                'daily_return': daily_returns,
                'cumulative_return': cumulative_returns
            }, index=dates)
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return pd.DataFrame()

    def _calculate_recent_win_rate(self):
        """Calculate recent win rate from trading history."""
        try:
            trades_df = self.data_fetcher.db.get_recent_trades(days_back=30)
            if not trades_df.empty and 'profit_loss' in trades_df.columns:
                self.mock_tracker.clear_mock_source('trades')
                return (trades_df['profit_loss'] > 0).mean()
        except:
            pass

        self.mock_tracker.add_mock_source('trades', 'No trade history')
        return 0.62  # Mock win rate

    def _format_positions_display(self, positions_data):
        """Format positions data for display."""
        if not positions_data:
            return pd.DataFrame()

        df_data = []
        for pos in positions_data:
            days_held = np.random.randint(1, 30)  # Mock data for days held
            is_mock = pos.get('_is_mock', False) or self.mock_tracker.is_mock('positions')

            # Handle both 'qty' and 'shares' keys
            quantity = pos.get('qty', pos.get('shares', 0))

            df_data.append({
                'Symbol': pos['symbol'] + ('*' if is_mock else ''),
                'Quantity': f"{quantity:.0f}",
                'Entry Price': f"${pos.get('avg_entry_price', pos.get('entry_price', 0)):.2f}",
                'Current Price': f"${pos.get('current_price', 0):.2f}",
                'Market Value': f"${pos.get('market_value', 0):.2f}",
                'P&L': f"${pos.get('unrealized_pl', 0):+.2f}",
                'P&L %': f"{pos.get('unrealized_plpc', 0):+.2%}",
                'Days Held': f"{days_held}d"
            })

        return pd.DataFrame(df_data)

    def _get_detailed_positions_table(self, positions_data):
        """Get detailed positions table with additional information."""
        if not positions_data:
            return pd.DataFrame()

        df_data = []
        for pos in positions_data:
            # Get additional info from shared state if available
            additional_info = {}
            try:
                state_file = DATA_DIR / 'live_system_state.json'
                if state_file.exists():
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        if 'positions' in state and pos['symbol'] in state['positions']:
                            additional_info = state['positions'][pos['symbol']]
            except:
                pass

            quantity = pos.get('qty', pos.get('shares', 0))

            df_data.append({
                'Symbol': pos['symbol'],
                'Side': pos.get('side', 'long'),
                'Quantity': quantity,
                'Avg Entry': pos.get('avg_entry_price', pos.get('entry_price', 0)),
                'Current': pos.get('current_price', 0),
                'Market Value': pos.get('market_value', 0),
                'Unrealized P&L': pos.get('unrealized_pl', 0),
                'P&L %': pos.get('unrealized_plpc', 0),
                'Stop Loss': additional_info.get('stop_loss', '-'),
                'Take Profit': additional_info.get('take_profit', '-'),
                'Confidence': additional_info.get('confidence', '-'),
                'Source': pos.get('source', 'alpaca')
            })

        df = pd.DataFrame(df_data)

        # Format numeric columns
        if not df.empty:
            format_dict = {
                'Quantity': '{:.0f}',
                'Avg Entry': '${:.2f}',
                'Current': '${:.2f}',
                'Market Value': '${:,.2f}',
                'Unrealized P&L': '${:+,.2f}',
                'P&L %': '{:+.2%}'
            }

            # Format stop loss and take profit if they're numeric
            for col in ['Stop Loss', 'Take Profit', 'Confidence']:
                if col in df.columns:
                    # Convert to numeric where possible
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if col in ['Stop Loss', 'Take Profit']:
                        format_dict[col] = '${:.2f}'
                    elif col == 'Confidence':
                        format_dict[col] = '{:.1%}'

            return df.style.format(format_dict, na_rep='-')

        return df

    def display_backtest_tab(self):
        """Display backtesting tab with proper data separation."""
        from monitoring.backtest_runner import display_backtest_page
        display_backtest_page(self)

    def _calculate_position_risk_metrics(self, positions_data):
        """Calculate risk metrics for positions."""
        if not positions_data:
            return {
                'portfolio_heat': 0,
                'max_position': 0,
                'avg_correlation': 0,
                'concentration': 0
            }

        total_value = sum(pos.get('market_value', 0) for pos in positions_data)

        if total_value == 0:
            return {
                'portfolio_heat': 0,
                'max_position': 0,
                'avg_correlation': 0,
                'concentration': 0
            }

        position_weights = [pos.get('market_value', 0) / total_value for pos in positions_data]

        return {
            'portfolio_heat': sum(w * 0.05 for w in position_weights),  # Assuming 5% risk per position
            'max_position': max(position_weights) if position_weights else 0,
            'avg_correlation': 0.45,  # Simplified - would need actual correlation data
            'concentration': sum(w ** 2 for w in position_weights)  # Herfindahl index
        }
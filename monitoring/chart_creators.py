# monitoring/chart_creators.py
"""Chart creation methods for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class ChartCreator:
    """Handles all chart creation for the dashboard."""

    def __init__(self, mock_tracker):
        self.mock_tracker = mock_tracker

    def create_benchmark_comparison_chart(self, portfolio_history):
        """Create portfolio vs SPY benchmark comparison."""
        # Get SPY data
        spy_data = self._get_spy_benchmark_data()

        fig = go.Figure()

        # Portfolio performance
        fig.add_trace(go.Scatter(
            x=portfolio_history.index,
            y=portfolio_history['cumulative_return'] * 100,
            mode='lines',
            name='Portfolio' + (' (MOCK)' if self.mock_tracker.is_mock('performance') else ''),
            line=dict(color='#00ff88', width=3)
        ))

        # SPY benchmark
        if spy_data is not None:
            fig.add_trace(go.Scatter(
                x=spy_data.index,
                y=spy_data['cumulative_return'] * 100,
                mode='lines',
                name='SPY Benchmark',
                line=dict(color='#ffaa00', width=2, dash='dash')
            ))

        fig.update_layout(
            title="Portfolio vs SPY Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_dark",
            height=400,
            hovermode='x unified'
        )

        return fig

    def create_position_distribution_chart(self, positions_data):
        """Create position distribution pie chart."""
        if not positions_data:
            return go.Figure()

        symbols = [pos['symbol'] for pos in positions_data]
        values = [pos['market_value'] for pos in positions_data]

        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=values,
            hole=0.3,
            marker=dict(
                colors=px.colors.qualitative.Set3
            )
        )])

        fig.update_layout(
            title="Position Distribution" + (" (MOCK)" if self.mock_tracker.is_mock('positions') else ""),
            template="plotly_dark",
            height=400
        )

        return fig

    def create_position_performance_chart(self, positions_data):
        """Create position performance bar chart."""
        if not positions_data:
            return go.Figure()

        symbols = [pos['symbol'] for pos in positions_data]
        returns = [pos['unrealized_plpc'] * 100 for pos in positions_data]
        colors = ['#00ff88' if r > 0 else '#ff4444' for r in returns]

        fig = go.Figure(data=[go.Bar(
            x=symbols,
            y=returns,
            marker_color=colors
        )])

        fig.update_layout(
            title="Position Performance (%)" + (" (MOCK)" if self.mock_tracker.is_mock('positions') else ""),
            xaxis_title="Symbol",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400
        )

        return fig

    def create_cumulative_returns_chart(self, portfolio_history):
        """Create cumulative returns chart."""
        if portfolio_history.empty:
            return go.Figure()

        fig = go.Figure()

        cumulative_returns = (1 + portfolio_history['daily_return']).cumprod() - 1

        fig.add_trace(go.Scatter(
            x=portfolio_history.index,
            y=cumulative_returns * 100,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))

        fig.update_layout(
            title="Cumulative Returns" + (" (MOCK)" if self.mock_tracker.is_mock('performance') else ""),
            xaxis_title="Date",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400
        )

        return fig

    def create_drawdown_chart(self, portfolio_history):
        """Create drawdown chart."""
        if portfolio_history.empty:
            return go.Figure()

        cumulative_returns = (1 + portfolio_history['daily_return']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=portfolio_history.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.1)'
        ))

        fig.update_layout(
            title="Portfolio Drawdown" + (" (MOCK)" if self.mock_tracker.is_mock('performance') else ""),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=400
        )

        return fig

    def create_technical_chart(self, price_data, symbol, chart_type):
        """Create technical analysis chart."""
        fig = go.Figure()

        is_mock = self.mock_tracker.is_mock('price_data')

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price' + (' (MOCK)' if is_mock else '')
            ))
        elif chart_type == "OHLC":
            fig.add_trace(go.Ohlc(
                x=price_data.index,
                open=price_data['Open'],
                high=price_data['High'],
                low=price_data['Low'],
                close=price_data['Close'],
                name='Price' + (' (MOCK)' if is_mock else '')
            ))
        else:  # Line chart
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name='Close Price' + (' (MOCK)' if is_mock else ''),
                line=dict(color='#00ff88', width=2)
            ))

        # Add moving averages
        ma20 = price_data['Close'].rolling(window=20).mean()
        ma50 = price_data['Close'].rolling(window=50).mean()

        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=ma20,
            mode='lines',
            name='MA20',
            line=dict(color='yellow', width=1)
        ))

        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=ma50,
            mode='lines',
            name='MA50',
            line=dict(color='orange', width=1)
        ))

        # Add volume subplot
        fig.add_trace(go.Bar(
            x=price_data.index,
            y=price_data['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='lightblue',
            opacity=0.5
        ))

        fig.update_layout(
            title=f"{symbol} Technical Analysis" + (" (MOCK DATA)" if is_mock else ""),
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template="plotly_dark",
            height=600,
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False,
                range=[0, price_data['Volume'].max() * 4]
            ),
            xaxis_rangeslider_visible=False
        )

        return fig

    def _get_spy_benchmark_data(self):
        """Get SPY benchmark data for comparison."""
        try:
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="1y")

            # Calculate cumulative returns
            spy_data['daily_return'] = spy_data['Close'].pct_change()
            spy_data['cumulative_return'] = (1 + spy_data['daily_return']).cumprod() - 1

            return spy_data
        except Exception as e:
            logger.error(f"Error getting SPY data: {e}")
            return None
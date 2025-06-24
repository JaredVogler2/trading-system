# monitoring/backtest_runner.py
"""
Backtest runner with proper temporal separation and no data leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Tuple, Any

from config.settings import WATCHLIST, DATA_DIR, MODEL_DIR, BACKTEST_CONFIG
from backtesting.engine import BacktestEngine, run_walk_forward_analysis
from features.pipeline import FeatureEngineeringPipeline
from data.collector import DataCollector

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Runs backtests with proper data separation."""

    def __init__(self):
        self.feature_pipeline = None
        self.backtest_engine = None
        self.results = None

    def prepare_backtest_data(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            train_test_split: float = 0.7,
            temporal_gap_days: int = 7
    ) -> Tuple[Dict[str, pd.DataFrame], datetime, datetime]:
        """
        Prepare data for backtesting with proper temporal separation.

        Args:
            symbols: List of symbols to backtest
            start_date: Overall start date
            end_date: Overall end date
            train_test_split: Fraction of data for training (0.7 = 70%)
            temporal_gap_days: Days between train and test sets

        Returns:
            Tuple of (feature_data, train_end_date, test_start_date)
        """
        # Collect raw data
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
        clean_data = collector.add_market_data(clean_data)

        # Initialize feature pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(symbols)

        # Calculate features
        from features.base_features import engineer_base_features
        from features.interactions import engineer_interaction_features

        base_features = engineer_base_features(clean_data, save_to_db=False)
        feature_data = engineer_interaction_features(base_features)

        # Calculate split dates
        total_days = (end_date - start_date).days
        train_days = int(total_days * train_test_split)

        train_end_date = start_date + timedelta(days=train_days)
        test_start_date = train_end_date + timedelta(days=temporal_gap_days)

        return feature_data, train_end_date, test_start_date

    def train_models_for_backtest(
            self,
            feature_data: Dict[str, pd.DataFrame],
            train_end_date: datetime,
            temporal_gap: int = 7
    ) -> Tuple[List, Any, Dict]:
        """
        Train models using only data up to train_end_date.

        Returns:
            Tuple of (trainers, calibrator, scalers)
        """
        # Prepare training data with temporal cutoff
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.feature_pipeline.prepare_training_data(
                feature_data,
                prediction_horizon=21,
                train_end_date=train_end_date,
                temporal_gap=temporal_gap
            )

        # Create and train models
        from models.neural_networks import create_ensemble_models, train_ensemble

        input_dim = X_train.shape[1]
        models = create_ensemble_models(
            input_dim=input_dim,
            num_models=5
        )

        # Train ensemble
        trainers, calibrator = train_ensemble(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            models,
            epochs=50,  # Reduced for faster backtesting
            save_dir=None  # Don't save during backtest
        )

        return trainers, calibrator, self.feature_pipeline.scalers

    def run_standard_backtest(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            progress_callback=None
    ) -> Dict[str, Any]:
        """Run standard backtest with single train/test split."""

        if progress_callback:
            progress_callback(0.1, "Preparing data...")

        # Prepare data with temporal separation
        feature_data, train_end, test_start = self.prepare_backtest_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            train_test_split=0.7,
            temporal_gap_days=BACKTEST_CONFIG['temporal_gap_days']
        )

        if progress_callback:
            progress_callback(0.3, "Training models on historical data...")

        # Train models using only training period data
        trainers, calibrator, scalers = self.train_models_for_backtest(
            feature_data,
            train_end,
            temporal_gap=BACKTEST_CONFIG['temporal_gap_days']
        )

        if progress_callback:
            progress_callback(0.6, "Running backtest simulation...")

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            initial_capital=BACKTEST_CONFIG['initial_capital'],
            position_size=BACKTEST_CONFIG['position_size'],
            max_positions=BACKTEST_CONFIG['max_positions'],
            stop_loss=BACKTEST_CONFIG['stop_loss'],
            take_profit=BACKTEST_CONFIG['take_profit'],
            commission=BACKTEST_CONFIG['commission'],
            slippage=BACKTEST_CONFIG['slippage'],
            temporal_gap=BACKTEST_CONFIG['temporal_gap_days']
        )

        # Run backtest on test period only
        results = self.backtest_engine.run_backtest(
            feature_data=feature_data,
            trainers=trainers,
            calibrator=calibrator,
            scalers=scalers,
            train_end_date=train_end,
            test_start_date=test_start,
            test_end_date=end_date,
            prediction_horizon=21
        )

        if progress_callback:
            progress_callback(0.9, "Generating results...")

        # Store results
        self.results = results

        # Save results
        self._save_results(results)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return results

    def run_walk_forward_backtest(
            self,
            symbols: List[str],
            start_date: datetime,
            end_date: datetime,
            initial_train_months: int = 24,
            test_months: int = 3,
            retrain_frequency: int = 3,
            progress_callback=None
    ) -> Dict[str, Any]:
        """Run walk-forward analysis with periodic retraining."""

        if progress_callback:
            progress_callback(0.1, "Preparing data for walk-forward analysis...")

        # Prepare complete dataset
        feature_data, _, _ = self.prepare_backtest_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            train_test_split=0.8,  # Use more data
            temporal_gap_days=BACKTEST_CONFIG['temporal_gap_days']
        )

        if progress_callback:
            progress_callback(0.3, "Starting walk-forward analysis...")

        # Run walk-forward analysis
        results = run_walk_forward_analysis(
            feature_data=feature_data,
            initial_train_months=initial_train_months,
            test_months=test_months,
            retrain_frequency=retrain_frequency,
            start_date=start_date,
            end_date=end_date
        )

        if progress_callback:
            progress_callback(0.9, "Generating walk-forward results...")

        # Store results
        self.results = results

        # Save results
        self._save_results(results)

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results to file."""
        try:
            # Prepare serializable results
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'total_return': results.get('total_return', 0),
                'annualized_return': results.get('annualized_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'win_rate': results.get('win_rate', 0),
                'total_trades': results.get('total_trades', 0),
                'avg_win': results.get('avg_win', 0),
                'avg_loss': results.get('avg_loss', 0),
                'profit_factor': results.get('profit_factor', 0),
                'prediction_accuracy': results.get('prediction_accuracy', 0),
                'avg_holding_days': results.get('avg_holding_days', 0),
                'final_capital': results.get('final_capital', 0)
            }

            # Save portfolio history if available
            if 'portfolio_history' in results and not results['portfolio_history'].empty:
                portfolio_df = results['portfolio_history']
                save_data['equity_curve'] = portfolio_df['total_value'].tolist()
                save_data['daily_returns'] = portfolio_df['daily_return'].tolist()
                save_data['dates'] = [d.isoformat() for d in portfolio_df.index]

            # Save to file
            results_file = DATA_DIR / 'backtest_results.json'
            with open(results_file, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Backtest results saved to {results_file}")

        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")


def display_backtest_page(tab_display):
    """Display the backtest page in the dashboard."""
    st.markdown('<div class="section-header"><h2>📊 Backtesting Results</h2></div>',
                unsafe_allow_html=True)

    # Check for existing results
    results_file = DATA_DIR / 'backtest_results.json'

    if results_file.exists():
        # Load and display existing results
        try:
            with open(results_file, 'r') as f:
                saved_results = json.load(f)

            # Display results
            _display_backtest_results(saved_results)

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
        _run_new_backtest_interface()


def _run_new_backtest_interface():
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
            selected_symbols = st.multiselect(
                "Select Symbols",
                options=WATCHLIST,
                default=WATCHLIST[:10]
            )
        elif symbol_option == "Top 20 Symbols":
            selected_symbols = WATCHLIST[:20]
        else:
            selected_symbols = WATCHLIST

        st.info(f"Selected {len(selected_symbols)} symbols")

        # Advanced settings
        with st.expander("Advanced Settings"):
            if backtest_type == "Walk-Forward Analysis":
                initial_train = st.slider("Initial Training Months", 12, 36, 24)
                test_period = st.slider("Test Period Months", 1, 6, 3)
                retrain_freq = st.slider("Retrain Every N Months", 1, 6, 3)

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
                        initial_train_months=initial_train,
                        test_months=test_period,
                        retrain_frequency=retrain_freq,
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


def _display_backtest_results(results: Dict[str, Any]):
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

    # Monthly returns heatmap (if available)
    if 'monthly_returns' in results:
        st.markdown("### Monthly Returns Heatmap")
        st.info("Monthly returns visualization would be displayed here")

    # Information about temporal separation
    with st.expander("ℹ️ About Backtest Methodology"):
        st.markdown("""
        **Temporal Separation**

        This backtest uses strict temporal separation to prevent data leakage:

        1. **Training Period**: Models are trained only on historical data up to a cutoff date
        2. **Temporal Gap**: A 7-day gap between training and test data prevents information leakage
        3. **Test Period**: Trading simulation uses only future data the model has never seen
        4. **No Look-Ahead Bias**: Features are calculated using only past information at each point

        **Walk-Forward Analysis**

        For more realistic results, use walk-forward analysis which:
        - Periodically retrains models with new data
        - Simulates how the system would perform with regular updates
        - Tests robustness across different market conditions
        """)

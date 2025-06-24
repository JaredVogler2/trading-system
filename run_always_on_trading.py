# run_always_on_trading.py
"""
Enhanced live trading runner that operates 24/7 with continuous monitoring.
Updated to use the comprehensive dashboard with full functionality.
UPDATED: Sequential execution to prevent progress bar interruptions + Progress Tracking
"""

import asyncio
import threading
import time
import argparse
import logging
import schedule
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import pandas as pd
import textwrap
from tqdm import tqdm
import colorama
from colorama import Fore, Style

from main import TradingSystem
from trading.live_trader import LiveTradingSystem, RiskManager
from config.settings import WATCHLIST, DATA_DIR
from dotenv import load_dotenv

# Initialize colorama for Windows
colorama.init()

# Load environment variables
load_dotenv()


# Set up logging with color
class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


# Configure logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

file_handler = logging.FileHandler('logs/always_on_trading.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and displays system initialization progress."""

    def __init__(self, total_steps=10):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = []
        self.start_time = time.time()

    def start_step(self, description):
        """Start a new step."""
        self.current_step += 1
        self.step_descriptions.append(description)

        # Clear line and print progress
        print(f"\r{' ' * 100}\r", end='')  # Clear line
        print(f"{Fore.CYAN}[{self.current_step}/{self.total_steps}]{Style.RESET_ALL} {description}...", end='',
              flush=True)

        return time.time()

    def complete_step(self, start_time, success=True, message=""):
        """Complete current step."""
        duration = time.time() - start_time

        if success:
            status = f"{Fore.GREEN}✓{Style.RESET_ALL}"
            color = Fore.GREEN
        else:
            status = f"{Fore.RED}✗{Style.RESET_ALL}"
            color = Fore.RED

        # Clear line and print final status
        print(f"\r{' ' * 100}\r", end='')  # Clear line
        print(f"{status} [{self.current_step}/{self.total_steps}] {self.step_descriptions[-1]} "
              f"{color}({duration:.1f}s){Style.RESET_ALL} {message}")

    def get_elapsed_time(self):
        """Get total elapsed time."""
        return time.time() - self.start_time


class AlwaysOnTradingOrchestrator:
    """24/7 trading orchestrator with continuous monitoring and data updates."""

    def __init__(self, symbols=None, paper=True):
        """Initialize the always-on trading orchestrator."""
        self.symbols = symbols or WATCHLIST  # Use ALL symbols from WATCHLIST
        self.paper = paper
        self.trading_system = None
        self.live_system = None
        self.dashboard_process = None
        self.progress = ProgressTracker(total_steps=12)

        # Background tasks
        self.news_analyzer = None
        self.prediction_updater = None
        self.data_refresher = None

        # State management
        self.state_file = DATA_DIR / 'always_on_state.json'
        self.state = {
            'last_prediction_update': None,
            'last_news_update': None,
            'last_data_refresh': None,
            'market_hours_today': None,
            'system_start_time': datetime.now().isoformat(),
            'last_model_retrain': None,
            'model_version': 'v1.0'
        }

        self.load_state()

    def load_state(self):
        """Load persistent state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    saved_state = json.load(f)
                    self.state.update(saved_state)
                logger.info("Loaded persistent state")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")

    def save_state(self):
        """Save current state."""
        try:
            self.state['last_update'] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")

    def initialize_systems(self):
        """Initialize all trading systems with progress tracking."""
        print(f"\n{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Initializing Always-On Trading System{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'=' * 60}{Style.RESET_ALL}\n")

        try:
            # Step 1: Initialize trading system
            step_time = self.progress.start_step("Initializing trading system")
            self.trading_system = TradingSystem(symbols=self.symbols)
            self.progress.complete_step(step_time, message=f"{len(self.symbols)} symbols")

            # Step 2: Collect and prepare data
            step_time = self.progress.start_step("Collecting market data")
            print(f"\n{Fore.CYAN}This may take a few minutes for {len(self.symbols)} symbols...{Style.RESET_ALL}")

            # Modified to skip database saves for faster startup
            from features.pipeline import FeatureEngineeringPipeline
            self.trading_system.feature_pipeline = FeatureEngineeringPipeline(self.symbols)

            feature_data = self.trading_system.feature_pipeline.run_full_pipeline(
                save_to_db=False,  # Skip database saves for faster startup
                scale_features=True
            )
            self.progress.complete_step(step_time, message=f"{len(feature_data)} symbols processed")

            # Step 3: Load or train models
            step_time = self.progress.start_step("Loading AI models")
            try:
                self.trading_system._load_models(feature_data)
                self.progress.complete_step(step_time, message="Models loaded from disk")
            except Exception as e:
                self.progress.complete_step(step_time, False, "Failed to load")

                # Step 3b: Train new models
                step_time = self.progress.start_step("Training new AI models")
                print(f"\n{Fore.YELLOW}Training new models, this will take several minutes...{Style.RESET_ALL}")
                self.trading_system.train_models(feature_data)
                self.progress.complete_step(step_time, message="Training complete")

            # Step 4: Initialize live trading system
            step_time = self.progress.start_step("Setting up live trading system")
            self.live_system = LiveTradingSystem(
                trading_system=self.trading_system,
                paper=self.paper
            )
            self.progress.complete_step(step_time, message="Paper trading" if self.paper else "LIVE trading")

            # Step 5: Update correlations
            step_time = self.progress.start_step("Calculating correlations")
            self._update_correlations(feature_data)
            self.progress.complete_step(step_time)

            print(
                f"\n{Fore.GREEN}✅ System initialization complete in {self.progress.get_elapsed_time():.1f}s{Style.RESET_ALL}\n")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise

    def _update_correlations(self, feature_data):
        """Update correlation matrix in risk manager."""
        try:
            returns_data = {}
            for symbol, df in tqdm(feature_data.items(), desc="Processing returns", leave=False):
                if 'returns_1d' in df.columns and len(df) > 30:
                    returns_data[symbol] = df['returns_1d'].fillna(0)

            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                self.live_system.risk_manager.update_correlation_matrix(returns_df)
                logger.info("✅ Correlation matrix updated")
        except Exception as e:
            logger.warning(f"Could not update correlations: {e}")

    def start_dashboard(self):
        """Start the enhanced dashboard."""
        step_time = self.progress.start_step("Starting dashboard")

        try:
            # Create dashboard runner script that uses the enhanced dashboard
            dashboard_script = textwrap.dedent("""
                import sys
                import os
                sys.path.append('.')

                # IMPORTANT: Load environment variables BEFORE any other imports
                from dotenv import load_dotenv
                load_dotenv()

                import streamlit as st
                import logging

                # Verify environment variables are loaded
                api_key = os.getenv("ALPACA_API_KEY")
                secret_key = os.getenv("ALPACA_SECRET_KEY")

                if not api_key or not secret_key:
                    st.error("ERROR: Alpaca API keys not found in environment!")
                    st.error("Please check your .env file")
                    st.stop()

                # Import and run the ENHANCED dashboard instead of comprehensive
                from monitoring.enhanced_dashboard import main

                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)

                # Run the enhanced dashboard
                try:
                    main()
                except Exception as e:
                    st.error(f"Dashboard error: {e}")
                    logger.error(f"Dashboard error: {e}", exc_info=True)
            """).strip()

            # Save the dashboard script
            with open('temp_enhanced_dashboard_runner.py', 'w', encoding='utf-8') as f:
                f.write(dashboard_script)

            # Start Streamlit dashboard
            self.dashboard_process = subprocess.Popen([
                sys.executable, '-m', 'streamlit', 'run',
                'temp_enhanced_dashboard_runner.py',
                '--server.port', '8501',
                '--server.headless', 'true',
                '--server.runOnSave', 'true'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            time.sleep(3)  # Give dashboard time to start
            self.progress.complete_step(step_time, message="http://localhost:8501")

        except Exception as e:
            self.progress.complete_step(step_time, False, str(e))
            raise

    def schedule_background_tasks(self):
        """Schedule background tasks that run regardless of market hours."""
        step_time = self.progress.start_step("Scheduling background tasks")

        # Data refresh - every 5 minutes during market hours, every hour otherwise
        schedule.every(5).minutes.do(self._refresh_data_if_market_open)
        schedule.every().hour.do(self._refresh_data_always)

        # Predictions - every 10 minutes during market hours, every 2 hours otherwise
        schedule.every(10).minutes.do(self._update_predictions_if_market_open)
        schedule.every(2).hours.do(self._update_predictions_always)

        # News analysis - every 30 minutes during market hours, every 4 hours otherwise
        schedule.every(30).minutes.do(self._update_news_if_market_open)
        schedule.every(4).hours.do(self._update_news_always)

        # State persistence - every minute
        schedule.every().minute.do(self.save_state)

        # System health check - every 15 minutes
        schedule.every(15).minutes.do(self._health_check)

        # Update shared state for dashboard - every 5 seconds
        schedule.every(5).seconds.do(self.update_shared_state)

        # Nightly model retraining at 2 AM ET
        schedule.every().day.at("02:00").do(self._retrain_models_nightly)

        self.progress.complete_step(step_time, message="All tasks scheduled")

    def _retrain_models_nightly(self):
        """Retrain models with latest data including today's price action."""
        try:
            logger.info("=" * 60)
            logger.info("🌙 Starting nightly model retraining...")
            logger.info(f"Retraining started at: {datetime.now()}")

            # Check if it's a weekday (market was open today)
            if datetime.now().weekday() >= 5:  # Saturday = 5, Sunday = 6
                logger.info("Weekend - skipping retraining")
                return

            # Save current state before retraining
            self.save_state()

            # Collect fresh data including today's closing prices
            logger.info("📊 Collecting latest market data...")
            feature_data = self.trading_system.collect_and_prepare_data(
                force_refresh=True,
                end_date=datetime.now()  # Include today's data
            )

            # Backup existing models
            self._backup_models()

            # Retrain models with new data
            logger.info("🧠 Training models with latest data...")
            self.trading_system.train_models(feature_data, retrain=True)

            # Update correlation matrix with new data
            self._update_correlations(feature_data)

            # Verify new models work
            logger.info("✅ Verifying new models...")
            test_predictions = self.trading_system.generate_predictions()

            if test_predictions is not None and not test_predictions.empty:
                logger.info(f"✅ Model retraining successful! Generated {len(test_predictions)} predictions")
                logger.info(f"Average confidence: {test_predictions['confidence'].mean():.2%}")

                # Update state with retraining info
                self.state['last_model_retrain'] = datetime.now().isoformat()
                self.state['model_version'] = datetime.now().strftime('%Y%m%d')
                self.save_state()
            else:
                logger.error("❌ Model verification failed - reverting to backup")
                self._restore_model_backup()

            logger.info(f"Retraining completed at: {datetime.now()}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"❌ Nightly retraining failed: {e}")
            logger.error("Keeping existing models")
            # Restore from backup if available
            try:
                self._restore_model_backup()
            except:
                pass

    def _backup_models(self):
        """Backup existing models before retraining."""
        try:
            import shutil
            from config.settings import MODEL_DIR

            backup_dir = MODEL_DIR / f"backup_{datetime.now().strftime('%Y%m%d')}"
            backup_dir.mkdir(exist_ok=True)

            # Copy all model files
            for model_file in MODEL_DIR.glob("model_*.pt"):
                shutil.copy2(model_file, backup_dir / model_file.name)

            # Copy scaler files
            for scaler_file in MODEL_DIR.glob("scaler_*.pkl"):
                shutil.copy2(scaler_file, backup_dir / scaler_file.name)

            logger.info(f"✅ Models backed up to {backup_dir}")

        except Exception as e:
            logger.error(f"Failed to backup models: {e}")

    def _restore_model_backup(self):
        """Restore models from most recent backup."""
        try:
            import shutil
            from config.settings import MODEL_DIR

            # Find most recent backup
            backup_dirs = sorted([d for d in MODEL_DIR.iterdir() if d.is_dir() and d.name.startswith("backup_")])

            if not backup_dirs:
                logger.error("No backup found to restore")
                return

            latest_backup = backup_dirs[-1]

            # Restore model files
            for model_file in latest_backup.glob("model_*.pt"):
                shutil.copy2(model_file, MODEL_DIR / model_file.name)

            # Restore scaler files
            for scaler_file in latest_backup.glob("scaler_*.pkl"):
                shutil.copy2(scaler_file, MODEL_DIR / scaler_file.name)

            logger.info(f"✅ Models restored from {latest_backup}")

        except Exception as e:
            logger.error(f"Failed to restore models: {e}")

    def _is_market_open(self):
        """Check if market is currently open."""
        now = datetime.now()
        weekday = now.weekday()

        # Weekend check
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Time check (9:30 AM - 4:00 PM ET)
        current_time = now.time()
        market_open = current_time >= pd.Timestamp("09:30").time()
        market_close = current_time <= pd.Timestamp("16:00").time()

        return market_open and market_close

    def _refresh_data_if_market_open(self):
        """Refresh data only if market is open."""
        if self._is_market_open():
            self._refresh_data_always()

    def _refresh_data_always(self):
        """Refresh market data (always runs) - SYNCHRONOUS."""
        try:
            print(f"\n{Fore.CYAN}🔄 Refreshing market data...{Style.RESET_ALL}", end='', flush=True)
            start_time = time.time()

            # Update feature data synchronously
            feature_data = self.trading_system.collect_and_prepare_data(force_refresh=True)

            # Update correlations
            self._update_correlations(feature_data)

            self.state['last_data_refresh'] = datetime.now().isoformat()

            duration = time.time() - start_time
            print(f"\r{Fore.GREEN}✅ Market data refreshed ({duration:.1f}s){Style.RESET_ALL}")

        except Exception as e:
            print(f"\r{Fore.RED}❌ Data refresh failed: {e}{Style.RESET_ALL}")
            logger.error(f"Data refresh failed: {e}")

    def _update_predictions_if_market_open(self):
        """Update predictions only if market is open."""
        if self._is_market_open():
            self._update_predictions_always()

    def _update_predictions_always(self):
        """Update predictions (always runs) - SYNCHRONOUS."""
        try:
            print(f"\n{Fore.CYAN}🧠 Updating predictions...{Style.RESET_ALL}", end='', flush=True)
            start_time = time.time()

            # Generate new predictions
            predictions = self.trading_system.generate_predictions()

            if predictions is not None and not predictions.empty:
                # Save predictions to CSV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                predictions_file = DATA_DIR / f'predictions_{timestamp}.csv'
                predictions.to_csv(predictions_file, index=False)

                # Keep only last 10 prediction files
                prediction_files = sorted(DATA_DIR.glob('predictions_*.csv'))
                for old_file in prediction_files[:-10]:
                    old_file.unlink()

                self.state['last_prediction_update'] = datetime.now().isoformat()

                duration = time.time() - start_time
                print(f"\r{Fore.GREEN}✅ Generated {len(predictions)} predictions ({duration:.1f}s){Style.RESET_ALL}")
            else:
                print(f"\r{Fore.YELLOW}⚠️ No predictions generated{Style.RESET_ALL}")

        except Exception as e:
            print(f"\r{Fore.RED}❌ Prediction update failed: {e}{Style.RESET_ALL}")
            logger.error(f"Prediction update failed: {e}")

    def _update_news_if_market_open(self):
        """Update news only if market is open."""
        if self._is_market_open():
            self._update_news_always()

    def _update_news_always(self):
        """Update news analysis (always runs) - SYNCHRONOUS."""
        try:
            # Check if OpenAI API key is available
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key or openai_key == "your-openai-api-key-here":
                logger.info("⚠️ OpenAI API key not available, skipping news analysis")
                return

            print(f"\n{Fore.CYAN}📰 Updating news analysis...{Style.RESET_ALL}", end='', flush=True)
            start_time = time.time()

            # Run news analysis SYNCHRONOUSLY (not in a thread)
            try:
                import asyncio
                from news.news_analyzer import NewsAnalyzer

                async def analyze():
                    analyzer = NewsAnalyzer(openai_key)
                    await analyzer.run_analysis(hours_back=24)

                # Run in current thread, not a background thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(analyze())
                loop.close()

                self.state['last_news_update'] = datetime.now().isoformat()

                duration = time.time() - start_time
                print(f"\r{Fore.GREEN}✅ News analysis completed ({duration:.1f}s){Style.RESET_ALL}")

            except Exception as e:
                print(f"\r{Fore.RED}❌ News analysis failed: {e}{Style.RESET_ALL}")
                logger.error(f"News analysis failed: {e}")

        except Exception as e:
            logger.error(f"❌ News update setup failed: {e}")

    def _health_check(self):
        """Perform system health check."""
        try:
            logger.info("🔍 Performing health check...")

            health_status = {
                'timestamp': datetime.now().isoformat(),
                'market_open': self._is_market_open(),
                'trading_system_loaded': self.trading_system is not None,
                'live_system_loaded': self.live_system is not None,
                'dashboard_running': self.dashboard_process is not None and self.dashboard_process.poll() is None,
                'active_positions': len(self.live_system.positions) if self.live_system else 0,
                'daily_pnl': self.live_system.daily_pnl if self.live_system else 0,
                'last_prediction_update': self.state.get('last_prediction_update'),
                'last_news_update': self.state.get('last_news_update'),
                'last_data_refresh': self.state.get('last_data_refresh')
            }

            # Save health status
            health_file = DATA_DIR / 'system_health.json'
            with open(health_file, 'w') as f:
                json.dump(health_status, f, indent=2)

            # Check for issues
            issues = []

            if not health_status['trading_system_loaded']:
                issues.append("Trading system not loaded")

            if not health_status['dashboard_running']:
                issues.append("Dashboard not running")
                # Try to restart dashboard
                self.start_dashboard()

            # Check if data is stale (over 2 hours old)
            if health_status['last_data_refresh']:
                last_refresh = datetime.fromisoformat(health_status['last_data_refresh'])
                if (datetime.now() - last_refresh).total_seconds() > 7200:  # 2 hours
                    issues.append("Data refresh overdue")
                    self._refresh_data_always()

            if issues:
                logger.warning(f"⚠️ Health check issues found: {issues}")
            else:
                logger.info("✅ Health check passed")

        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")

    # In the update_shared_state method, replace the existing method with this enhanced version:

    def update_shared_state(self):
        """Update shared state file for dashboard."""
        try:
            if not self.live_system:
                return

            # Get current status
            status = self.live_system.get_status()

            # Get REAL Alpaca positions directly
            alpaca_positions = []
            try:
                if self.live_system.trading_client:
                    positions = self.live_system.trading_client.get_all_positions()
                    for pos in positions:
                        alpaca_positions.append({
                            'symbol': pos.symbol,
                            'qty': float(pos.qty),
                            'avg_entry_price': float(pos.avg_entry_price),
                            'current_price': float(pos.current_price) if hasattr(pos, 'current_price') else float(
                                pos.market_value) / float(pos.qty),
                            'market_value': float(pos.market_value),
                            'unrealized_pl': float(pos.unrealized_pl),
                            'unrealized_plpc': float(pos.unrealized_plpc),
                            'side': pos.side,
                            'cost_basis': float(pos.cost_basis) if hasattr(pos, 'cost_basis') else float(
                                pos.avg_entry_price) * float(pos.qty)
                        })
                    logger.info(f"Found {len(alpaca_positions)} Alpaca positions")
            except Exception as e:
                logger.error(f"Error getting Alpaca positions: {e}")

            # Get orders from Alpaca
            alpaca_orders = []
            try:
                if self.live_system.trading_client:
                    orders = self.live_system.trading_client.get_orders()
                    for order in orders[:20]:  # Last 20 orders
                        alpaca_orders.append({
                            'symbol': order.symbol,
                            'qty': float(order.qty) if order.qty else float(order.notional) if hasattr(order,
                                                                                                       'notional') else 0,
                            'side': order.side.value if hasattr(order.side, 'value') else str(order.side),
                            'type': order.order_type.value if hasattr(order.order_type, 'value') else str(
                                order.order_type),
                            'status': order.status.value if hasattr(order.status, 'value') else str(order.status),
                            'submitted_at': order.submitted_at.isoformat() if hasattr(order.submitted_at,
                                                                                      'isoformat') else str(
                                order.submitted_at),
                            'filled_at': order.filled_at.isoformat() if hasattr(order,
                                                                                'filled_at') and order.filled_at else None,
                            'filled_avg_price': float(order.filled_avg_price) if hasattr(order,
                                                                                         'filled_avg_price') and order.filled_avg_price else None
                        })
            except Exception as e:
                logger.error(f"Error getting Alpaca orders: {e}")

            # Prepare positions data (merge internal tracking with Alpaca data)
            positions_data = {}

            # First, add all Alpaca positions
            for pos in alpaca_positions:
                symbol = pos['symbol']
                positions_data[symbol] = {
                    'symbol': symbol,
                    'shares': pos['qty'],
                    'qty': pos['qty'],  # For compatibility
                    'entry_price': pos['avg_entry_price'],
                    'avg_entry_price': pos['avg_entry_price'],
                    'current_price': pos['current_price'],
                    'market_value': pos['market_value'],
                    'unrealized_pl': pos['unrealized_pl'],
                    'unrealized_plpc': pos['unrealized_plpc'],
                    'side': pos['side'],
                    'source': 'alpaca'
                }

                # If we have internal tracking, add that info
                if symbol in self.live_system.positions:
                    internal_pos = self.live_system.positions[symbol]
                    positions_data[symbol].update({
                        'entry_time': internal_pos.entry_time.isoformat(),
                        'predicted_return': internal_pos.predicted_return,
                        'confidence': internal_pos.confidence,
                        'stop_loss': internal_pos.stop_loss,
                        'take_profit': internal_pos.take_profit,
                        'return_pct': internal_pos.return_pct
                    })

            # Get Alpaca account info
            alpaca_account = {}
            try:
                if self.live_system.trading_client:
                    account = self.live_system.trading_client.get_account()
                    alpaca_account = {
                        'account_number': account.account_number,
                        'buying_power': float(account.buying_power),
                        'portfolio_value': float(account.portfolio_value),
                        'cash': float(account.cash),
                        'status': account.status,
                        'equity': float(account.equity) if hasattr(account, 'equity') else float(
                            account.portfolio_value),
                        'last_equity': float(account.last_equity) if hasattr(account, 'last_equity') else float(
                            account.portfolio_value),
                        'long_market_value': float(account.long_market_value) if hasattr(account,
                                                                                         'long_market_value') else 0,
                        'short_market_value': float(account.short_market_value) if hasattr(account,
                                                                                           'short_market_value') else 0,
                        'initial_margin': float(account.initial_margin) if hasattr(account, 'initial_margin') else 0,
                        'maintenance_margin': float(account.maintenance_margin) if hasattr(account,
                                                                                           'maintenance_margin') else 0,
                        'daytrade_count': int(account.daytrade_count) if hasattr(account, 'daytrade_count') else 0,
                        'pattern_day_trader': bool(account.pattern_day_trader) if hasattr(account,
                                                                                          'pattern_day_trader') else False
                    }

                    # Calculate daily P&L
                    daily_pl = alpaca_account['equity'] - alpaca_account['last_equity']
                    daily_pl_pct = (daily_pl / alpaca_account['last_equity'] * 100) if alpaca_account[
                                                                                           'last_equity'] > 0 else 0

                    alpaca_account['daily_pl'] = daily_pl
                    alpaca_account['daily_pl_pct'] = daily_pl_pct

            except Exception as e:
                logger.error(f"Error getting Alpaca account info: {e}")

            # Create state dictionary with all the data
            shared_state = {
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'positions': positions_data,
                'alpaca_positions': alpaca_positions,  # Raw Alpaca positions
                'alpaca_orders': alpaca_orders,  # Recent orders
                'trades_today': self.live_system.trades_today,
                'daily_pnl': alpaca_account.get('daily_pl', self.live_system.daily_pnl),
                'portfolio_value': alpaca_account.get('portfolio_value', self.live_system._get_portfolio_value()),
                'market_open': self._is_market_open(),
                'alpaca_connected': bool(self.live_system.trading_client),
                'alpaca_account': alpaca_account,
                'system_health': {
                    'system_start_time': self.state['system_start_time'],
                    'uptime': (datetime.now() - datetime.fromisoformat(
                        self.state['system_start_time'])).total_seconds(),
                    'last_prediction_update': self.state.get('last_prediction_update'),
                    'last_news_update': self.state.get('last_news_update'),
                    'last_data_refresh': self.state.get('last_data_refresh'),
                    'last_model_retrain': self.state.get('last_model_retrain'),
                    'model_version': self.state.get('model_version', 'v1.0')
                }
            }

            # Write to shared state file
            state_file = DATA_DIR / 'live_system_state.json'
            with open(state_file, 'w') as f:
                json.dump(shared_state, f, indent=2, default=str)

            # Log summary
            logger.info(f"Updated shared state: {len(positions_data)} positions, "
                        f"${alpaca_account.get('portfolio_value', 0):,.2f} portfolio value")

        except Exception as e:
            logger.error(f"Error updating shared state: {e}", exc_info=True)

    async def run_trading_system(self):
        """Run the trading system (only executes trades when market is open)."""
        logger.info("Starting trading system...")

        while True:
            try:
                # Update shared state for dashboard
                self.update_shared_state()

                # Check if market is open for trading
                if self._is_market_open():
                    logger.info("🟢 Market is open - trading enabled")

                    # Run trading logic
                    await self._execute_trading_cycle()

                    # Wait 5 minutes between trading cycles during market hours
                    await asyncio.sleep(300)

                else:
                    logger.info("🔴 Market is closed - monitoring only")

                    # During market close, just monitor and update data
                    await self._monitor_positions()

                    # Wait 60 seconds between monitoring cycles when market is closed
                    await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Trading system error: {e}")
                await asyncio.sleep(60)

    async def _execute_trading_cycle(self):
        """Execute one trading cycle (only when market is open)."""
        try:
            # Generate fresh predictions
            predictions = self.trading_system.generate_predictions()

            if predictions is None or predictions.empty:
                logger.warning("No predictions available for trading")
                return

            # Filter for high-quality trades
            good_trades = predictions[
                (predictions['confidence'] >= 0.7) &
                (predictions['predicted_return'] >= 0.02)
                ].sort_values('predicted_return', ascending=False)

            if good_trades.empty:
                logger.info("No high-quality trading opportunities found")
                return

            logger.info(f"Found {len(good_trades)} potential trading opportunities")

            # Get account information
            try:
                account = self.live_system.trading_client.get_account()
                portfolio_value = float(account.portfolio_value)
                buying_power = float(account.buying_power)

                logger.info(f"Portfolio value: ${portfolio_value:,.2f}, Buying power: ${buying_power:,.2f}")

            except Exception as e:
                logger.error(f"Could not get account info: {e}")
                return

            # Execute trades based on risk management
            for _, trade_opportunity in good_trades.head(10).iterrows():  # Top 10 opportunities
                if trade_opportunity['symbol'] in self.live_system.positions:
                    continue  # Already have position

                # Risk management check
                should_trade, position_size, reason = self.live_system.risk_manager.should_take_trade(
                    trade_opportunity['symbol'],
                    trade_opportunity['predicted_return'],
                    trade_opportunity['confidence'],
                    self.live_system.positions,
                    portfolio_value
                )

                if not should_trade:
                    logger.info(f"Skipping {trade_opportunity['symbol']}: {reason}")
                    continue

                # Calculate order size
                position_value = portfolio_value * position_size
                current_price = trade_opportunity.get('current_price', 100)  # Fallback price
                shares = int(position_value / current_price)

                if shares < 1:
                    logger.info(f"Position too small for {trade_opportunity['symbol']}")
                    continue

                # Place order
                logger.info(f"Placing order: BUY {shares} shares of {trade_opportunity['symbol']}")

                await self.live_system._place_order(
                    symbol=trade_opportunity['symbol'],
                    shares=shares,
                    predicted_return=trade_opportunity['predicted_return'],
                    confidence=trade_opportunity['confidence'],
                    current_price=current_price
                )

                # Don't place too many orders at once
                if len(self.live_system.positions) >= 10:
                    logger.info("Maximum position limit reached")
                    break

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")

    async def _monitor_positions(self):
        """Monitor existing positions (runs always)."""
        try:
            if not self.live_system.positions:
                return

            logger.info(f"Monitoring {len(self.live_system.positions)} positions...")

            # Update position prices (using yfinance when market closed)
            import yfinance as yf

            for symbol, position in list(self.live_system.positions.items()):
                try:
                    # Get current price
                    ticker = yf.Ticker(symbol)

                    if self._is_market_open():
                        # Use real-time price during market hours
                        current_price = ticker.info.get('regularMarketPrice', position.current_price)
                    else:
                        # Use last close price when market is closed
                        current_price = ticker.info.get('previousClose', position.current_price)

                    position.update_price(current_price)

                    # Check exit conditions
                    if self._is_market_open():
                        # Only execute exits during market hours
                        if current_price <= position.stop_loss:
                            logger.info(f"Stop loss triggered for {symbol}")
                            await self.live_system._close_position(symbol, "stop_loss")

                        elif current_price >= position.take_profit:
                            logger.info(f"Take profit triggered for {symbol}")
                            await self.live_system._close_position(symbol, "take_profit")

                        elif (datetime.now() - position.entry_time).days >= 21:
                            logger.info(f"Time exit for {symbol}")
                            await self.live_system._close_position(symbol, "time_exit")

                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")

        except Exception as e:
            logger.error(f"Position monitoring error: {e}")

    def run_background_scheduler(self):
        """Run the background task scheduler."""
        logger.info("Starting background task scheduler...")

        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)

    def run(self):
        """Run the complete always-on trading system with SEQUENTIAL initialization."""
        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🚀 Starting Always-On Trading System{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")

        try:
            # Step 1: Initialize all systems
            self.initialize_systems()

            # Step 2: Start dashboard
            self.start_dashboard()

            # Step 3: Schedule background tasks
            self.schedule_background_tasks()

            # Step 4: Run initial data refresh (SYNCHRONOUSLY)
            step_time = self.progress.start_step("Initial data refresh")
            self._refresh_data_always()
            self.progress.complete_step(step_time)

            # Step 5: Run initial prediction update (SYNCHRONOUSLY)
            step_time = self.progress.start_step("Initial predictions")
            self._update_predictions_always()
            self.progress.complete_step(step_time)

            # Step 6: Run initial news analysis (SYNCHRONOUSLY)
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key-here":
                step_time = self.progress.start_step("Initial news analysis")
                self._update_news_always()  # Now runs synchronously
                self.progress.complete_step(step_time)
            else:
                print(f"\n{Fore.YELLOW}⚠️ No OpenAI API key - news analysis disabled{Style.RESET_ALL}")
                print(f"   Add OPENAI_API_KEY to .env file to enable news analysis")

            # Final status
            print(f"\n{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
            print(f"{Fore.GREEN}✅ All initialization steps completed successfully{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'=' * 60}{Style.RESET_ALL}")
            print(f"\n{Fore.CYAN}📊 Dashboard available at: {Fore.WHITE}http://localhost:8501{Style.RESET_ALL}")
            print(
                f"{Fore.CYAN}📈 Mode: {Fore.WHITE}{'Paper Trading' if self.paper else 'LIVE TRADING'}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🎯 Symbols: {Fore.WHITE}{len(self.symbols)} tracked{Style.RESET_ALL}")
            print(f"{Fore.CYAN}🤖 Models: {Fore.WHITE}Loaded and ready{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}System is now running. Press Ctrl+C to stop.{Style.RESET_ALL}\n")

            # Start background scheduler
            scheduler_thread = threading.Thread(target=self.run_background_scheduler, daemon=True)
            scheduler_thread.start()

            # Run main trading system
            asyncio.run(self.run_trading_system())

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Shutdown requested by user{Style.RESET_ALL}")
            self.shutdown()
        except Exception as e:
            print(f"\n{Fore.RED}❌ Fatal error: {e}{Style.RESET_ALL}")
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.shutdown()
            raise

    def shutdown(self):
        """Clean shutdown of all systems."""
        print(f"\n{Fore.YELLOW}🛑 Shutting down always-on trading system...{Style.RESET_ALL}")

        try:
            # Save final state
            self.save_state()

            # Stop dashboard
            if self.dashboard_process:
                logger.info("Stopping dashboard...")
                self.dashboard_process.terminate()
                try:
                    self.dashboard_process.wait(timeout=10)
                except:
                    self.dashboard_process.kill()

            # Clean up temporary files
            temp_files = [
                'temp_enhanced_dashboard_runner.py',  # Updated filename
                'temp_comprehensive_dashboard_runner.py',  # Keep old one for cleanup
                'temp_always_on_dashboard.py',
                'temp_full_dashboard.py',
                'temp_live_system_ref.pkl'
            ]

            for file in temp_files:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                        logger.info(f"Cleaned up {file}")
                    except:
                        pass

            print(f"{Fore.GREEN}✅ Shutdown complete{Style.RESET_ALL}")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point for always-on trading system."""
    parser = argparse.ArgumentParser(description='Always-On Trading System with Comprehensive Dashboard')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--paper', action='store_true', default=True,
                        help='Use paper trading (default)')
    parser.add_argument('--live', action='store_true',
                        help='Use LIVE trading (use with extreme caution!)')
    parser.add_argument('--dashboard-only', action='store_true',
                        help='Run dashboard only without trading')
    parser.add_argument('--no-news', action='store_true',
                        help='Disable news analysis')

    args = parser.parse_args()

    # Safety check for live trading
    if args.live:
        print(f"\n{Fore.RED}⚠️  WARNING: LIVE TRADING MODE SELECTED ⚠️{Style.RESET_ALL}")
        print("This will use REAL MONEY and execute REAL TRADES!")
        print("Make sure you understand the risks and have tested thoroughly.")
        response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
        if response != 'I UNDERSTAND THE RISKS':
            print("Live trading cancelled for safety.")
            return
        paper = False
    else:
        paper = True

    # Create and run orchestrator
    orchestrator = AlwaysOnTradingOrchestrator(
        symbols=args.symbols,
        paper=paper
    )

    if args.dashboard_only:
        # Run dashboard only
        logger.info("Running dashboard-only mode")
        orchestrator.start_dashboard()

        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            orchestrator.shutdown()
    else:
        # Run full system
        orchestrator.run()


if __name__ == "__main__":
    main()
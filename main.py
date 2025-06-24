# main.py
"""
Main script for the GPU-accelerated algorithmic trading system.
This script orchestrates the entire pipeline from data collection to predictions.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from config.settings import (
    WATCHLIST, MODEL_CONFIG, PREDICTION_CONFIG,
    BACKTEST_CONFIG, MODEL_DIR, LOG_DIR
)
from data.collector import DataCollector
from data.database import DatabaseManager, initialize_database
from features.pipeline import FeatureEngineeringPipeline
from models.neural_networks import (
    create_ensemble_models, train_ensemble,
    make_ensemble_predictions, device
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system class that orchestrates all components."""

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the trading system.

        Args:
            symbols: List of symbols to trade (uses WATCHLIST if None)
        """
        self.symbols = symbols or WATCHLIST
        self.db_manager = DatabaseManager()
        self.feature_pipeline = FeatureEngineeringPipeline(self.symbols)
        self.models = None
        self.trainers = None
        self.calibrator = None

        logger.info(f"Initialized trading system with {len(self.symbols)} symbols")
        logger.info(f"Using device: {device}")

    def setup_database(self):
        """Initialize database tables."""
        logger.info("Setting up database...")
        initialize_database()

    def _models_exist(self) -> bool:
        """Check if trained models exist."""
        for i in range(MODEL_CONFIG['ensemble_size']):
            model_path = MODEL_DIR / f"model_{i + 1}.pt"
            if not model_path.exists():
                return False
        return True

    @staticmethod
    def clean_old_models():
        """Remove all existing model files before training new ones."""
        model_files = list(MODEL_DIR.glob("model_*.pt"))
        scaler_files = list(MODEL_DIR.glob("scaler_*.pkl"))

        for file in model_files + scaler_files:
            try:
                file.unlink()
                logger.info(f"Removed old model file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")

    def collect_and_prepare_data(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect and prepare data for training.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            force_refresh: Force data refresh even if exists in DB

        Returns:
            Dictionary of feature DataFrames
        """
        logger.info("Collecting and preparing data...")

        # Check if features exist in database
        if not force_refresh:
            try:
                from features.pipeline import load_features_from_db
                feature_data = load_features_from_db(
                    self.symbols, start_date, end_date
                )
                if len(feature_data) == len(self.symbols):
                    logger.info("Loaded features from database")
                    return feature_data
            except Exception as e:
                logger.warning(f"Could not load features from DB: {e}")

        # Run full pipeline
        feature_data = self.feature_pipeline.run_full_pipeline(
            start_date=start_date,
            end_date=end_date,
            save_to_db=True,
            scale_features=True
        )

        return feature_data

    def train_models(self, feature_data: Dict[str, pd.DataFrame], retrain: bool = True) -> None:
        """Train ensemble models."""

        # Always clean up old models to ensure fresh training
        logger.info("Cleaning up old models to ensure fresh training...")
        self.clean_old_models()

        logger.info("Training new ensemble models...")

        # Prepare training data
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.feature_pipeline.prepare_training_data(
                feature_data,
                prediction_horizon=PREDICTION_CONFIG['horizon_days']
            )

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}")

        # Create ensemble models
        input_dim = X_train.shape[1]
        self.models = create_ensemble_models(
            input_dim=input_dim,
            hidden_dims=MODEL_CONFIG['hidden_sizes'],
            dropout_rate=MODEL_CONFIG['dropout_rate'],
            num_models=MODEL_CONFIG['ensemble_size']
        )

        # Train ensemble
        self.trainers, self.calibrator = train_ensemble(
            X_train.values, y_train.values,
            X_val.values, y_val.values,
            self.models,
            batch_size=MODEL_CONFIG['batch_size'],
            epochs=MODEL_CONFIG['epochs'],
            save_dir=MODEL_DIR
        )

        # Evaluate on test set
        self._evaluate_models(X_test.values, y_test.values)

        logger.info("✅ Model training complete - all models have been updated")

    def _load_models(self, feature_data: Dict[str, pd.DataFrame]):
        """Load pre-trained models."""
        # Get input dimension
        first_symbol = list(feature_data.keys())[0]
        input_dim = len(self.feature_pipeline.all_features)

        from models.neural_networks import (
            ModelTrainer, ConfidenceCalibrator,
            DeepNeuralNetwork, AttentionNeuralNetwork,
            LSTMNeuralNetwork, ConvolutionalNeuralNetwork,
            BaseNeuralNetwork
        )

        self.models = []
        self.trainers = []
        self.calibrator = ConfidenceCalibrator()

        # Architecture mapping
        architecture_map = {
            'DeepNeuralNetwork': DeepNeuralNetwork,
            'AttentionNeuralNetwork': AttentionNeuralNetwork,
            'LSTMNeuralNetwork': LSTMNeuralNetwork,
            'ConvolutionalNeuralNetwork': ConvolutionalNeuralNetwork,
            'BaseNeuralNetwork': BaseNeuralNetwork
        }

        # Default architectures for each model position
        default_architectures = [
            DeepNeuralNetwork,
            AttentionNeuralNetwork,
            LSTMNeuralNetwork,
            ConvolutionalNeuralNetwork,
            BaseNeuralNetwork
        ]

        models_loaded = 0
        for i in range(MODEL_CONFIG['ensemble_size']):
            model_path = MODEL_DIR / f"model_{i + 1}.pt"
            if model_path.exists():
                try:
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=device)

                    # Get model architecture and dimensions from checkpoint
                    if 'architecture' in checkpoint and 'hidden_dims' in checkpoint:
                        arch_name = checkpoint['architecture']
                        hidden_dims = checkpoint['hidden_dims']
                        arch_class = architecture_map.get(arch_name, BaseNeuralNetwork)
                    else:
                        # Fallback: use default architecture for this position
                        arch_class = default_architectures[i % len(default_architectures)]
                        # Use default hidden dimensions
                        hidden_dims = MODEL_CONFIG['hidden_sizes']
                        arch_name = arch_class.__name__

                        # For AttentionNeuralNetwork, ensure first dim is divisible by 8
                        if arch_class == AttentionNeuralNetwork:
                            hidden_dims = hidden_dims.copy()
                            hidden_dims[0] = ((hidden_dims[0] + 7) // 8) * 8

                    # Create model with correct architecture
                    model = arch_class(input_dim, hidden_dims, MODEL_CONFIG['dropout_rate'])

                    # Try to load state dict
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    except RuntimeError as e:
                        # If dimensions don't match, skip this model
                        logger.warning(f"Model {i + 1} architecture mismatch: {e}")
                        logger.warning(f"Skipping model {i + 1}, will retrain if needed")
                        continue

                    model.to(device)
                    model.eval()

                    # Create trainer
                    trainer = ModelTrainer(model)
                    trainer.model = model

                    self.models.append(model)
                    self.trainers.append(trainer)
                    models_loaded += 1

                    logger.info(f"Loaded model {i + 1} ({arch_name}) from {model_path}")

                except Exception as e:
                    logger.error(f"Error loading model {i + 1}: {e}")
                    continue
            else:
                logger.warning(f"Model file not found: {model_path}")

        if models_loaded == 0:
            raise ValueError("No trained models found. Please run training first.")

        logger.info(f"Loaded {models_loaded} models successfully")

    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate models on test set."""
        logger.info("Evaluating models on test set...")

        # Make predictions
        predictions, confidences = make_ensemble_predictions(
            self.trainers,
            X_test,
            self.calibrator
        )

        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)

        # Directional accuracy
        pred_direction = np.sign(predictions)
        true_direction = np.sign(y_test)
        directional_accuracy = np.mean(pred_direction == true_direction)

        # Profitable predictions (above threshold)
        min_return = PREDICTION_CONFIG['min_predicted_return']
        profitable_preds = predictions > min_return
        profitable_accuracy = np.mean(
            y_test[profitable_preds] > min_return
        ) if profitable_preds.any() else 0

        logger.info(f"Test Set Performance:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2%}")
        logger.info(f"  Profitable Prediction Accuracy: {profitable_accuracy:.2%}")
        logger.info(f"  Mean Confidence: {np.mean(confidences):.3f}")

    def generate_predictions(
            self,
            feature_data: Optional[Dict[str, pd.DataFrame]] = None,
            save_to_db: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions for all symbols.

        Args:
            feature_data: Pre-calculated features (fetches latest if None)
            save_to_db: Whether to save predictions to database

        Returns:
            DataFrame with predictions
        """
        logger.info("Generating predictions...")

        # Load models if not already loaded
        if self.trainers is None:
            logger.info("Loading saved models...")
            try:
                # Get feature data if not provided
                if feature_data is None:
                    feature_data = self.collect_and_prepare_data()

                # Try to load existing models
                self._load_models(feature_data)

            except Exception as e:
                logger.error(f"Could not load models: {e}")
                raise ValueError("Models not trained. Run train_models() first.")

        # Get latest features if not provided
        if feature_data is None:
            # Fetch latest data
            collector = DataCollector()
            latest_data = collector.fetch_latest_data(
                symbols=self.symbols,
                lookback_days=60  # Need history for features
            )

            # Calculate features
            from features.base_features import engineer_base_features
            from features.interactions import engineer_interaction_features

            base_features = engineer_base_features(latest_data)
            feature_data = engineer_interaction_features(base_features)

        # Generate predictions for each symbol
        predictions_list = []

        for symbol, features in feature_data.items():
            if features.empty or symbol == 'SPY':  # Skip market benchmark
                continue

            # Get latest features
            latest_features = features[self.feature_pipeline.all_features].iloc[-1:].values

            # Scale features if scaler exists
            if symbol in self.feature_pipeline.scalers:
                scaler = self.feature_pipeline.scalers[symbol]
                latest_features = scaler.transform(latest_features)

            # Make prediction
            pred, conf = make_ensemble_predictions(
                self.trainers,
                latest_features,
                self.calibrator
            )

            # Get current price from raw data if available
            current_price = None
            if hasattr(self.feature_pipeline, 'raw_data') and symbol in self.feature_pipeline.raw_data:
                current_price = self.feature_pipeline.raw_data[symbol]['close'].iloc[-1]

            # Create prediction record
            prediction_date = features.index[-1]
            target_date = prediction_date + timedelta(days=PREDICTION_CONFIG['horizon_days'])

            pred_record = {
                'symbol': symbol,
                'prediction_date': prediction_date,
                'target_date': target_date,
                'predicted_return': pred[0],
                'confidence': conf[0],
                'current_price': current_price
            }

            predictions_list.append(pred_record)

            # Save to database
            if save_to_db:
                self.db_manager.save_prediction(
                    symbol=symbol,
                    prediction_date=prediction_date,
                    target_date=target_date,
                    predicted_return=pred[0],
                    confidence=conf[0],
                    model_version=f"ensemble_v{MODEL_CONFIG.get('version', '1.0')}"
                )

        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_list)

        if predictions_df.empty:
            logger.warning("No predictions generated")
            return predictions_df

        # Sort by predicted return
        predictions_df = predictions_df.sort_values('predicted_return', ascending=False)

        # Filter by confidence and minimum return
        high_confidence = predictions_df[
            (predictions_df['confidence'] >= PREDICTION_CONFIG['confidence_threshold']) &
            (predictions_df['predicted_return'] >= PREDICTION_CONFIG['min_predicted_return'])
            ]

        logger.info(f"Generated {len(predictions_df)} predictions")
        logger.info(f"High confidence predictions: {len(high_confidence)}")

        return predictions_df

    def run_backtest(
            self,
            start_date: datetime,
            end_date: datetime,
            feature_data: Optional[Dict[str, pd.DataFrame]] = None,
            walk_forward: bool = False
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with no data leakage.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            feature_data: Pre-calculated features
            walk_forward: Whether to use walk-forward analysis

        Returns:
            Backtest results
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")

        if self.trainers is None:
            # Try to load models
            if feature_data is None:
                feature_data = self.collect_and_prepare_data(start_date, end_date)

            try:
                self._load_models(feature_data)
            except:
                raise ValueError("Models not trained. Run train_models() first.")

        # Import backtesting engine
        from backtesting.engine import BacktestEngine, run_walk_forward_analysis

        # Get feature data if not provided
        if feature_data is None:
            feature_data = self.collect_and_prepare_data(start_date, end_date)

        if walk_forward:
            # Run walk-forward analysis
            results = run_walk_forward_analysis(
                feature_data,
                initial_train_months=24,
                test_months=3,
                retrain_frequency=3,
                start_date=start_date,
                end_date=end_date
            )
        else:
            # Single backtest with proper temporal separation
            # Determine train/test split
            total_days = (end_date - start_date).days
            train_days = int(total_days * 0.7)  # 70% for training

            train_end = start_date + timedelta(days=train_days)
            test_start = train_end + timedelta(days=BACKTEST_CONFIG['temporal_gap_days'])

            logger.info(f"Training period: {start_date} to {train_end}")
            logger.info(f"Testing period: {test_start} to {end_date}")

            # Create backtesting engine
            engine = BacktestEngine(
                initial_capital=BACKTEST_CONFIG['initial_capital'],
                position_size=BACKTEST_CONFIG['position_size'],
                max_positions=BACKTEST_CONFIG['max_positions'],
                stop_loss=BACKTEST_CONFIG['stop_loss'],
                take_profit=BACKTEST_CONFIG['take_profit'],
                commission=BACKTEST_CONFIG['commission'],
                slippage=BACKTEST_CONFIG['slippage'],
                temporal_gap=BACKTEST_CONFIG['temporal_gap_days']
            )

            # Run backtest
            results = engine.run_backtest(
                feature_data,
                self.trainers,
                self.calibrator,
                self.feature_pipeline.scalers,
                train_end,
                test_start,
                end_date,
                PREDICTION_CONFIG['horizon_days']
            )

            # Plot results
            engine.plot_results(results)

        return results


def main():
    """Main entry point for the trading system."""
    parser = argparse.ArgumentParser(description='GPU-Accelerated Trading System')
    parser.add_argument('--symbols', nargs='+', help='List of symbols to trade')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward analysis')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force-refresh', action='store_true', help='Force data refresh')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None

    # Initialize system
    system = TradingSystem(symbols=args.symbols)
    system.setup_database()

    # Collect data
    feature_data = system.collect_and_prepare_data(
        start_date=start_date,
        end_date=end_date,
        force_refresh=args.force_refresh
    )

    # Train models if requested
    if args.train:
        system.train_models(feature_data, retrain=args.retrain)

    # Generate predictions if requested
    if args.predict:
        predictions = system.generate_predictions(feature_data)

        if not predictions.empty:
            # Display top predictions
            print("\nTop Predictions:")
            print(predictions.head(20))

            # Save to CSV
            predictions.to_csv('predictions.csv', index=False)
            print(f"\nPredictions saved to predictions.csv")

            # Show summary
            print(f"\nSummary:")
            print(f"Total predictions: {len(predictions)}")
            print(f"Average predicted return: {predictions['predicted_return'].mean():.2%}")
            print(
                f"Best prediction: {predictions.iloc[0]['symbol']} with {predictions.iloc[0]['predicted_return']:.2%} return")

            # Show high confidence trades
            high_conf = predictions[predictions['confidence'] >= 0.7]
            if not high_conf.empty:
                print(f"\nHigh confidence predictions (>70%):")
                print(high_conf[['symbol', 'predicted_return', 'confidence', 'current_price']].head(10))

    # Run backtest if requested
    if args.backtest:
        if not start_date or not end_date:
            # Default to last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)

        results = system.run_backtest(
            start_date,
            end_date,
            feature_data,
            walk_forward=args.walk_forward
        )

        if not args.walk_forward:
            print("\nBacktest Results:")
            for metric, value in results.items():
                if not isinstance(value, (pd.DataFrame, pd.Series)):
                    print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
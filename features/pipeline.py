# features/pipeline.py
"""
Complete feature engineering pipeline that orchestrates data collection,
base feature calculation, and interaction engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from pathlib import Path

from config.settings import WATCHLIST, DATA_DIR, MODEL_DIR
from data.collector import DataCollector, fetch_data_for_training
from data.database import DatabaseManager
from features.base_features import BaseFeatureCalculator, engineer_base_features
from features.interactions import FeatureInteractionCalculator, engineer_interaction_features

# Set up logging
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Complete pipeline for feature engineering including:
    - Data collection
    - Base feature calculation
    - Interaction engineering
    - Feature scaling and normalization
    - Train/validation/test splitting
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize the feature engineering pipeline.

        Args:
            symbols: List of symbols to process (uses WATCHLIST if None)
        """
        self.symbols = symbols or WATCHLIST
        self.data_collector = DataCollector()
        self.base_calculator = BaseFeatureCalculator()
        self.interaction_calculator = FeatureInteractionCalculator()
        self.db_manager = DatabaseManager()

        # Feature names
        self.base_features = self.base_calculator.feature_names
        self.interaction_features = self.interaction_calculator.interaction_names
        self.all_features = self.base_features + self.interaction_features

        # Scalers for normalization
        self.scalers = {}

        # Store raw data for target calculation
        self.raw_data = {}

    def run_full_pipeline(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            save_to_db: bool = True,
            scale_features: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete feature engineering pipeline.

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            save_to_db: Whether to save features to database
            scale_features: Whether to scale/normalize features

        Returns:
            Dictionary mapping symbols to feature DataFrames
        """
        logger.info("Starting feature engineering pipeline")

        # Step 1: Collect raw data
        logger.info("Step 1: Collecting raw data")
        raw_data = self._collect_data(start_date, end_date)

        # Step 2: Calculate base features
        logger.info("Step 2: Calculating base features")
        base_features = self._calculate_base_features(raw_data)

        # Step 3: Engineer interactions
        logger.info("Step 3: Engineering feature interactions")
        all_features = self._engineer_interactions(base_features)

        # Step 4: Scale features if requested
        if scale_features:
            logger.info("Step 4: Scaling features")
            all_features = self._scale_features(all_features)

        # Step 5: Save to database if requested
        if save_to_db:
            logger.info("Step 5: Saving features to database")
            self._save_features_to_db(all_features)

        logger.info("Feature engineering pipeline completed")
        return all_features

    def _collect_data(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Collect raw data for all symbols."""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years default

        # Fetch data
        raw_data = self.data_collector.fetch_historical_data(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date
        )

        # Validate data
        clean_data = self.data_collector.validate_data(raw_data)

        # Calculate returns
        for symbol in clean_data:
            clean_data[symbol] = self.data_collector.calculate_returns(clean_data[symbol])

        # Add market data
        clean_data = self.data_collector.add_market_data(clean_data)

        # Store raw data for later use in target calculation
        self.raw_data = clean_data.copy()

        logger.info(f"Collected data for {len(clean_data)} symbols")
        return clean_data

    def _calculate_base_features(
            self,
            raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Calculate base features for all symbols."""
        return engineer_base_features(raw_data, save_to_db=False)

    def _engineer_interactions(
            self,
            base_features: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Engineer interaction features."""
        return engineer_interaction_features(base_features)

    def _scale_features(
            self,
            feature_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Scale features using RobustScaler (handles outliers better).

        Args:
            feature_data: Dictionary of feature DataFrames

        Returns:
            Dictionary of scaled feature DataFrames
        """
        scaled_data = {}

        for symbol, features in feature_data.items():
            # Create scaler for this symbol
            scaler = RobustScaler()

            # Fit and transform features
            scaled_features = features.copy()
            feature_values = scaler.fit_transform(features.values)
            scaled_features.iloc[:, :] = feature_values

            # Store scaler for later use
            self.scalers[symbol] = scaler

            # Save scaler to disk
            scaler_path = MODEL_DIR / f"scaler_{symbol}.pkl"
            joblib.dump(scaler, scaler_path)

            scaled_data[symbol] = scaled_features

        return scaled_data

    def _save_features_to_db(
            self,
            feature_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Save features to database."""
        for symbol, features in feature_data.items():
            try:
                for timestamp, row in features.iterrows():
                    self.db_manager.save_features(
                        symbol=symbol,
                        timestamp=timestamp,
                        features=row.to_dict()
                    )
            except Exception as e:
                logger.error(f"Error saving features for {symbol}: {e}")

    def prepare_training_data(
            self,
            feature_data: Dict[str, pd.DataFrame],
            prediction_horizon: int = 21,
            train_end_date: Optional[datetime] = None,
            val_split: float = 0.15,
            test_split: float = 0.15,
            temporal_gap: int = 7
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training with proper temporal splits.

        Args:
            feature_data: Dictionary of feature DataFrames
            prediction_horizon: Days ahead to predict
            train_end_date: End date for training data
            val_split: Fraction of data for validation
            test_split: Fraction of data for test
            temporal_gap: Days between train and validation/test

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        all_features = []
        all_targets = []
        all_symbols = []
        all_dates = []

        # Combine all symbols' data
        for symbol, features in feature_data.items():
            if symbol == 'SPY':  # Skip market benchmark
                continue

            # Get close prices from raw data if available
            if hasattr(self, 'raw_data') and symbol in self.raw_data:
                raw_df = self.raw_data[symbol]
                # Align indices
                close_prices = raw_df['close'].reindex(features.index)
                # Calculate forward returns as targets
                target = close_prices.pct_change(prediction_horizon).shift(-prediction_horizon)
            else:
                # Fallback: use returns_1d to approximate
                if 'returns_1d' in features.columns:
                    # Sum of returns over prediction horizon (approximation)
                    target = features['returns_1d'].rolling(window=prediction_horizon).sum().shift(-prediction_horizon)
                else:
                    logger.warning(f"No price data available for {symbol}, skipping")
                    continue

            features = features.copy()
            features['target'] = target

            # Remove last rows without targets
            features = features[:-prediction_horizon]

            # Remove any rows with NaN
            features = features.dropna()

            if len(features) > 252:  # At least 1 year of data
                all_features.append(features[self.all_features])
                all_targets.append(features['target'])
                all_symbols.extend([symbol] * len(features))
                all_dates.extend(features.index.tolist())

        if not all_features:
            raise ValueError("No valid data available for training")

        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        dates = pd.Series(all_dates)
        symbols = pd.Series(all_symbols)

        # Sort by date
        sort_idx = dates.argsort()
        X = X.iloc[sort_idx]
        y = y.iloc[sort_idx]
        dates = dates.iloc[sort_idx]
        symbols = symbols.iloc[sort_idx]

        # Determine split dates
        if train_end_date is None:
            # Use 70% for training, 15% validation, 15% test
            total_days = (dates.max() - dates.min()).days
            train_days = int(total_days * (1 - val_split - test_split))
            val_days = int(total_days * val_split)

            train_end_date = dates.min() + timedelta(days=train_days)
            val_end_date = train_end_date + timedelta(days=val_days + temporal_gap)
        else:
            val_end_date = train_end_date + timedelta(days=temporal_gap + 30)  # 30 days for validation

        # Create temporal splits with gaps
        train_mask = dates <= train_end_date
        val_mask = (dates > train_end_date + timedelta(days=temporal_gap)) & (dates <= val_end_date)
        test_mask = dates > val_end_date + timedelta(days=temporal_gap)

        # Split data
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]

        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

        # Log split information
        logger.info(f"Training data: {len(X_train)} samples, "
                    f"dates: {dates[train_mask].min()} to {dates[train_mask].max()}")
        logger.info(f"Validation data: {len(X_val)} samples, "
                    f"dates: {dates[val_mask].min()} to {dates[val_mask].max()}")
        logger.info(f"Test data: {len(X_test)} samples, "
                    f"dates: {dates[test_mask].min()} to {dates[test_mask].max()}")

        # Store metadata for later use
        self.train_symbols = symbols[train_mask]
        self.val_symbols = symbols[val_mask]
        self.test_symbols = symbols[test_mask]

        self.train_dates = dates[train_mask]
        self.val_dates = dates[val_mask]
        self.test_dates = dates[test_mask]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_statistics(
            self,
            feature_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate statistics for all features across all symbols.

        Args:
            feature_data: Dictionary of feature DataFrames

        Returns:
            DataFrame with feature statistics
        """
        all_features = pd.concat(list(feature_data.values()))

        stats = pd.DataFrame({
            'mean': all_features.mean(),
            'std': all_features.std(),
            'min': all_features.min(),
            'max': all_features.max(),
            'skew': all_features.skew(),
            'kurtosis': all_features.kurtosis(),
            'null_count': all_features.isna().sum(),
            'zero_count': (all_features == 0).sum()
        })

        return stats

    def validate_features(
            self,
            feature_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[str]]:
        """
        Validate features and identify potential issues.

        Args:
            feature_data: Dictionary of feature DataFrames

        Returns:
            Dictionary of issues found
        """
        issues = {
            'missing_features': [],
            'constant_features': [],
            'high_correlation': [],
            'infinite_values': []
        }

        for symbol, features in feature_data.items():
            # Check for missing features
            missing = set(self.all_features) - set(features.columns)
            if missing:
                issues['missing_features'].append(f"{symbol}: {missing}")

            # Check for constant features
            constant = features.columns[features.std() == 0].tolist()
            if constant:
                issues['constant_features'].append(f"{symbol}: {constant}")

            # Check for infinite values
            inf_cols = features.columns[np.isinf(features).any()].tolist()
            if inf_cols:
                issues['infinite_values'].append(f"{symbol}: {inf_cols}")

        # Check for highly correlated features (using first symbol as example)
        if feature_data:
            first_symbol = list(feature_data.keys())[0]
            corr_matrix = feature_data[first_symbol].corr().abs()

            # Find features with correlation > 0.95
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(
                            f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: "
                            f"{corr_matrix.iloc[i, j]:.3f}"
                        )

            if high_corr_pairs:
                issues['high_correlation'] = high_corr_pairs

        return issues


# Convenience functions
def create_features_for_training(
        symbols: Optional[List[str]] = None,
        years_back: int = 2,
        scale: bool = True,
        save_to_db: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create features and prepare for training.

    Args:
        symbols: List of symbols (uses WATCHLIST if None)
        years_back: Number of years of historical data
        scale: Whether to scale features
        save_to_db: Whether to save to database

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(symbols)

    # Run feature engineering
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)

    feature_data = pipeline.run_full_pipeline(
        start_date=start_date,
        end_date=end_date,
        save_to_db=save_to_db,
        scale_features=scale
    )

    # Prepare training data
    return pipeline.prepare_training_data(feature_data)


def load_features_from_db(
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load pre-calculated features from database.

    Args:
        symbols: List of symbols to load
        start_date: Start date for features
        end_date: End date for features

    Returns:
        Dictionary mapping symbols to feature DataFrames
    """
    db = DatabaseManager()
    feature_data = {}

    symbols = symbols or WATCHLIST

    for symbol in symbols:
        features = db.get_features(symbol, start_date, end_date)
        if not features.empty:
            feature_data[symbol] = features

    return feature_data


if __name__ == "__main__":
    # Test the complete pipeline
    print("Testing complete feature engineering pipeline...")

    # Use a small subset for testing
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(test_symbols)

    # Run pipeline
    feature_data = pipeline.run_full_pipeline(
        save_to_db=False,
        scale_features=True
    )

    # Display results
    print(f"\nProcessed {len(feature_data)} symbols")

    for symbol, features in feature_data.items():
        print(f"\n{symbol}:")
        print(f"  Feature shape: {features.shape}")
        print(f"  Date range: {features.index.min()} to {features.index.max()}")
        print(f"  Features: {len(features.columns)}")

    # Get feature statistics
    stats = pipeline.get_feature_statistics(feature_data)
    print("\nFeature Statistics:")
    print(stats.head(10))

    # Validate features
    issues = pipeline.validate_features(feature_data)
    print("\nValidation Results:")
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"\n{issue_type}:")
            for issue in issue_list[:5]:  # Show first 5
                print(f"  - {issue}")

    # Test train/val/test split
    print("\nTesting data splitting...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(feature_data)

        print(f"\nData shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    except Exception as e:
        print(f"Error in data splitting: {e}")
# config/always_on_config.py
"""
Configuration for the always-on trading system.
Controls behavior during market hours vs. market closed periods.
"""

from datetime import time

# Market Hours Configuration
MARKET_HOURS = {
    'open_time': time(9, 30),  # 9:30 AM
    'close_time': time(16, 0),  # 4:00 PM
    'timezone': 'US/Eastern',
    'trading_days': [0, 1, 2, 3, 4]  # Monday-Friday (0=Monday, 6=Sunday)
}

# Data Refresh Configuration
DATA_REFRESH = {
    'market_hours': {
        'interval_minutes': 5,  # Every 5 minutes when market is open
        'enable_realtime': True,  # Use real-time data feeds
        'sources': ['alpaca', 'yfinance']
    },
    'market_closed': {
        'interval_hours': 1,  # Every hour when market is closed
        'enable_realtime': False,  # Use delayed/cached data
        'sources': ['yfinance']  # Fallback to free sources
    }
}

# Prediction Updates Configuration
PREDICTIONS = {
    'market_hours': {
        'interval_minutes': 10,  # Every 10 minutes when market is open
        'min_confidence': 0.70,  # Higher confidence required for trading
        'min_return': 0.02,  # 2% minimum predicted return
        'max_positions': 10  # Maximum concurrent positions
    },
    'market_closed': {
        'interval_hours': 2,  # Every 2 hours when market is closed
        'min_confidence': 0.60,  # Lower confidence for analysis
        'min_return': 0.01,  # 1% minimum for screening
        'generate_for_next_day': True  # Prepare trades for market open
    }
}

# News Analysis Configuration
NEWS_ANALYSIS = {
    'market_hours': {
        'interval_minutes': 30,  # Every 30 minutes when market is open
        'hours_lookback': 2,  # Look back 2 hours for recent news
        'impact_threshold': 0.3,  # Minimum impact score to act on
        'enable_real_time': True  # Process breaking news immediately
    },
    'market_closed': {
        'interval_hours': 4,  # Every 4 hours when market is closed
        'hours_lookback': 24,  # Look back 24 hours for comprehensive analysis
        'impact_threshold': 0.2,  # Lower threshold for overnight analysis
        'enable_real_time': False  # No immediate action needed
    },
    'sources': [
        'yahoo_finance',
        'bloomberg_rss',
        'reuters_rss',
        'cnbc_rss',
        'marketwatch_rss',
        'techcrunch_rss',
        'wsj_rss'
    ]
}

# Dashboard Configuration
DASHBOARD = {
    'auto_refresh': {
        'interval_seconds': 30,  # Refresh every 30 seconds
        'smart_refresh': True,  # Faster refresh during market hours
        'market_hours_interval': 15,  # 15 seconds during market hours
        'market_closed_interval': 60  # 60 seconds when market closed
    },
    'data_retention': {
        'prediction_files': 10,  # Keep last 10 prediction files
        'news_files': 20,  # Keep last 20 news analysis files
        'state_history_days': 7  # Keep 7 days of state history
    },
    'features': {
        'show_paper_trading_banner': True,
        'enable_notifications': True,
        'show_market_countdown': True,
        'display_extended_hours': False,
        'enable_dark_mode': True
    }
}

# Risk Management Configuration
RISK_MANAGEMENT = {
    'always_on': {
        'portfolio_heat_limit': 0.10,  # 10% maximum portfolio heat
        'max_position_size': 0.05,  # 5% maximum single position
        'max_correlation': 0.70,  # 70% maximum correlation
        'daily_loss_limit': 0.02,  # 2% daily loss limit
        'stop_loss_default': 0.05,  # 5% default stop loss
        'take_profit_default': 0.15  # 15% default take profit
    },
    'market_hours': {
        'enable_dynamic_sizing': True,  # Adjust position sizes based on volatility
        'enable_intraday_stops': True,  # Allow intraday stop losses
        'max_trades_per_day': 20  # Maximum trades per day
    },
    'market_closed': {
        'enable_dynamic_sizing': False,  # Static analysis only
        'enable_intraday_stops': False,  # No position changes
        'prepare_exit_orders': True  # Prepare orders for market open
    }
}

# System Health Monitoring
HEALTH_MONITORING = {
    'checks': {
        'interval_minutes': 15,  # Health check every 15 minutes
        'restart_dashboard_on_failure': True,
        'alert_on_data_staleness': True,
        'max_data_age_hours': 2  # Alert if data older than 2 hours
    },
    'alerts': {
        'enable_email': False,  # Email alerts disabled by default
        'enable_logging': True,  # Log all alerts
        'critical_threshold': 3,  # Number of failures before critical alert
        'email_recipients': [],  # Add email addresses for alerts
        'log_level': 'INFO'
    }
}

# Performance Optimization
PERFORMANCE = {
    'caching': {
        'enable_feature_cache': True,  # Cache computed features
        'cache_duration_minutes': 30,  # Cache for 30 minutes
        'enable_prediction_cache': True,  # Cache predictions
        'prediction_cache_minutes': 10  # Cache predictions for 10 minutes
    },
    'concurrency': {
        'max_worker_threads': 4,  # Maximum background worker threads
        'enable_async_news': True,  # Run news analysis asynchronously
        'enable_async_predictions': True,  # Run predictions asynchronously
        'batch_size_symbols': 50  # Process symbols in batches of 50
    }
}

# Database Configuration
DATABASE = {
    'connection': {
        'type': 'sqlite',  # Use SQLite for simplicity
        'path': 'data/trading_system.db',
        'enable_wal_mode': True,  # Enable WAL mode for better concurrency
        'pool_size': 10
    },
    'retention': {
        'trade_history_days': 365,  # Keep trade history for 1 year
        'price_data_days': 180,  # Keep price data for 6 months
        'news_data_days': 90,  # Keep news data for 3 months
        'feature_data_days': 90  # Keep feature data for 3 months
    }
}

# API Configuration
API_CONFIG = {
    'alpaca': {
        'timeout_seconds': 30,
        'retry_attempts': 3,
        'rate_limit_per_minute': 200
    },
    'openai': {
        'timeout_seconds': 60,
        'retry_attempts': 2,
        'max_tokens_per_request': 500,
        'model': 'gpt-4o-mini'  # Use 4o-mini for cost efficiency
    },
    'yfinance': {
        'timeout_seconds': 15,
        'retry_attempts': 3,
        'delay_between_requests': 0.1  # 100ms delay between requests
    }
}

# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'files': {
        'main_log': 'logs/always_on_trading.log',
        'trading_log': 'logs/trading_activity.log',
        'news_log': 'logs/news_analysis.log',
        'error_log': 'logs/errors.log'
    },
    'rotation': {
        'max_file_size_mb': 100,  # Rotate when file exceeds 100MB
        'backup_count': 5  # Keep 5 backup files
    }
}

# Feature Flags
FEATURE_FLAGS = {
    'enable_news_analysis': True,  # Enable news sentiment analysis
    'enable_social_sentiment': False,  # Disable social media sentiment (not implemented)
    'enable_options_data': False,  # Disable options data analysis
    'enable_crypto_analysis': False,  # Disable cryptocurrency analysis
    'enable_forex_analysis': False,  # Disable forex analysis
    'enable_commodities': False,  # Disable commodities analysis
    'enable_experimental_features': False  # Disable experimental features
}

# Market Data Sources Priority
DATA_SOURCES = {
    'primary': 'alpaca',  # Primary data source
    'fallback': ['yfinance', 'alphavantage'],  # Fallback sources in order
    'news_sources': [
        'yahoo_finance',
        'reuters',
        'bloomberg',
        'cnbc',
        'marketwatch'
    ]
}

# Development/Testing Configuration
DEVELOPMENT = {
    'enable_debug_mode': False,  # Enable debug logging
    'enable_paper_trading_only': True,  # Force paper trading
    'enable_backtesting_mode': False,  # Enable backtesting mode
    'mock_market_hours': False,  # Mock market hours for testing
    'simulate_market_data': False,  # Use simulated data for testing
    'enable_profiling': False  # Enable performance profiling
}


def get_config_for_market_status(market_open: bool) -> dict:
    """
    Get configuration appropriate for current market status.

    Args:
        market_open: Whether market is currently open

    Returns:
        Configuration dictionary for current market status
    """
    if market_open:
        return {
            'data_refresh_minutes': DATA_REFRESH['market_hours']['interval_minutes'],
            'prediction_update_minutes': PREDICTIONS['market_hours']['interval_minutes'],
            'news_update_minutes': NEWS_ANALYSIS['market_hours']['interval_minutes'],
            'dashboard_refresh_seconds': DASHBOARD['auto_refresh']['market_hours_interval'],
            'enable_trading': True,
            'enable_position_changes': True,
            'min_confidence': PREDICTIONS['market_hours']['min_confidence'],
            'min_return': PREDICTIONS['market_hours']['min_return']
        }
    else:
        return {
            'data_refresh_minutes': DATA_REFRESH['market_closed']['interval_hours'] * 60,
            'prediction_update_minutes': PREDICTIONS['market_closed']['interval_hours'] * 60,
            'news_update_minutes': NEWS_ANALYSIS['market_closed']['interval_hours'] * 60,
            'dashboard_refresh_seconds': DASHBOARD['auto_refresh']['market_closed_interval'],
            'enable_trading': False,
            'enable_position_changes': False,
            'min_confidence': PREDICTIONS['market_closed']['min_confidence'],
            'min_return': PREDICTIONS['market_closed']['min_return']
        }


def is_market_hours() -> bool:
    """
    Check if current time is within market hours.

    Returns:
        True if market is open, False otherwise
    """
    from datetime import datetime
    import pytz

    # Get current time in market timezone
    tz = pytz.timezone(MARKET_HOURS['timezone'])
    now = datetime.now(tz)

    # Check if it's a trading day
    if now.weekday() not in MARKET_HOURS['trading_days']:
        return False

    # Check if it's within trading hours
    current_time = now.time()
    return MARKET_HOURS['open_time'] <= current_time <= MARKET_HOURS['close_time']


def get_next_market_open() -> datetime:
    """
    Get the next market open time.

    Returns:
        Datetime of next market open
    """
    from datetime import datetime, timedelta
    import pytz

    tz = pytz.timezone(MARKET_HOURS['timezone'])
    now = datetime.now(tz)

    # Start with today
    candidate = now.replace(
        hour=MARKET_HOURS['open_time'].hour,
        minute=MARKET_HOURS['open_time'].minute,
        second=0,
        microsecond=0
    )

    # If market already opened today, move to next trading day
    if candidate <= now:
        candidate += timedelta(days=1)

    # Find next trading day
    while candidate.weekday() not in MARKET_HOURS['trading_days']:
        candidate += timedelta(days=1)

    return candidate


def get_time_to_market_open() -> timedelta:
    """
    Get time remaining until market opens.

    Returns:
        Timedelta until market open
    """
    from datetime import datetime
    import pytz

    tz = pytz.timezone(MARKET_HOURS['timezone'])
    now = datetime.now(tz)
    next_open = get_next_market_open()

    return next_open - now


# Validation function
def validate_config():
    """Validate configuration settings."""
    errors = []

    # Check that intervals are reasonable
    if DATA_REFRESH['market_hours']['interval_minutes'] < 1:
        errors.append("Data refresh interval too short during market hours")

    if PREDICTIONS['market_hours']['interval_minutes'] < 5:
        errors.append("Prediction update interval too short during market hours")

    if NEWS_ANALYSIS['market_hours']['interval_minutes'] < 15:
        errors.append("News analysis interval too short during market hours")

    # Check risk management limits
    if RISK_MANAGEMENT['always_on']['portfolio_heat_limit'] > 0.5:
        errors.append("Portfolio heat limit too high (>50%)")

    if RISK_MANAGEMENT['always_on']['max_position_size'] > 0.2:
        errors.append("Maximum position size too high (>20%)")

    # Check performance settings
    if PERFORMANCE['concurrency']['max_worker_threads'] > 8:
        errors.append("Too many worker threads (>8)")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    return True


# Run validation on import
validate_config()
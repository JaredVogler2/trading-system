# config/settings.py
"""
Configuration settings for the algorithmic trading system.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "trading_system"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "password")
}

# Original Stock Watchlist
STOCK_WATCHLIST = [
    # Top tech stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "NFLX", "AVGO", "CSCO", "QCOM", "TXN", "IBM", "NOW", "UBER",

    # Financial sector
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "AXP", "BLK",
    "SCHW", "COF", "SPGI", "CME", "ICE", "V", "MA", "PYPL", "SQ", "COIN",

    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "CVS", "MRK", "DHR", "AMGN",
    "GILD", "ISRG", "VRTX", "REGN", "ZTS", "BIIB", "ILMN", "IDXX", "ALGN", "DXCM",

    # Consumer
    "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "CVX", "XOM", "COP", "SLB", "DIS", "CMCSA", "VZ", "T", "TMUS", "GME",

    # Industrial & Energy
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "NOC", "DE",
    "EMR", "ETN", "ITW", "PH", "GD", "FDX", "NSC", "UNP", "CSX", "DAL",

    # Real Estate & Utilities
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "AVB", "EQR", "SPG",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",

    # Additional high-volume stocks
    "F", "GM", "RIVN", "LCID", "AAL", "UAL", "CCL", "RCL", "WYNN", "MGM",
    "DKNG", "PENN", "CHTR", "ROKU", "SNAP", "PINS", "AMD", "ZM", "DOCU", "OKTA"
]

# Market Benchmarks and Sector ETFs
MARKET_ETFS = [
    # Major Indices
    "SPY",  # S&P 500
    "QQQ",  # NASDAQ 100
    "DIA",  # Dow Jones
    "IWM",  # Russell 2000 (small caps)
    "VTI",  # Total Market

    # Sector ETFs
    "XLF",  # Financials
    "XLK",  # Technology
    "XLE",  # Energy
    "XLV",  # Healthcare
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLRE",  # Real Estate
    "XLB",  # Materials
    "XLC",  # Communication Services
]

# Leveraged ETFs (3x)
LEVERAGED_ETFS = [
    # Technology
    "TQQQ",  # 3x NASDAQ Bull
    "SQQQ",  # 3x NASDAQ Bear
    "TECL",  # 3x Tech Bull
    "TECS",  # 3x Tech Bear
    "SOXL",  # 3x Semiconductor Bull
    "SOXS",  # 3x Semiconductor Bear

    # S&P 500
    "UPRO",  # 3x S&P 500 Bull
    "SPXS",  # 3x S&P 500 Bear
    "SPXL",  # 3x S&P 500 Bull (alternative)
    "SPXU",  # 3x S&P 500 Bear (alternative)

    # Financial
    "FAS",  # 3x Financial Bull
    "FAZ",  # 3x Financial Bear

    # Energy
    "ERX",  # 3x Energy Bull
    "ERY",  # 3x Energy Bear
    "GUSH",  # 3x Oil & Gas Bull
    "DRIP",  # 3x Oil & Gas Bear

    # Real Estate
    "DRN",  # 3x Real Estate Bull
    "DRV",  # 3x Real Estate Bear

    # Biotech
    "LABU",  # 3x Biotech Bull
    "LABD",  # 3x Biotech Bear

    # Gold/Mining
    "NUGT",  # 3x Gold Miners Bull
    "DUST",  # 3x Gold Miners Bear
    "JNUG",  # 3x Junior Gold Miners Bull
    "JDST",  # 3x Junior Gold Miners Bear

    # Small Caps
    "TNA",  # 3x Russell 2000 Bull
    "TZA",  # 3x Russell 2000 Bear

    # Volatility
    "UVXY",  # 1.5x VIX
    "SVXY",  # -0.5x VIX (inverse)

    # China/Emerging Markets
    "YINN",  # 3x China Bull
    "YANG",  # 3x China Bear
    "EDC",  # 3x Emerging Markets Bull
    "EDZ",  # 3x Emerging Markets Bear
]

# Combine all lists for complete watchlist
WATCHLIST = STOCK_WATCHLIST + MARKET_ETFS + LEVERAGED_ETFS

# Feature Engineering Configuration
BASE_FEATURES = [
    "returns_1d", "returns_5d", "returns_20d",
    "price_position_10d", "price_position_20d",
    "volatility_10d", "volatility_20d",
    "price_acceleration",
    "sma_ratio_5_20", "sma_ratio_12_26",
    "ema_ratio_5_20", "ema_ratio_12_26",
    "ma_alignment", "ma_crossover",
    "rsi_14", "rsi_divergence",
    "macd_line", "macd_signal", "macd_histogram",
    "stochastic_k", "stochastic_d", "williams_r",
    "cci_20",
    "volume_ratio", "volume_trend", "volume_breakout", "volume_divergence",
    "support_level", "resistance_level", "support_strength", "resistance_strength",
    "support_distance", "resistance_distance",
    "trend_short", "trend_medium", "trend_long",
    "trend_strength", "trend_acceleration", "trend_consistency",
    "volatility_regime", "trend_regime", "volume_regime",
    "mean_reversion_signal", "momentum_signal",
    "day_of_week_effect", "month_effect", "quarter_effect",
    "downside_volatility", "max_drawdown", "var_95", "sharpe_components",
    "market_correlation", "sector_relative_strength", "beta_stability"
]

# Model Configuration
MODEL_CONFIG = {
    "ensemble_size": 5,
    "hidden_sizes": [512, 256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 100,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "initial_capital": 100000,
    "position_size": 0.02,  # 2% of capital per position
    "max_positions": 10,
    "stop_loss": 0.05,  # 5% stop loss
    "take_profit": 0.15,  # 15% take profit
    "commission": 0.001,  # 0.1% commission
    "slippage": 0.001,  # 0.1% slippage
    "temporal_gap_days": 7  # Days between training and backtest data
}

# Prediction Configuration
PREDICTION_CONFIG = {
    "horizon_days": 21,
    "confidence_threshold": 0.7,
    "min_predicted_return": 0.02,  # 2% minimum predicted return
    "rebalance_frequency": "weekly"
}

# Special handling for leveraged ETFs
LEVERAGED_SYMBOLS = set(LEVERAGED_ETFS)

# Risk configuration for different instrument types
RISK_CONFIG = {
    "regular_position_size": 0.02,  # 2% for regular stocks
    "leveraged_position_size": 0.01,  # 1% for leveraged ETFs
    "etf_position_size": 0.03,  # 3% for regular ETFs
    "max_portfolio_heat": 0.10,  # 10% max portfolio risk
    "max_correlation": 0.70,  # 70% max correlation
    "max_daily_loss": 0.02,  # 2% daily loss limit
    "kelly_fraction": 0.25,  # Use 25% of Kelly
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": LOG_DIR / "trading_system.log"
}

# GPU Configuration
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU available. Training will be slow on CPU.")


# Helper function to identify instrument type
def get_instrument_type(symbol: str) -> str:
    """Identify if symbol is a stock, ETF, or leveraged ETF."""
    if symbol in LEVERAGED_SYMBOLS:
        return "leveraged"
    elif symbol in MARKET_ETFS:
        return "etf"
    else:
        return "stock"


# Helper function to get position size for symbol
def get_position_size(symbol: str) -> float:
    """Get appropriate position size based on instrument type."""
    instrument_type = get_instrument_type(symbol)

    if instrument_type == "leveraged":
        return RISK_CONFIG["leveraged_position_size"]
    elif instrument_type == "etf":
        return RISK_CONFIG["etf_position_size"]
    else:
        return RISK_CONFIG["regular_position_size"]


# Validate configuration
def validate_config():
    """Validate that all required configuration is present."""
    errors = []

    if not ALPACA_API_KEY:
        errors.append("ALPACA_API_KEY not set in .env file")
    if not ALPACA_SECRET_KEY:
        errors.append("ALPACA_SECRET_KEY not set in .env file")

    # Print summary
    print(f"\nTrading System Configuration:")
    print(f"Total symbols: {len(WATCHLIST)}")
    print(f"  - Stocks: {len(STOCK_WATCHLIST)}")
    print(f"  - Market ETFs: {len(MARKET_ETFS)}")
    print(f"  - Leveraged ETFs: {len(LEVERAGED_ETFS)}")
    print(f"Max positions: {BACKTEST_CONFIG['max_positions']}")
    print(f"Risk per trade: {RISK_CONFIG['regular_position_size'] * 100:.1f}% (stocks), "
          f"{RISK_CONFIG['leveraged_position_size'] * 100:.1f}% (leveraged)")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))


# Run validation when module is imported
validate_config()
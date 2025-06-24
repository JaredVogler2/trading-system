# Trading System

An algorithmic trading system with machine learning capabilities for backtesting and live trading.

## Overview

This trading system implements:
- ML-based price prediction models
- Comprehensive backtesting framework
- Real-time dashboard monitoring
- Feature engineering pipeline
- Live trading capabilities with Alpaca integration

## Project Structure

- analysis/ - Analysis and reporting modules
- backtesting/ - Backtesting engine and related components
- config/ - Configuration files and settings
- data/ - Data collection and database management
- features/ - Feature engineering pipeline
- models/ - Machine learning models (neural networks, etc.)
- monitoring/ - Dashboard and monitoring tools
- news/ - News analysis and sentiment components
- trading/ - Live trading execution
- utils/ - Utility functions

## Key Features

- Walk-forward analysis for ML model validation
- Data leakage prevention in backtesting
- Real-time performance monitoring
- Multi-timeframe analysis
- Risk management systems

## Technologies Used

- Python 3.x
- PyTorch for neural networks
- Streamlit for dashboard
- SQLite for data storage
- Alpaca API for trading
- OpenAI API for analysis

## Setup

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Copy .env.example to .env and add your API keys
4. Run the system: python run_always_on_trading.py

## License

Private repository - All rights reserved

## Author

Jared Vogler

---

Last updated: 2025-06-24

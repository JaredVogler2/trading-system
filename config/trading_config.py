# config/trading_config.py
'''Configuration for live trading system'''

# Trading parameters
TRADING_CONFIG = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'AMD'],
    'max_positions': 5,
    'position_size': 0.02,  # 2% per position
    'stop_loss': 0.05,      # 5% stop loss
    'take_profit': 0.15,    # 15% take profit
    'min_confidence': 0.70,
    'min_return': 0.02,     # 2% minimum predicted return
}

# Risk management
RISK_CONFIG = {
    'max_portfolio_heat': 0.10,   # 10% max risk
    'max_correlation': 0.70,      # 70% max correlation
    'max_daily_loss': 0.02,       # 2% daily loss limit
    'kelly_fraction': 0.25,       # Use 25% of Kelly
}

# Monitoring
ALERT_CONFIG = {
    'email_alerts': False,        # Enable email alerts
    'slack_alerts': False,        # Enable Slack alerts
    'alert_email': '',           # Your email
    'slack_webhook': '',         # Slack webhook URL
}

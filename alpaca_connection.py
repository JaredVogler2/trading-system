# alpaca_connection.py
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

try:
    client = TradingClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
        paper=True
    )

    account = client.get_account()
    print("✅ Alpaca Connection Successful!")
    print(f"Account ID: {account.id}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")

except Exception as e:
    print(f"❌ Alpaca Connection Failed: {e}")
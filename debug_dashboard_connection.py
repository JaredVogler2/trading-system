# debug_dashboard_connection.py
import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

# Load environment variables
load_dotenv()

print("=== Debugging Alpaca Connection ===")
print(f"ALPACA_API_KEY exists: {bool(os.getenv('ALPACA_API_KEY'))}")
print(f"ALPACA_SECRET_KEY exists: {bool(os.getenv('ALPACA_SECRET_KEY'))}")
print(f"API Key starts with: {os.getenv('ALPACA_API_KEY', '')[:10]}...")

try:
    # Try exact same initialization as dashboard
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    print("\nAttempting TradingClient connection...")
    trading_client = TradingClient(api_key, secret_key, paper=True)
    account = trading_client.get_account()
    print(f"✅ TradingClient connected! Account: {account.account_number}")

    print("\nAttempting StockHistoricalDataClient connection...")
    data_client = StockHistoricalDataClient(api_key, secret_key)
    print("✅ StockHistoricalDataClient connected!")

except Exception as e:
    print(f"❌ Connection failed: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
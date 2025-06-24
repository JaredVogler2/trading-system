# debug_alpaca_data_fetch.py
"""Debug script to test actual Alpaca data fetching."""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd

# Load environment variables
load_dotenv()

print("=== Testing Alpaca Data Fetching ===")

# Get credentials
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

print(f"API Key exists: {bool(api_key)}")
print(f"Secret Key exists: {bool(secret_key)}")

try:
    # Test 1: Trading Client (Account Data)
    print("\n1. Testing Trading Client...")
    trading_client = TradingClient(api_key, secret_key, paper=True)

    account = trading_client.get_account()
    print(f"✅ Account connected!")
    print(f"   Account Number: {account.account_number}")
    print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
    print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    print(f"   Cash: ${float(account.cash):,.2f}")

    # Test 2: Get Positions
    print("\n2. Testing Positions...")
    positions = trading_client.get_all_positions()
    print(f"✅ Found {len(positions)} positions")
    for pos in positions[:5]:  # Show first 5
        print(f"   {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price}")

    # Test 3: Historical Data Client
    print("\n3. Testing Historical Data...")
    data_client = StockHistoricalDataClient(api_key, secret_key)

    # Get recent bars for AAPL
    request = StockBarsRequest(
        symbol_or_symbols="AAPL",
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=5),
        end=datetime.now()
    )

    bars = data_client.get_stock_bars(request)

    if 'AAPL' in bars.data:
        print(f"✅ Historical data working!")
        print(f"   Got {len(bars.data['AAPL'])} bars for AAPL")
        latest = bars.data['AAPL'][-1]
        print(f"   Latest: {latest.timestamp} - Close: ${latest.close}")

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()

print("\n=== Testing Data Fetcher Component ===")

# Test the actual data fetcher from the dashboard
try:
    from monitoring.data_fetchers import DataFetcher
    from monitoring.mock_data_tracker import MockDataTracker
    from monitoring.utility_helpers import DashboardUtilities

    mock_tracker = MockDataTracker()
    utilities = DashboardUtilities()

    fetcher = DataFetcher(trading_client, data_client, mock_tracker, utilities)

    print("\n4. Testing DataFetcher.get_account_data()...")
    account_data = fetcher.get_account_data()
    print(f"   Result: {account_data}")
    print(f"   Is Mock: {mock_tracker.is_mock('account')}")

    print("\n5. Testing DataFetcher.get_positions_data()...")
    positions_data = fetcher.get_positions_data()
    print(f"   Found {len(positions_data)} positions")
    print(f"   Is Mock: {mock_tracker.is_mock('positions')}")

except Exception as e:
    print(f"\n❌ DataFetcher Error: {e}")
    import traceback

    traceback.print_exc()
# create_helper_scripts.py
import os

# Get the current directory (should be C:\Users\jared\PycharmProjects\trading_system)
current_dir = os.getcwd()
print(f"Creating scripts in: {current_dir}")

# Create clear_dashboard_cache.py
with open('clear_dashboard_cache.py', 'w', encoding='utf-8') as f:
    f.write("""# clear_dashboard_cache.py
import os
import shutil
from pathlib import Path

print("Clearing dashboard caches...")

# Clear Streamlit cache
streamlit_cache = Path.home() / '.streamlit' / 'cache'
if streamlit_cache.exists():
    shutil.rmtree(streamlit_cache)
    print("✓ Cleared Streamlit cache")

# Clear any session state files
temp_files = [
    'temp_enhanced_dashboard_runner.py',
    'temp_comprehensive_dashboard_runner.py',  # Keep for cleanup of old files
    'temp_always_on_dashboard.py',
    'temp_full_dashboard.py',
    'temp_live_system_ref.pkl'
]

for file in temp_files:
    if os.path.exists(file):
        os.remove(file)
        print(f"✓ Removed {file}")

# Clear data directory caches
data_dir = Path('data')
if data_dir.exists():
    cache_patterns = ['*cache*', '*.tmp', '*.pkl']
    for pattern in cache_patterns:
        for file in data_dir.glob(pattern):
            if file.is_file():
                file.unlink()
                print(f"✓ Removed {file}")

print("\\nCache cleared! Please restart the dashboard.")
""")
print("✓ Created clear_dashboard_cache.py")

# Create check_prediction_files.py
with open('check_prediction_files.py', 'w', encoding='utf-8') as f:
    f.write("""# check_prediction_files.py
from pathlib import Path
import pandas as pd
from datetime import datetime

data_dir = Path('data')

print("=== Checking for prediction files ===\\n")

if not data_dir.exists():
    print("Data directory not found!")
else:
    # Look for all CSV files in data directory
    csv_files = list(data_dir.glob('*.csv'))

    if not csv_files:
        print("No CSV files found in data directory")
    else:
        print(f"Found {len(csv_files)} CSV files:\\n")

        for file in sorted(csv_files):
            # Get file info
            size = file.stat().st_size / 1024  # KB
            modified = datetime.fromtimestamp(file.stat().st_mtime)

            print(f"📄 {file.name}")
            print(f"   Size: {size:.1f} KB")
            print(f"   Modified: {modified}")

            # Try to read and show preview
            try:
                df = pd.read_csv(file)
                print(f"   Rows: {len(df)}, Columns: {df.columns.tolist()[:5]}")

                # Check if it's a prediction file
                if all(col in df.columns for col in ['symbol', 'predicted_return', 'confidence']):
                    print(f"   ✅ Valid prediction file")
                    print(f"   Top predictions: {df.nlargest(3, 'predicted_return')['symbol'].tolist()}")
            except Exception as e:
                print(f"   ❌ Error reading: {e}")

            print()
""")
print("✓ Created check_prediction_files.py")

# Create quick_generate_predictions.py
with open('quick_generate_predictions.py', 'w', encoding='utf-8') as f:
    f.write("""# quick_generate_predictions.py
from main import TradingSystem
import logging

logging.basicConfig(level=logging.INFO)

print("Generating predictions...")

try:
    # Initialize system
    system = TradingSystem()

    # Load models
    print("Loading models...")
    feature_data = system.collect_and_prepare_data()
    system._load_models(feature_data)

    # Generate predictions
    print("Generating predictions...")
    predictions = system.generate_predictions(save_to_db=True)

    if predictions is not None and not predictions.empty:
        print(f"\\n✅ Generated {len(predictions)} predictions")
        print(f"Saved to: predictions.csv")
        print(f"\\nTop 5 predictions:")
        print(predictions.head()[['symbol', 'predicted_return', 'confidence']])
    else:
        print("❌ No predictions generated")

except Exception as e:
    print(f"❌ Error: {e}")
""")
print("✓ Created quick_generate_predictions.py")

print(f"\nAll scripts created successfully in: {current_dir}")
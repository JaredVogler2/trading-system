# check_prediction_files.py
from pathlib import Path
import pandas as pd
from datetime import datetime

data_dir = Path('data')

print("=== Checking for prediction files ===\n")

if not data_dir.exists():
    print("Data directory not found!")
else:
    # Look for all CSV files in data directory
    csv_files = list(data_dir.glob('*.csv'))

    if not csv_files:
        print("No CSV files found in data directory")
    else:
        print(f"Found {len(csv_files)} CSV files:\n")

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

# clear_dashboard_cache.py
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

print("\nCache cleared! Please restart the dashboard.")

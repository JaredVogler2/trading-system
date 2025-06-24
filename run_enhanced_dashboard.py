# run_enhanced_dashboard.py
"""
Runner script for the enhanced real-data trading dashboard.
This replaces the mock data dashboard with a fully integrated real-data system.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import time
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dashboard_script():
    """Create the dashboard runner script."""
    dashboard_code = '''
import sys
import os
sys.path.append('.')

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# Verify environment variables
api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    import streamlit as st
    st.error("❌ ERROR: Alpaca API keys not found!")
    st.error("Please add your API keys to the .env file:")
    st.code("""
# .env file
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
    """)
    st.stop()

# Import and run the enhanced dashboard
from monitoring.enhanced_dashboard import main

if __name__ == "__main__":
    main()
'''

    with open('temp_enhanced_dashboard.py', 'w') as f:
        f.write(dashboard_code)

    return 'temp_enhanced_dashboard.py'


def verify_setup():
    """Verify that all required components are in place."""
    issues = []

    # Check for required files
    required_files = [
        'monitoring/enhanced_dashboard.py',
        'monitoring/realtime_data_sync.py',
        '.env'
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            issues.append(f"Missing required file: {file_path}")

    # Check for required directories
    required_dirs = ['data', 'models', 'logs', 'monitoring']

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    # Check environment variables
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY") == "your_api_key_here":
        issues.append("ALPACA_API_KEY not configured in .env file")

    if not os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") == "your_secret_key_here":
        issues.append("ALPACA_SECRET_KEY not configured in .env file")

    return issues


def setup_enhanced_dashboard():
    """Set up the enhanced dashboard files."""
    logger.info("Setting up enhanced dashboard...")

    # Create monitoring directory if it doesn't exist
    Path('monitoring').mkdir(exist_ok=True)

    # Check if files already exist
    if not Path('monitoring/enhanced_dashboard.py').exists():
        logger.error("monitoring/enhanced_dashboard.py not found!")
        logger.info(
            "Please save the 'Enhanced Real-Data Trading Dashboard' artifact as monitoring/enhanced_dashboard.py")
        return False

    if not Path('monitoring/realtime_data_sync.py').exists():
        logger.error("monitoring/realtime_data_sync.py not found!")
        logger.info(
            "Please save the 'Real-time Data Synchronization Module' artifact as monitoring/realtime_data_sync.py")
        return False

    # Create __init__.py if needed
    init_file = Path('monitoring/__init__.py')
    if not init_file.exists():
        init_file.write_text('"""Enhanced monitoring package."""\n')

    return True


def run_dashboard(port=8501, browser=True):
    """Run the enhanced dashboard."""
    logger.info("Starting enhanced trading dashboard...")

    # Verify setup
    issues = verify_setup()
    if issues:
        logger.error("Setup issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False

    # Set up dashboard files
    if not setup_enhanced_dashboard():
        return False

    # Create runner script
    script_path = create_dashboard_script()

    try:
        # Start Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            script_path,
            '--server.port', str(port),
            '--server.headless', 'true'
        ]

        if not browser:
            cmd.append('--server.browser.gatherUsageStats')
            cmd.append('false')

        logger.info(f"Launching dashboard on http://localhost:{port}")
        logger.info("Press Ctrl+C to stop")

        # Run streamlit
        process = subprocess.Popen(cmd)

        # Wait a moment for startup
        time.sleep(3)

        # Keep running
        process.wait()

    except KeyboardInterrupt:
        logger.info("\nShutting down dashboard...")
        if process:
            process.terminate()
            process.wait(timeout=5)

    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return False

    finally:
        # Clean up temp file
        if Path(script_path).exists():
            Path(script_path).unlink()

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Enhanced Trading Dashboard')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on (default: 8501)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser')
    parser.add_argument('--setup-only', action='store_true', help='Only set up files, do not run')

    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════╗
    ║        Enhanced Real-Data Trading Dashboard          ║
    ╠══════════════════════════════════════════════════════╣
    ║  • Real-time Alpaca portfolio data                  ║
    ║  • Live ML predictions and signals                  ║
    ║  • News sentiment analysis                          ║
    ║  • Backtesting results                              ║
    ║  • NO MOCK DATA - Everything is real!               ║
    ╚══════════════════════════════════════════════════════╝
    """)

    if args.setup_only:
        issues = verify_setup()
        if issues:
            print("\n❌ Setup issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("\n✅ Setup verified - ready to run!")
            print(f"\nRun without --setup-only to start the dashboard")
    else:
        success = run_dashboard(port=args.port, browser=not args.no_browser)

        if success:
            print("\n✅ Dashboard stopped successfully")
        else:
            print("\n❌ Dashboard encountered errors")
            print("\nTroubleshooting:")
            print("1. Check your .env file has valid Alpaca API keys")
            print("2. Ensure all required files are in place")
            print("3. Run with --setup-only to verify setup")
            print("4. Check logs for detailed error messages")


if __name__ == "__main__":
    main()
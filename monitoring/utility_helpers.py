# monitoring/utility_helpers.py
"""Utility helper functions for the dashboard."""

import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DashboardUtilities:
    """Utility methods for dashboard operations."""

    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_duration = 30  # seconds

    def is_market_open(self) -> bool:
        """Check if US market is currently open."""
        now = datetime.now()

        # Simple check - enhance with holiday calendar
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.time() >= pd.Timestamp("09:30").time()
        market_close = now.time() <= pd.Timestamp("16:00").time()

        return market_open and market_close

    def is_cache_valid(self, key: str, duration: int = None) -> bool:
        """Check if cached data is still valid."""
        if duration is None:
            duration = self.cache_duration

        if key not in self.cache_timestamps:
            return False

        age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return age < duration

    def cache_data(self, key: str, data) -> None:
        """Cache data with timestamp."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")

    def get_market_status(self):
        """Get current market status and next open time."""
        now = datetime.now()
        is_open = self.is_market_open()

        if is_open:
            # Calculate time to close
            close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            time_to_close = close_time - now
            return {
                'status': f"🟢 Open (closes in {time_to_close})",
                'is_open': True,
                'next_open': None
            }
        else:
            # Calculate next open
            next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            if now.time() > pd.Timestamp("16:00").time() or now.weekday() >= 5:
                # Move to next business day
                next_open += pd.Timedelta(days=1)
                while next_open.weekday() >= 5:
                    next_open += pd.Timedelta(days=1)

            return {
                'status': "🔴 Closed",
                'is_open': False,
                'next_open': next_open.strftime("%Y-%m-%d %H:%M")
            }

    def format_mock_value(self, value, format_str=None):
        """Format a value with mock data indicator."""
        if format_str:
            formatted = format_str.format(value)
        else:
            formatted = str(value)
        return f"{formatted} ⚠️*"
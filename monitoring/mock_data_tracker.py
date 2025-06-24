# monitoring/mock_data_tracker.py
"""
Mock data tracker for dashboard functionality.
This module helps identify when real data is unavailable.
"""

from typing import Dict, Set, Any, Optional


class MockDataTracker:
    """Tracks which data sources are using mock data."""

    def __init__(self):
        """Initialize the mock data tracker."""
        self.mock_sources: Set[str] = set()
        self.mock_reasons: Dict[str, str] = {}
        self.mock_values: Dict[str, Dict[str, Any]] = {}

    def add_mock_source(self, source: str, reason: str = "Data unavailable"):
        """Mark a data source as using mock data."""
        self.mock_sources.add(source)
        self.mock_reasons[source] = reason

        if source not in self.mock_values:
            self.mock_values[source] = {
                'is_mock': True,
                'reason': reason
            }

    def clear_mock_source(self, source: str):
        """Mark a data source as using real data."""
        self.mock_sources.discard(source)
        if source in self.mock_reasons:
            del self.mock_reasons[source]
        if source in self.mock_values:
            del self.mock_values[source]

    def is_mock(self, source: str) -> bool:
        """Check if a data source is using mock data."""
        return source in self.mock_sources

    def get_mock_count(self) -> int:
        """Get the number of sources using mock data."""
        return len(self.mock_sources)

    def get_warning_banner(self) -> Optional[str]:
        """Get a warning banner if any mock data is in use."""
        if not self.mock_sources:
            return None

        count = len(self.mock_sources)
        sources = ", ".join(sorted(self.mock_sources))

        if count == 1:
            return f"⚠️ WARNING: Using mock data for {sources}"
        else:
            return f"⚠️ WARNING: Using mock data for {count} sources: {sources}"

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of mock data status."""
        return {
            'using_mock_data': bool(self.mock_sources),
            'mock_count': len(self.mock_sources),
            'mock_sources': list(self.mock_sources),
            'mock_reasons': self.mock_reasons
        }

    def reset(self):
        """Reset all mock data tracking."""
        self.mock_sources.clear()
        self.mock_reasons.clear()
        self.mock_values.clear()


# Create a default instance
_default_tracker = MockDataTracker()


def get_mock_tracker() -> MockDataTracker:
    """Get the default mock data tracker instance."""
    return _default_tracker
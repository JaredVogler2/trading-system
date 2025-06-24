# monitoring/css_styles.py
"""CSS styles for the dashboard."""


def get_custom_css():
    """Get custom CSS for the comprehensive dashboard."""
    return """
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    .nav-button {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 10px 15px;
        margin: 5px;
        color: white;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s;
    }
    .nav-button:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    .nav-button.active {
        background: rgba(0,255,136,0.3);
        border-color: #00ff88;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #444;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-connected {
        color: #00ff88;
        font-weight: bold;
    }
    .status-disconnected {
        color: #ff4444;
        font-weight: bold;
    }
    .status-warning {
        color: #ffaa00;
        font-weight: bold;
    }
    .section-header {
        background: linear-gradient(90deg, rgba(0,255,136,0.1) 0%, rgba(0,123,255,0.1) 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 20px 0 10px 0;
    }
    .performance-positive {
        color: #00ff88;
        font-weight: bold;
    }
    .performance-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .news-item {
        background: rgba(255,255,255,0.02);
        border-left: 3px solid #00ff88;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        animation: pulse 2s infinite;
    }
    .mock-value {
        background-color: rgba(255, 193, 7, 0.1) !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
        position: relative;
    }
    .mock-value::after {
        content: "⚠️ MOCK";
        position: absolute;
        top: 2px;
        right: 2px;
        font-size: 10px;
        color: #FFC107;
        font-weight: bold;
    }
    .mock-warning-banner {
        background: linear-gradient(90deg, #ff6b6b 0%, #ff8787 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
        text-align: center;
        font-size: 16px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """
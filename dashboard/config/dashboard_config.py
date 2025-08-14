"""
Dashboard Configuration Settings
Central configuration for the Options Analysis Dashboard.
"""

from pathlib import Path
from typing import Dict, List, Any
import streamlit as st

class DashboardConfig:
    """Configuration settings for the dashboard."""
    
    # Path settings
    BASE_PATH = Path(__file__).parent.parent.parent
    DATA_PATH = BASE_PATH / 'data'
    SCRAPED_DATA_PATH = BASE_PATH / 'scraped_data'
    PRICE_DATA_PATH = DATA_PATH / 'price_data'
    
    # Data file paths
    UNIFIED_ARTICLES_PATH = SCRAPED_DATA_PATH / 'playwright' / 'unified_articles_complete.json'
    COMBINED_PRICE_DATA_PATH = PRICE_DATA_PATH / 'combined_daily_prices.csv'
    
    # Dashboard settings
    DASHBOARD_TITLE = "Options Analysis Dashboard"
    DASHBOARD_ICON = "📊"
    LAYOUT = "wide"
    
    # Page configuration
    PAGES = {
        "🎯 Executive Summary": {
            "id": "executive_summary",
            "description": "KPI overview and performance summary"
        },
        "📈 Weekly Analysis": {
            "id": "weekly_analysis", 
            "description": "Weekly correlation tracking and signal analysis"
        },
        "📅 Monthly Deep Dive": {
            "id": "monthly_analysis",
            "description": "Monthly trend analysis and market regime correlation"
        },
        "⚡ Strategy Backtesting": {
            "id": "strategy_backtesting",
            "description": "Convert article signals into trading strategies"
        },
        "🧠 Advanced Analytics": {
            "id": "advanced_analytics",
            "description": "Machine learning and statistical analysis"
        },
        "🖼️ Image Intelligence": {
            "id": "image_intelligence",
            "description": "Visual pattern recognition and chart analysis"
        }
    }
    
    # Filter settings
    DEFAULT_ASSETS = ["BTC", "ETH"]
    DEFAULT_THEMES = ["volatility", "options_strategy"]
    DEFAULT_CONFIDENCE_THRESHOLD = 0.8
    
    AVAILABLE_THEMES = [
        "volatility",
        "options_strategy", 
        "btc_focus",
        "eth_focus",
        "macro_events",
        "market_structure",
        "sentiment",
        "technical"
    ]
    
    MARKET_PERIODS = [
        "bear_market_2022",
        "recovery_early_2023",
        "bull_run_mid_2023",
        "etf_approval_period",
        "halving_period_2024",
        "post_halving_2024",
        "bull_continuation_2025"
    ]
    
    # Analysis settings
    TIME_HORIZONS = [1, 3, 7, 14, 30]  # Days
    CORRELATION_WINDOW = 7  # Days for primary correlation analysis
    MIN_SAMPLES_FOR_ANALYSIS = 10
    
    # Visualization settings
    CHART_HEIGHT = 400
    LARGE_CHART_HEIGHT = 600
    HEATMAP_SIZE = 500
    
    COLOR_SCHEME = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ffaa00',
        'info': '#17a2b8',
        'btc': '#f7931a',
        'eth': '#627eea',
        'bullish': '#2ca02c',
        'bearish': '#d62728',
        'neutral': '#6c757d'
    }
    
    # Performance metrics
    PERFORMANCE_METRICS = [
        'mean_return',
        'volatility', 
        'sharpe_ratio',
        'max_return',
        'min_return',
        'max_drawdown',
        'win_rate',
        'profit_factor'
    ]
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour in seconds
    MAX_CACHE_ENTRIES = 100
    
    # Display settings
    DECIMAL_PLACES = {
        'percentage': 2,
        'currency': 2,
        'ratio': 3,
        'correlation': 3
    }
    
    # Export settings
    EXPORT_FORMATS = ['CSV', 'JSON', 'Excel']
    MAX_EXPORT_ROWS = 10000
    
    @classmethod
    def get_page_config(cls) -> Dict[str, Any]:
        """Get Streamlit page configuration."""
        return {
            "page_title": cls.DASHBOARD_TITLE,
            "page_icon": cls.DASHBOARD_ICON,
            "layout": cls.LAYOUT,
            "initial_sidebar_state": "expanded",
            "menu_items": {
                'Get help': 'https://github.com/anthropics/claude-code',
                'Report a bug': 'https://github.com/anthropics/claude-code/issues',
                'About': f"""
                # {cls.DASHBOARD_TITLE}
                
                Advanced correlation analysis between options flow articles and market performance.
                
                **Data Coverage:**
                - 126 classified articles (Sept 2022 - Aug 2025)
                - 3,370 BTC/ETH price records with 49 indicators
                - 14,674 AI-analyzed images
                
                **Analysis Types:**
                - Weekly/Monthly correlation tracking
                - Strategy backtesting with risk metrics  
                - Machine learning insights
                - Visual pattern recognition
                """
            }
        }
    
    @classmethod
    def get_filter_defaults(cls) -> Dict[str, Any]:
        """Get default filter values."""
        return {
            'assets': cls.DEFAULT_ASSETS,
            'themes': cls.DEFAULT_THEMES,
            'confidence_threshold': cls.DEFAULT_CONFIDENCE_THRESHOLD
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration and required paths."""
        required_paths = [
            cls.DATA_PATH,
            cls.SCRAPED_DATA_PATH,
            cls.PRICE_DATA_PATH
        ]
        
        for path in required_paths:
            if not path.exists():
                st.error(f"Required path does not exist: {path}")
                return False
        
        required_files = [
            cls.UNIFIED_ARTICLES_PATH,
            cls.COMBINED_PRICE_DATA_PATH
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                st.error(f"Required data file does not exist: {file_path}")
                return False
        
        return True
    
    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get caching configuration."""
        return {
            'ttl': cls.CACHE_TTL,
            'max_entries': cls.MAX_CACHE_ENTRIES,
            'show_spinner': True,
            'persist': True
        }
    
    @classmethod
    def format_value(cls, value: float, value_type: str) -> str:
        """Format value according to type."""
        if value_type == 'percentage':
            decimals = cls.DECIMAL_PLACES['percentage']
            return f"{value:.{decimals}%}"
        elif value_type == 'currency':
            decimals = cls.DECIMAL_PLACES['currency']
            if abs(value) >= 1_000_000:
                return f"${value/1_000_000:.{decimals}f}M"
            elif abs(value) >= 1_000:
                return f"${value/1_000:.{decimals}f}K"
            else:
                return f"${value:.{decimals}f}"
        elif value_type in cls.DECIMAL_PLACES:
            decimals = cls.DECIMAL_PLACES[value_type]
            return f"{value:.{decimals}f}"
        else:
            return str(value)
    
    @classmethod
    def get_theme_info(cls) -> Dict[str, Dict[str, str]]:
        """Get theme information and descriptions."""
        return {
            "volatility": {
                "name": "Volatility Analysis",
                "description": "Articles focused on volatility patterns and IV analysis",
                "color": "#1f77b4"
            },
            "options_strategy": {
                "name": "Options Strategy",
                "description": "Articles discussing specific options trading strategies",
                "color": "#ff7f0e"
            },
            "btc_focus": {
                "name": "BTC Focus", 
                "description": "Articles primarily focused on Bitcoin options",
                "color": "#f7931a"
            },
            "eth_focus": {
                "name": "ETH Focus",
                "description": "Articles primarily focused on Ethereum options", 
                "color": "#627eea"
            },
            "macro_events": {
                "name": "Macro Events",
                "description": "Articles related to macro economic events",
                "color": "#d62728"
            },
            "market_structure": {
                "name": "Market Structure",
                "description": "Articles analyzing market microstructure and flows",
                "color": "#9467bd"
            },
            "sentiment": {
                "name": "Market Sentiment",
                "description": "Articles focused on market sentiment analysis",
                "color": "#8c564b"
            },
            "technical": {
                "name": "Technical Analysis",
                "description": "Articles with technical analysis and chart patterns",
                "color": "#e377c2"
            }
        }
    
    @classmethod
    def get_help_text(cls, section: str) -> str:
        """Get help text for different sections."""
        help_texts = {
            'date_range': "Filter analysis by date range. Leave empty to include all dates.",
            'assets': "Select which assets to include in the analysis (BTC, ETH).",
            'themes': "Filter articles by their primary themes or content focus.",
            'confidence_threshold': "Minimum extraction confidence for articles (0.0 = all articles, 1.0 = only perfect extractions).",
            'time_horizon': "Number of days forward to analyze price impact after article publication.",
            'correlation_window': "Time window for calculating correlations between article metrics and price performance.",
            'signal_strength': "Measure of how strong/actionable the trading signal is in the article (0.0 - 1.0).",
            'directional_bias': "Overall directional bias extracted from article content (bullish/bearish/neutral).",
            'market_period': "Market regime classification based on article publication date.",
            'risk_metrics': "Risk-adjusted performance metrics including Sharpe ratio, max drawdown, and win rate."
        }
        
        return help_texts.get(section, "")
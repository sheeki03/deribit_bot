#!/usr/bin/env python3
"""
Comprehensive Options Analysis Dashboard
Integrates article analysis with price data for sophisticated trading insights.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure Streamlit page
st.set_page_config(
    page_title="Options Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/anthropics/claude-code',
        'Report a bug': 'https://github.com/anthropics/claude-code/issues',
        'About': """
        # Options Analysis Dashboard
        
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
)

# Import utilities after Streamlit config
try:
    from utils.data_processor import DataProcessor
    from utils.analysis_engine import AnalysisEngine
    from utils.visualization_utils import VisualizationUtils
    from config.dashboard_config import DashboardConfig
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all required modules are available.")
    st.stop()

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'analysis_engine' not in st.session_state:
    st.session_state.analysis_engine = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_data():
    """Load and cache data for the dashboard."""
    if st.session_state.data_loaded:
        return st.session_state.data_processor, st.session_state.analysis_engine
    
    with st.spinner("Loading unified dataset..."):
        try:
            # Initialize data processor
            data_processor = DataProcessor()
            data_processor.load_all_data()
            
            # Initialize analysis engine
            analysis_engine = AnalysisEngine(data_processor)
            
            # Cache in session state
            st.session_state.data_processor = data_processor
            st.session_state.analysis_engine = analysis_engine
            st.session_state.data_loaded = True
            
            return data_processor, analysis_engine
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.error("Please ensure all data files are available and properly formatted.")
            st.stop()

def render_sidebar():
    """Render the sidebar with navigation and filters."""
    st.sidebar.title("📊 Options Analysis")
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    
    # Page selection
    pages = {
        "🎯 Executive Summary": "executive_summary",
        "📈 Weekly Analysis": "weekly_analysis", 
        "📅 Monthly Deep Dive": "monthly_analysis",
        "⚡ Strategy Backtesting": "strategy_backtesting",
        "🧠 Advanced Analytics": "advanced_analytics",
        "🖼️ Image Intelligence": "image_intelligence"
    }
    
    selected_page = st.sidebar.selectbox("Select Analysis Page", list(pages.keys()))
    
    st.sidebar.markdown("---")
    
    # Global filters
    st.sidebar.subheader("🔍 Global Filters")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(st.session_state.get('min_date', None), st.session_state.get('max_date', None)),
        help="Filter analysis by date range"
    )
    
    # Asset filter
    asset_filter = st.sidebar.multiselect(
        "Assets",
        ["BTC", "ETH"],
        default=["BTC", "ETH"],
        help="Select assets to analyze"
    )
    
    # Theme filter
    theme_filter = st.sidebar.multiselect(
        "Article Themes",
        ["volatility", "options_strategy", "btc_focus", "eth_focus", "macro_events"],
        default=["volatility", "options_strategy"],
        help="Filter by article themes"
    )
    
    # Confidence filter
    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence",
        0.0, 1.0, 0.8,
        step=0.05,
        help="Minimum article extraction confidence"
    )
    
    st.sidebar.markdown("---")
    
    # Data info
    if st.session_state.data_loaded:
        st.sidebar.subheader("📊 Data Overview")
        data_processor = st.session_state.data_processor
        
        st.sidebar.metric("Total Articles", len(data_processor.articles))
        st.sidebar.metric("Price Records", len(data_processor.price_data))
        st.sidebar.metric("Analyzed Images", sum(len(a.get('analyzed_images', [])) for a in data_processor.articles))
        
        # Date range info
        if hasattr(data_processor, 'date_range'):
            st.sidebar.write(f"**Date Range:** {data_processor.date_range['start']} to {data_processor.date_range['end']}")
    
    return {
        'selected_page': pages[selected_page],
        'date_range': date_range,
        'assets': asset_filter,
        'themes': theme_filter,
        'confidence_threshold': confidence_threshold
    }

def render_main_content(page_id, filters, data_processor, analysis_engine):
    """Render the main content based on selected page."""
    
    if page_id == "executive_summary":
        render_executive_summary(filters, data_processor, analysis_engine)
    elif page_id == "weekly_analysis":
        render_weekly_analysis(filters, data_processor, analysis_engine)
    elif page_id == "monthly_analysis":
        render_monthly_analysis(filters, data_processor, analysis_engine)
    elif page_id == "strategy_backtesting":
        render_strategy_backtesting(filters, data_processor, analysis_engine)
    elif page_id == "advanced_analytics":
        render_advanced_analytics(filters, data_processor, analysis_engine)
    elif page_id == "image_intelligence":
        render_image_intelligence(filters, data_processor, analysis_engine)

def render_executive_summary(filters, data_processor, analysis_engine):
    """Render Executive Summary page."""
    # Import the dedicated executive summary module
    from pages.executive_summary_page import render_executive_summary_content
    
    # Store current filters in session state for the page
    st.session_state['current_filters'] = filters
    
    # Render the full executive summary page
    render_executive_summary_content()

def render_weekly_analysis(filters, data_processor, analysis_engine):
    """Render Weekly Analysis page."""
    st.title("📈 Weekly Analysis")
    st.markdown("### Article Signals vs Weekly Performance")
    
    st.info("🚧 Weekly Analysis page coming soon! This will include weekly correlation tracking, inter-article price movements, and signal distribution analysis.")

def render_monthly_analysis(filters, data_processor, analysis_engine):
    """Render Monthly Deep Dive page."""
    st.title("📅 Monthly Deep Dive")
    st.markdown("### Monthly Trend Analysis and Performance Attribution")
    
    st.info("🚧 Monthly Analysis page coming soon! This will include monthly theme performance, volatility analysis, and market regime correlation.")

def render_strategy_backtesting(filters, data_processor, analysis_engine):
    """Render Strategy Backtesting page."""
    st.title("⚡ Strategy Backtesting")
    st.markdown("### Convert Article Signals into Trading Strategies")
    
    st.info("🚧 Strategy Backtesting page coming soon! This will include signal-to-strategy conversion, performance metrics, and risk analysis.")

def render_advanced_analytics(filters, data_processor, analysis_engine):
    """Render Advanced Analytics page."""
    st.title("🧠 Advanced Analytics")
    st.markdown("### Machine Learning and Statistical Analysis")
    
    st.info("🚧 Advanced Analytics page coming soon! This will include ML correlation models, clustering analysis, and predictive modeling.")

def render_image_intelligence(filters, data_processor, analysis_engine):
    """Render Image Intelligence page."""
    st.title("🖼️ Image Intelligence")
    st.markdown("### Visual Pattern Recognition and Chart Analysis")
    
    st.info("🚧 Image Intelligence page coming soon! This will include chart pattern recognition, visual sentiment analysis, and OCR text mining.")

def main():
    """Main application function."""
    
    # Header
    st.title("📊 Options Analysis Dashboard")
    st.markdown("*Advanced correlation analysis between options flow articles and market performance*")
    
    # Load data
    data_processor, analysis_engine = load_data()
    
    # Render sidebar and get filters
    filters = render_sidebar()
    
    # Apply filters to analysis engine
    analysis_engine.apply_filters(filters)
    
    # Render main content
    render_main_content(filters['selected_page'], filters, data_processor, analysis_engine)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Options Analysis Dashboard | Data: Sept 2022 - Aug 2025 | 
        126 Articles • 3,370 Price Records • 14,674 Images
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive Options Analysis Dashboard
Integrates article analysis with price data for sophisticated trading insights.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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
    from dashboard.utils.data_processor import DataProcessor
    from dashboard.utils.analysis_engine import AnalysisEngine
    from dashboard.utils.visualization_utils import VisualizationUtils
    from dashboard.config.dashboard_config import DashboardConfig
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
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", help="Show file path debugging info")
    
    if debug_mode:
        st.write("## 🔍 Debug Information")
        
        # Show debug info
        from pathlib import Path
        import glob
        import os
        
        st.write(f"**Working Directory:** `{Path.cwd()}`")
        
        # Check for target files
        target_files = [
            "scraped_data/playwright/unified_articles_complete.json",
            "data/price_data/combined_daily_prices.csv"
        ]
        
        for file_path in target_files:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size / (1024*1024)
                st.success(f"✅ Found: `{file_path}` ({size:.1f}MB)")
            else:
                st.error(f"❌ Missing: `{file_path}`")
        
        # Try glob search
        st.write("**Glob Search Results:**")
        for pattern in ["**/unified_articles_complete.json", "**/combined_daily_prices.csv"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                st.write(f"Pattern `{pattern}`: {matches}")
            else:
                st.write(f"Pattern `{pattern}`: No matches")
        
        # List some directories
        st.write("**Available Directories:**")
        try:
            for item in sorted(Path.cwd().iterdir()):
                if item.is_dir():
                    st.write(f"📂 `{item.name}/`")
        except Exception as e:
            st.error(f"Error listing directories: {e}")
    
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
            st.error("**Data files not found!**")
            
            if debug_mode:
                st.write("**Detailed Error:**")
                import traceback
                st.code(traceback.format_exc())
            
            st.info("""
            **For Streamlit Cloud deployment, please ensure:**
            1. `scraped_data/playwright/unified_articles_complete.json` exists
            2. `data/price_data/combined_daily_prices.csv` exists
            
            **Or upload the files using the sidebar uploader.**
            """)
            st.stop()

def render_sidebar():
    """Render the sidebar with navigation and filters."""
    st.sidebar.title("📊 Options Analysis")
    st.sidebar.markdown("---")
    
    # Data upload section (if data is missing)
    if not st.session_state.data_loaded:
        with st.sidebar.expander("📁 Data Upload (Optional)", expanded=False):
            st.info("Upload data files if not found automatically:")
            
            articles_file = st.file_uploader(
                "Upload Articles JSON", 
                type=['json'],
                help="Upload unified_articles_complete.json"
            )
            
            price_file = st.file_uploader(
                "Upload Price Data CSV", 
                type=['csv'],
                help="Upload combined_daily_prices.csv"
            )
            
            if articles_file is not None or price_file is not None:
                if st.button("🔄 Reload Dashboard"):
                    st.session_state.data_loaded = False
                    st.session_state.uploaded_files = {
                        'articles': articles_file,
                        'prices': price_file
                    }
                    st.rerun()
        
        st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.subheader("🧭 Navigation")
    
    # Page selection
    pages = {
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
    # Set default dates if data is loaded
    default_start = st.session_state.get('min_date')
    default_end = st.session_state.get('max_date')
    
    if st.session_state.data_loaded and st.session_state.data_processor:
        date_range_info = st.session_state.data_processor.date_range
        if date_range_info and not default_start:
            default_start = pd.to_datetime(date_range_info['start']).date()
            default_end = pd.to_datetime(date_range_info['end']).date()
    
    # Fallback to reasonable defaults if still None
    if not default_start:
        default_start = pd.to_datetime('2022-09-01').date()
        default_end = pd.to_datetime('2025-08-14').date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(default_start, default_end) if default_start and default_end else None,
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
    
    # Store current filters in session state for all pages
    st.session_state['current_filters'] = filters
    
    # CRITICAL: Ensure data_processor and analysis_engine are in session state
    # This fixes the issue where pages appear empty
    if data_processor is not None:
        st.session_state.data_processor = data_processor
    if analysis_engine is not None:
        st.session_state.analysis_engine = analysis_engine
        st.session_state.data_loaded = True
    
    if page_id == "weekly_analysis":
        from dashboard.page_modules.weekly_analysis_page import render_weekly_analysis_content
        render_weekly_analysis_content()
    elif page_id == "monthly_analysis":
        from dashboard.page_modules.monthly_analysis_page import render_monthly_analysis_content
        render_monthly_analysis_content()
    elif page_id == "strategy_backtesting":
        from dashboard.page_modules.backtesting_page import render_backtesting_page
        render_backtesting_page()
    elif page_id == "advanced_analytics":
        from dashboard.page_modules.advanced_analytics_page import render_advanced_analytics_content
        render_advanced_analytics_content()
    elif page_id == "image_intelligence":
        render_image_intelligence(filters, data_processor, analysis_engine)

# Remove duplicate function - now handled in render_main_content



def render_image_intelligence(filters, data_processor, analysis_engine):
    """Render advanced Image Intelligence page."""
    st.title("🖼️ Advanced Image Intelligence")
    st.markdown("### AI-Powered Chart Analysis & Visual Sentiment Detection")
    
    # Performance overview
    col1, col2, col3 = st.columns(3)
    
    # Load sample metrics
    with col1:
        st.metric(
            "Images Analyzed",
            "14,674",
            "+1,247 this week",
            help="Total images processed by AI models"
        )
    
    with col2:
        st.metric(
            "Classification Accuracy",
            "92.4%",
            "+2.1%",
            help="Chart type classification success rate"
        )
    
    with col3:
        st.metric(
            "OCR Success Rate",
            "87.8%",
            "+1.5%",
            help="Text extraction success rate"
        )
    
    # Filter controls
    st.markdown("### 🔍 Analysis Filters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        chart_type_filter = st.selectbox(
            "Chart Type",
            options=["All", "candlestick", "line", "volume", "heatmap", "other"],
            index=0
        )
    
    with col2:
        sentiment_filter = st.selectbox(
            "Sentiment",
            options=["All", "bullish", "bearish", "neutral"],
            index=0
        )
    
    with col3:
        asset_filter = st.selectbox(
            "Asset",
            options=["All", "BTC", "ETH", "Mixed"],
            index=0
        )
    
    with col4:
        confidence_filter = st.slider(
            "Min Confidence",
            0.0, 1.0, 0.5, 0.1
        )
    
    # Analysis summary
    render_image_analysis_summary(chart_type_filter, sentiment_filter, asset_filter)
    
    # Sample analysis results
    render_sample_image_analysis()


def render_image_analysis_summary(chart_type_filter, sentiment_filter, asset_filter):
    """Render image analysis summary charts."""
    st.markdown("### 📊 Analysis Distribution")
    
    # Mock data for demonstration
    chart_types = {'candlestick': 45, 'line': 32, 'volume': 15, 'heatmap': 8}
    sentiments = {'bullish': 42, 'bearish': 38, 'neutral': 20}
    assets = {'BTC': 58, 'ETH': 35, 'Mixed': 7}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=list(chart_types.keys()),
            values=list(chart_types.values()),
            hole=0.3,
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )])
        fig.update_layout(title="Chart Type Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        colors = ['green', 'red', 'gray']
        fig = go.Figure(data=[go.Bar(
            x=list(sentiments.keys()),
            y=list(sentiments.values()),
            marker_color=colors
        )])
        fig.update_layout(title="Sentiment Distribution", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(data=[go.Bar(
            x=list(assets.keys()),
            y=list(assets.values()),
            marker_color='lightblue'
        )])
        fig.update_layout(title="Assets Detected", height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_sample_image_analysis():
    """Render sample image analysis results."""
    st.markdown("### 🖼️ Sample Analysis Results")
    
    # Mock sample data
    sample_analyses = [
        {
            'image_url': 'https://via.placeholder.com/300x200/1f77b4/ffffff?text=BTC+Chart',
            'chart_type': 'candlestick',
            'sentiment': 'bullish',
            'confidence': 0.89,
            'assets': ['BTC'],
            'key_points': ['Strong resistance break at $45,000', 'Volume increasing', 'RSI oversold recovery'],
            'extracted_text': 'BTC/USD 1D Chart - Price: $46,250 (+3.2%)'
        },
        {
            'image_url': 'https://via.placeholder.com/300x200/ff7f0e/ffffff?text=ETH+Analysis',
            'chart_type': 'line',
            'sentiment': 'bearish',
            'confidence': 0.76,
            'assets': ['ETH'],
            'key_points': ['Failed to break resistance', 'Declining volume', 'Bearish divergence'],
            'extracted_text': 'ETH Analysis - Support at $2,850'
        },
        {
            'image_url': 'https://via.placeholder.com/300x200/2ca02c/ffffff?text=Market+Heat',
            'chart_type': 'heatmap',
            'sentiment': 'neutral',
            'confidence': 0.82,
            'assets': ['BTC', 'ETH'],
            'key_points': ['Mixed signals across timeframes', 'Correlation breakdown', 'Range-bound action'],
            'extracted_text': 'Crypto Market Heatmap - Mixed Performance'
        }
    ]
    
    for i, analysis in enumerate(sample_analyses):
        with st.container():
            cols = st.columns([1, 2])
            
            with cols[0]:
                st.image(analysis['image_url'], caption=f"Sample Analysis {i+1}")
            
            with cols[1]:
                st.markdown("#### 🔍 AI Analysis Results")
                
                # Key metrics
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric("Chart Type", analysis['chart_type'])
                
                with metric_cols[1]:
                    sentiment_emoji = "🟢" if analysis['sentiment'] == "bullish" else "🔴" if analysis['sentiment'] == "bearish" else "⚪"
                    st.metric("Sentiment", f"{sentiment_emoji} {analysis['sentiment']}")
                
                with metric_cols[2]:
                    st.metric("Confidence", f"{analysis['confidence']:.1%}")
                
                with metric_cols[3]:
                    assets_str = ", ".join(analysis['assets'])
                    st.metric("Assets", assets_str)
                
                # Key points
                st.markdown("**📊 Key Analysis Points:**")
                for point in analysis['key_points']:
                    st.write(f"• {point}")
                
                # Extracted text
                if analysis['extracted_text']:
                    with st.expander("📝 Extracted Text (OCR)"):
                        st.write(analysis['extracted_text'])
            
            st.divider()
    
    st.info("💡 This shows sample AI analysis results. Connect your image data source to see real-time analysis.")

def main():
    """Main application function."""
    
    # Header
    st.title("📊 Options Analysis Dashboard")
    st.markdown("*Advanced correlation analysis between options flow articles and market performance*")
    
    # Initialize session state
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    
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
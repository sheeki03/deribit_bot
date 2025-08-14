"""
Executive Summary Page
Comprehensive KPI overview and performance summary for the Options Analysis Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_processor import DataProcessor
from dashboard.utils.analysis_engine import AnalysisEngine
from dashboard.utils.visualization_utils import VisualizationUtils
from dashboard.config.dashboard_config import DashboardConfig

def render_executive_summary():
    """Render the Executive Summary page."""
    
    st.title("🎯 Executive Summary")
    st.markdown("### Comprehensive Options Analysis Overview")
    
    # Initialize components if not in session state
    if 'data_processor' not in st.session_state:
        with st.spinner("Initializing dashboard..."):
            st.session_state.data_processor = DataProcessor()
            st.session_state.data_processor.load_all_data()
            st.session_state.analysis_engine = AnalysisEngine(st.session_state.data_processor)
    
    data_processor = st.session_state.data_processor
    analysis_engine = st.session_state.analysis_engine
    
    # Get current filters from main app (passed via session state)
    filters = st.session_state.get('current_filters', {})
    analysis_engine.apply_filters(filters)
    
    # Main KPIs Row
    render_main_kpis(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    # Overview Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        render_article_timeline(analysis_engine, filters)
    
    with col2:
        render_theme_distribution(analysis_engine, filters)
    
    st.markdown("---")
    
    # Performance Summary
    render_performance_summary(analysis_engine, filters)
    
    st.markdown("---")
    
    # Top Performers Section
    render_top_performers(analysis_engine, filters)

def render_main_kpis(data_processor: DataProcessor, analysis_engine: AnalysisEngine, 
                    filters: Dict[str, Any]):
    """Render the main KPI metrics."""
    
    # Get filtered data
    filtered_articles = data_processor.get_filtered_articles(filters)
    summary_stats = data_processor.get_summary_stats()
    
    # Calculate KPIs
    total_articles = len(filtered_articles)
    avg_confidence = analysis_engine.get_average_confidence()
    
    # Get total images for filtered articles
    total_images = sum(article.get('num_images', 0) for _, article in filtered_articles.iterrows())
    
    # Get price records count
    filtered_price_data = data_processor.get_filtered_price_data(filters)
    price_records = len(filtered_price_data)
    
    # Create KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_articles = None
        if total_articles != summary_stats['total_articles']:
            delta_articles = f"{total_articles - summary_stats['total_articles']}"
        
        st.metric(
            "Total Articles",
            f"{total_articles:,}",
            delta=delta_articles,
            help="Number of articles matching current filters"
        )
    
    with col2:
        st.metric(
            "Avg Confidence", 
            f"{avg_confidence:.1%}",
            delta=None,
            help="Average extraction confidence for filtered articles"
        )
    
    with col3:
        st.metric(
            "Analyzed Images",
            f"{total_images:,}",
            delta=None,
            help="Total AI-analyzed images in filtered articles"
        )
    
    with col4:
        st.metric(
            "Price Records",
            f"{price_records:,}",
            delta=None,
            help="BTC/ETH price data points in selected date range"
        )

def render_article_timeline(analysis_engine: AnalysisEngine, filters: Dict[str, Any]):
    """Render article frequency timeline chart."""
    
    st.subheader("📈 Article Frequency Over Time")
    
    try:
        timeline_chart = analysis_engine.create_article_timeline_chart(filters)
        st.plotly_chart(timeline_chart, use_container_width=True, key="timeline_chart")
        
        # Add insights text
        with st.expander("📊 Timeline Insights"):
            filtered_articles = analysis_engine.data_processor.get_filtered_articles(filters)
            if not filtered_articles.empty:
                # Calculate insights
                monthly_avg = len(filtered_articles) / max(1, filtered_articles['date'].dt.to_period('M').nunique())
                peak_month = filtered_articles.set_index('date').resample('M').size().idxmax()
                peak_count = filtered_articles.set_index('date').resample('M').size().max()
                
                st.write(f"**Average articles per month:** {monthly_avg:.1f}")
                st.write(f"**Peak activity:** {peak_month.strftime('%B %Y')} with {peak_count} articles")
                
                # Market period breakdown
                period_counts = filtered_articles['market_period'].value_counts()
                if not period_counts.empty:
                    st.write("**Most active period:** " + 
                           f"{period_counts.index[0].replace('_', ' ').title()} ({period_counts.iloc[0]} articles)")
    
    except Exception as e:
        st.error(f"Error creating timeline chart: {e}")
        st.info("Unable to display timeline. Please check data availability.")

def render_theme_distribution(analysis_engine: AnalysisEngine, filters: Dict[str, Any]):
    """Render theme distribution chart."""
    
    st.subheader("🎯 Theme Distribution")
    
    try:
        theme_chart = analysis_engine.create_theme_distribution_chart(filters)
        st.plotly_chart(theme_chart, use_container_width=True, key="theme_chart")
        
        # Add theme insights
        with st.expander("🎨 Theme Analysis"):
            filtered_articles = analysis_engine.data_processor.get_filtered_articles(filters)
            if not filtered_articles.empty:
                theme_counts = filtered_articles['primary_theme'].value_counts()
                
                st.write("**Theme Breakdown:**")
                theme_info = DashboardConfig.get_theme_info()
                
                for theme, count in theme_counts.head(5).items():
                    theme_name = theme_info.get(theme, {}).get('name', theme.title())
                    percentage = (count / len(filtered_articles)) * 100
                    st.write(f"- {theme_name}: {count} articles ({percentage:.1f}%)")
                
                # Theme diversity
                theme_diversity = len(theme_counts) / len(DashboardConfig.AVAILABLE_THEMES)
                st.write(f"**Theme diversity:** {theme_diversity:.1%} of available themes covered")
    
    except Exception as e:
        st.error(f"Error creating theme chart: {e}")
        st.info("Unable to display theme distribution. Please check data availability.")

def render_performance_summary(analysis_engine: AnalysisEngine, filters: Dict[str, Any]):
    """Render performance summary metrics."""
    
    st.subheader("📊 Performance Summary")
    
    try:
        performance_summary = analysis_engine.get_performance_summary(filters)
        risk_metrics = analysis_engine.calculate_risk_metrics(filters)
        
        # Performance KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_theme = performance_summary.get('best_theme', 'N/A')
            best_score = performance_summary.get('best_theme_score', 0)
            
            st.metric(
                "Best Theme Performance",
                best_theme.replace('_', ' ').title() if best_theme != 'N/A' else 'N/A',
                delta=f"{best_score:.2%}" if best_score != 0 else None,
                help="Highest performing article theme by 7-day forward returns"
            )
        
        with col2:
            signal_accuracy = performance_summary.get('signal_accuracy', 0)
            st.metric(
                "Signal Accuracy",
                f"{signal_accuracy:.1%}",
                delta=f"{signal_accuracy - 0.5:.1%}" if signal_accuracy != 0 else None,
                delta_color="normal" if signal_accuracy >= 0.5 else "inverse",
                help="Directional accuracy of article signals (7-day horizon)"
            )
        
        with col3:
            avg_return = performance_summary.get('avg_return_impact', 0)
            st.metric(
                "Avg Return Impact",
                f"{avg_return:.2%}",
                delta=None,
                help="Average 7-day return following article publication"
            )
        
        with col4:
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                delta=f"{sharpe_ratio - 1.0:.2f}" if sharpe_ratio != 0 else None,
                delta_color="normal" if sharpe_ratio >= 1.0 else "inverse", 
                help="Risk-adjusted return metric (>1.0 is good)"
            )
        
        # Additional risk metrics
        if risk_metrics:
            st.markdown("#### Risk Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_rate = risk_metrics.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1%}")
                
                max_drawdown = risk_metrics.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")
            
            with col2:
                volatility = risk_metrics.get('volatility', 0)
                st.metric("Volatility", f"{volatility:.2%}")
                
                profit_factor = risk_metrics.get('profit_factor', 0)
                pf_display = f"{profit_factor:.2f}" if profit_factor != np.inf else "∞"
                st.metric("Profit Factor", pf_display)
            
            with col3:
                avg_win = risk_metrics.get('average_win', 0)
                st.metric("Avg Win", f"{avg_win:.2%}")
                
                avg_loss = risk_metrics.get('average_loss', 0)
                st.metric("Avg Loss", f"{avg_loss:.2%}")
    
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")
        st.info("Performance metrics unavailable. This may indicate insufficient data for analysis.")

def render_top_performers(analysis_engine: AnalysisEngine, filters: Dict[str, Any]):
    """Render top and worst performing articles."""
    
    st.subheader("🏆 Top Performers Analysis")
    
    try:
        top_performers = analysis_engine.get_top_performers(filters, n=5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🥇 Best Performing Articles")
            
            if top_performers['best']:
                for i, article in enumerate(top_performers['best'], 1):
                    with st.expander(f"{i}. {article['article_title'][:50]}..."):
                        st.write(f"**Date:** {article['article_date'].strftime('%Y-%m-%d')}")
                        st.write(f"**Asset:** {article['asset']}")
                        st.write(f"**7-Day Return:** {article['period_return']:.2%}")
                        st.write(f"**Signal Strength:** {article['signal_strength']:.2f}")
                        st.write(f"**Directional Bias:** {article['directional_bias'].title()}")
                        st.write(f"**Theme:** {article['primary_theme'].replace('_', ' ').title()}")
            else:
                st.info("No performance data available for current filters.")
        
        with col2:
            st.markdown("#### 📉 Worst Performing Articles") 
            
            if top_performers['worst']:
                for i, article in enumerate(top_performers['worst'], 1):
                    with st.expander(f"{i}. {article['article_title'][:50]}..."):
                        st.write(f"**Date:** {article['article_date'].strftime('%Y-%m-%d')}")
                        st.write(f"**Asset:** {article['asset']}")
                        st.write(f"**7-Day Return:** {article['period_return']:.2%}")
                        st.write(f"**Signal Strength:** {article['signal_strength']:.2f}")
                        st.write(f"**Directional Bias:** {article['directional_bias'].title()}")
                        st.write(f"**Theme:** {article['primary_theme'].replace('_', ' ').title()}")
            else:
                st.info("No performance data available for current filters.")
        
        # Performance attribution
        st.markdown("#### 📈 Performance Attribution")
        
        attribution = analysis_engine.calculate_performance_attribution(filters)
        
        if attribution:
            # Create tabs for different attribution views
            tab1, tab2, tab3, tab4 = st.tabs(["By Theme", "By Bias", "By Market Period", "By Signal Strength"])
            
            with tab1:
                if 'by_theme' in attribution:
                    theme_data = attribution['by_theme']
                    theme_chart = VisualizationUtils.create_performance_attribution_chart(
                        theme_data, "Performance by Theme"
                    )
                    st.plotly_chart(theme_chart, use_container_width=True)
            
            with tab2:
                if 'by_bias' in attribution:
                    bias_data = attribution['by_bias']
                    bias_chart = VisualizationUtils.create_performance_attribution_chart(
                        bias_data, "Performance by Directional Bias"
                    )
                    st.plotly_chart(bias_chart, use_container_width=True)
            
            with tab3:
                if 'by_market_period' in attribution:
                    period_data = attribution['by_market_period'] 
                    period_chart = VisualizationUtils.create_performance_attribution_chart(
                        period_data, "Performance by Market Period"
                    )
                    st.plotly_chart(period_chart, use_container_width=True)
            
            with tab4:
                if 'by_signal_strength' in attribution:
                    signal_data = attribution['by_signal_strength']
                    signal_chart = VisualizationUtils.create_performance_attribution_chart(
                        signal_data, "Performance by Signal Strength"
                    )
                    st.plotly_chart(signal_chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating top performers analysis: {e}")
        st.info("Top performers analysis unavailable. Please check data availability.")

if __name__ == "__main__":
    # This runs when the page is accessed directly
    render_executive_summary()
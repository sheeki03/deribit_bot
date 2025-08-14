"""
Executive Summary Page Content
This module provides the executive summary functionality for the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def render_executive_summary_content():
    """Render the executive summary page content."""
    
    st.title("🎯 Executive Summary")
    st.markdown("### Comprehensive Options Analysis Overview")
    
    # Check if data is loaded
    if 'data_processor' not in st.session_state:
        st.error("Data not loaded. Please reload the dashboard.")
        return
    
    data_processor = st.session_state.data_processor
    analysis_engine = st.session_state.analysis_engine
    
    # Get current filters
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

def render_main_kpis(data_processor, analysis_engine, filters):
    """Render the main KPI metrics."""
    
    try:
        # Get filtered data
        filtered_articles = data_processor.get_filtered_articles(filters)
        
        # Calculate KPIs
        total_articles = len(filtered_articles)
        avg_confidence = analysis_engine.get_average_confidence()
        
        # Get total images for filtered articles
        total_images = sum(article.get('num_images', 0) for _, article in filtered_articles.iterrows()) if not filtered_articles.empty else 0
        
        # Get price records count
        filtered_price_data = data_processor.get_filtered_price_data(filters)
        price_records = len(filtered_price_data)
        
        # Create KPI cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Articles",
                f"{total_articles:,}",
                help="Number of articles matching current filters"
            )
        
        with col2:
            st.metric(
                "Avg Confidence", 
                f"{avg_confidence:.1%}",
                help="Average extraction confidence for filtered articles"
            )
        
        with col3:
            st.metric(
                "Analyzed Images",
                f"{total_images:,}",
                help="Total AI-analyzed images in filtered articles"
            )
        
        with col4:
            st.metric(
                "Price Records",
                f"{price_records:,}",
                help="BTC/ETH price data points in selected date range"
            )
    
    except Exception as e:
        st.error(f"Error rendering KPIs: {e}")
        st.info("Some KPI data may be unavailable.")

def render_article_timeline(analysis_engine, filters):
    """Render article frequency timeline chart."""
    
    st.subheader("📈 Article Frequency Over Time")
    
    try:
        timeline_chart = analysis_engine.create_article_timeline_chart(filters)
        st.plotly_chart(timeline_chart, use_container_width=True)
        
        # Add basic insights
        filtered_articles = analysis_engine.data_processor.get_filtered_articles(filters)
        if not filtered_articles.empty:
            st.info(f"Showing timeline for {len(filtered_articles)} articles across {filtered_articles['date'].dt.to_period('M').nunique()} months")
    
    except Exception as e:
        st.error(f"Error creating timeline: {e}")
        st.info("Timeline chart unavailable. Please check data availability.")

def render_theme_distribution(analysis_engine, filters):
    """Render theme distribution chart."""
    
    st.subheader("🎯 Theme Distribution")
    
    try:
        theme_chart = analysis_engine.create_theme_distribution_chart(filters)
        st.plotly_chart(theme_chart, use_container_width=True)
        
        # Add theme summary
        filtered_articles = analysis_engine.data_processor.get_filtered_articles(filters)
        if not filtered_articles.empty:
            theme_counts = filtered_articles['primary_theme'].value_counts()
            top_theme = theme_counts.index[0] if not theme_counts.empty else "None"
            st.info(f"Most common theme: {top_theme.replace('_', ' ').title()} ({theme_counts.iloc[0]} articles)")
    
    except Exception as e:
        st.error(f"Error creating theme chart: {e}")
        st.info("Theme distribution chart unavailable. Please check data availability.")

def render_performance_summary(analysis_engine, filters):
    """Render performance summary metrics."""
    
    st.subheader("📊 Performance Summary")
    
    try:
        performance_summary = analysis_engine.get_performance_summary(filters)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_theme = performance_summary.get('best_theme', 'N/A')
            best_score = performance_summary.get('best_theme_score', 0)
            
            st.metric(
                "Best Theme Performance",
                best_theme.replace('_', ' ').title() if best_theme != 'N/A' else 'N/A',
                delta=f"{best_score:.2%}" if best_score != 0 else None,
                help="Highest performing article theme by forward returns"
            )
        
        with col2:
            signal_accuracy = performance_summary.get('signal_accuracy', 0)
            st.metric(
                "Signal Accuracy",
                f"{signal_accuracy:.1%}",
                delta=f"{signal_accuracy - 0.5:.1%}" if signal_accuracy > 0 else None,
                help="Directional accuracy of article signals"
            )
        
        with col3:
            avg_return = performance_summary.get('avg_return_impact', 0)
            st.metric(
                "Avg Return Impact",
                f"{avg_return:.2%}",
                help="Average return following article publication"
            )
    
    except Exception as e:
        st.error(f"Error calculating performance summary: {e}")
        st.info("Performance metrics may be unavailable due to insufficient data.")
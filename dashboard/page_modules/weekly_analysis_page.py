"""
Weekly Analysis Page Content
This module provides weekly correlation analysis between articles and price movements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

def render_weekly_analysis_content():
    """Render the comprehensive weekly analysis page content."""
    
    st.title("📈 Weekly Analysis Dashboard")
    st.markdown("### Advanced Article Signal Analysis & Weekly Price Performance")
    
    # Add comprehensive description
    with st.expander("📊 About Weekly Analysis", expanded=False):
        st.markdown("""
        This dashboard analyzes the correlation between article signals and subsequent weekly price movements.
        
        **Key Features**:
        - **Signal Accuracy**: Directional prediction accuracy over different time horizons
        - **Performance Attribution**: Returns broken down by themes, assets, and signal strength
        - **Inter-Article Analysis**: Price movements between consecutive article publications
        - **Risk-Adjusted Metrics**: Sharpe ratios, volatility analysis, and drawdown metrics
        - **Temporal Patterns**: Day-of-week effects, seasonal trends, and market regime analysis
        """)
    
    # Check if data is loaded with detailed status
    if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
        st.error("🚀 Data Processor not loaded. Please reload the dashboard.")
        st.info("The data processor loads and manages all article and price data for analysis.")
        
        # Offer manual loading option
        if st.button("🔄 Try Loading Data Manually"):
            try:
                from dashboard.utils.data_processor import DataProcessor
                from dashboard.utils.analysis_engine import AnalysisEngine
                
                with st.spinner("Loading data..."):
                    dp = DataProcessor()
                    dp.load_all_data()
                    ae = AnalysisEngine(dp)
                    
                    st.session_state.data_processor = dp
                    st.session_state.analysis_engine = ae
                    st.session_state.data_loaded = True
                    
                    st.success("Data loaded successfully! Refreshing page...")
                    st.rerun()
            except Exception as e:
                st.error(f"Manual loading failed: {e}")
                import traceback
                st.text(traceback.format_exc())
        return
    
    if 'analysis_engine' not in st.session_state or st.session_state.analysis_engine is None:
        st.error("🤖 Analysis Engine not loaded. Please reload the dashboard.")
        st.info("The analysis engine performs statistical calculations and correlation analysis.")
        return
    
    data_processor = st.session_state.data_processor
    analysis_engine = st.session_state.analysis_engine
    
    # Get current filters
    filters = st.session_state.get('current_filters', {})
    analysis_engine.apply_filters(filters)
    
    # Show data summary
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_articles = data_processor.get_filtered_articles(filters)
        correlation_data = data_processor.get_correlation_data(filters)
        
        with col1:
            st.metric("📄 Total Articles", len(filtered_articles) if not filtered_articles.empty else 0)
        with col2:
            st.metric("🔗 Correlation Records", len(correlation_data) if not correlation_data.empty else 0)
        with col3:
            if not filtered_articles.empty and 'asset' in filtered_articles.columns:
                assets = filtered_articles['asset'].value_counts()
                st.metric("🪙 Primary Asset", assets.index[0] if len(assets) > 0 else "N/A")
            else:
                st.metric("🪙 Primary Asset", "N/A")
        with col4:
            if not filtered_articles.empty:
                date_range = filtered_articles['date'].max() - filtered_articles['date'].min()
                st.metric("📅 Date Range", f"{date_range.days} days")
            else:
                st.metric("📅 Date Range", "N/A")
    
    st.markdown("---")
    
    # Enhanced Weekly Analysis Controls
    render_weekly_controls(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    # Main Analysis Grid - 2x2 layout
    col1, col2 = st.columns(2)
    
    with col1:
        render_weekly_correlation_chart(analysis_engine, filters)
        st.markdown("---")
        render_signal_accuracy_analysis(analysis_engine, filters)
    
    with col2:
        render_performance_metrics_summary(analysis_engine, filters)
        st.markdown("---")
        render_risk_adjusted_metrics(analysis_engine, filters)
    
    st.markdown("---")
    
    # Advanced Analysis Sections
    render_inter_article_analysis(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    render_weekly_performance_attribution(analysis_engine, filters)
    
    st.markdown("---")
    
    # Temporal Pattern Analysis
    render_temporal_pattern_analysis(data_processor, analysis_engine, filters)

def render_weekly_controls(data_processor, analysis_engine, filters):
    """Render weekly analysis control panel."""
    
    st.subheader("🎛️ Weekly Analysis Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Time horizon selector
        time_horizon = st.selectbox(
            "Analysis Window",
            options=[1, 3, 7, 14, 30],
            index=2,  # Default to 7 days
            help="Days forward to analyze price impact"
        )
        st.session_state['weekly_time_horizon'] = time_horizon
    
    with col2:
        # Minimum signal strength
        min_signal_strength = st.slider(
            "Min Signal Strength",
            0.0, 1.0, 0.5,
            step=0.1,
            help="Filter articles by minimum signal strength"
        )
        st.session_state['min_signal_strength'] = min_signal_strength
    
    with col3:
        # Theme focus
        available_themes = data_processor.articles_df['primary_theme'].unique().tolist()
        theme_focus = st.selectbox(
            "Theme Focus",
            options=['All'] + available_themes,
            help="Focus analysis on specific theme"
        )
        st.session_state['theme_focus'] = theme_focus
    
    with col4:
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type",
            options=['Correlation', 'Performance', 'Volatility', 'Volume'],
            help="Type of weekly analysis to perform"
        )
        st.session_state['analysis_type'] = analysis_type

def render_weekly_correlation_chart(analysis_engine, filters):
    """Render weekly correlation analysis chart."""
    
    st.subheader("📊 Weekly Article-Price Correlation")
    
    try:
        correlation_chart = analysis_engine.create_weekly_correlation_chart(filters)
        st.plotly_chart(correlation_chart, use_container_width=True)
        
        # Add correlation insights
        with st.expander("📈 Correlation Insights"):
            weekly_data = analysis_engine.data_processor.get_weekly_data(filters)
            
            if not weekly_data.empty:
                # Calculate correlations with bounds checking
                required_cols_1 = ['article_count', 'btc_return_1d']
                required_cols_2 = ['signal_strength', 'btc_return_1d']
                
                # Article Count vs BTC Returns correlation
                if all(col in weekly_data.columns for col in required_cols_1):
                    corr_1 = weekly_data[required_cols_1].corr()
                    if corr_1.shape[0] > 1 and corr_1.shape[1] > 1:
                        article_price_corr = corr_1.iloc[0, 1]
                        if pd.notna(article_price_corr):
                            st.write(f"**Article Count vs BTC Returns:** {article_price_corr:.3f}")
                        else:
                            st.write("**Article Count vs BTC Returns:** No correlation (insufficient data)")
                    else:
                        st.write("**Article Count vs BTC Returns:** Unable to compute (insufficient data)")
                else:
                    st.write("**Article Count vs BTC Returns:** Missing required columns")
                
                # Signal Strength vs BTC Returns correlation
                if all(col in weekly_data.columns for col in required_cols_2):
                    corr_2 = weekly_data[required_cols_2].corr()
                    if corr_2.shape[0] > 1 and corr_2.shape[1] > 1:
                        signal_btc_corr = corr_2.iloc[0, 1]
                        if pd.notna(signal_btc_corr):
                            st.write(f"**Signal Strength vs BTC Returns:** {signal_btc_corr:.3f}")
                        else:
                            st.write("**Signal Strength vs BTC Returns:** No correlation (insufficient data)")
                    else:
                        st.write("**Signal Strength vs BTC Returns:** Unable to compute (insufficient data)")
                else:
                    st.write("**Signal Strength vs BTC Returns:** Missing required columns")
                
                # Volatility insights
                if 'btc_volatility_7d' in weekly_data.columns:
                    vol_corr = weekly_data[['article_count', 'btc_volatility_7d']].corr().iloc[0, 1]
                    st.write(f"**Article Count vs BTC Volatility:** {vol_corr:.3f}")
    
    except Exception as e:
        st.error(f"Error creating weekly correlation chart: {e}")
        st.info("Weekly correlation analysis may be unavailable due to insufficient data.")

def render_signal_accuracy_analysis(analysis_engine, filters):
    """Render signal accuracy by time horizon analysis."""
    
    st.subheader("🎯 Signal Accuracy Analysis")
    
    try:
        accuracy_chart = analysis_engine.create_signal_accuracy_chart(filters)
        st.plotly_chart(accuracy_chart, use_container_width=True)
        
        # Add accuracy insights
        with st.expander("🔍 Accuracy Insights"):
            correlation_data = analysis_engine.data_processor.get_correlation_data(filters)
            
            if not correlation_data.empty:
                # Calculate accuracy by bias type
                accuracy_by_bias = {}
                for bias in ['bullish', 'bearish', 'neutral']:
                    bias_data = correlation_data[
                        (correlation_data['directional_bias'] == bias) &
                        (correlation_data['days_forward'] == 7)
                    ]
                    if not bias_data.empty:
                        accuracy = analysis_engine.calculate_signal_accuracy(bias_data)
                        accuracy_by_bias[bias] = accuracy
                
                st.write("**Accuracy by Directional Bias (7-day):**")
                for bias, accuracy in accuracy_by_bias.items():
                    st.write(f"- {bias.title()}: {accuracy:.1%}")
                
                # Best performing time horizon
                best_horizon_data = []
                for days in [1, 3, 7, 14, 30]:
                    horizon_data = correlation_data[correlation_data['days_forward'] == days]
                    if not horizon_data.empty:
                        accuracy = analysis_engine._calculate_signal_accuracy(horizon_data)
                        best_horizon_data.append((days, accuracy))
                
                if best_horizon_data:
                    best_horizon = max(best_horizon_data, key=lambda x: x[1])
                    st.write(f"**Best Time Horizon:** {best_horizon[0]} days ({best_horizon[1]:.1%} accuracy)")
    
    except Exception as e:
        st.error(f"Error creating signal accuracy chart: {e}")
        st.info("Signal accuracy analysis may be unavailable.")

def render_inter_article_analysis(data_processor, analysis_engine, filters):
    """Render inter-article price movement analysis."""
    
    st.subheader("🔄 Inter-Article Price Analysis")
    st.markdown("*Price movements between article publication dates*")
    
    try:
        # Get filtered articles and sort by date
        filtered_articles = data_processor.get_filtered_articles(filters)
        
        if filtered_articles.empty:
            st.warning("No articles match current filters.")
            return
        
        filtered_articles = filtered_articles.sort_values('date')
        
        # Calculate inter-article periods
        inter_article_data = []
        
        for i in range(len(filtered_articles) - 1):
            current_article = filtered_articles.iloc[i]
            next_article = filtered_articles.iloc[i + 1]
            
            # Get price data between articles
            start_date = current_article['date']
            end_date = next_article['date']
            period_days = (end_date - start_date).days
            
            if period_days > 0 and period_days <= 30:  # Reasonable period
                # Get BTC and ETH price data for this period
                price_data = data_processor.price_data[
                    (data_processor.price_data['date'] >= start_date) &
                    (data_processor.price_data['date'] <= end_date)
                ]
                
                for asset in ['BTC', 'ETH']:
                    asset_prices = price_data[price_data['asset'] == asset]
                    
                    if not asset_prices.empty:
                        start_price = asset_prices.iloc[0]['open']
                        end_price = asset_prices.iloc[-1]['close']
                        
                        # Guard against division by zero
                        if start_price == 0 or abs(start_price) < 1e-8:
                            period_return = float('nan')
                        else:
                            period_return = (end_price - start_price) / start_price
                        
                        inter_article_data.append({
                            'period_start': start_date,
                            'period_end': end_date,
                            'period_days': period_days,
                            'asset': asset,
                            'period_return': period_return,
                            'start_article_theme': current_article['primary_theme'],
                            'end_article_theme': next_article['primary_theme'],
                            'start_signal': current_article['signal_strength'],
                            'end_signal': next_article['signal_strength']
                        })
        
        if inter_article_data:
            inter_df = pd.DataFrame(inter_article_data)
            
            # Create tabs for different inter-article analyses
            tab1, tab2, tab3 = st.tabs(["Period Returns", "Theme Transitions", "Signal Evolution"])
            
            with tab1:
                render_period_returns_analysis(inter_df)
            
            with tab2:
                render_theme_transition_analysis(inter_df)
            
            with tab3:
                render_signal_evolution_analysis(inter_df)
        
        else:
            st.info("No inter-article periods found for current filters.")
    
    except Exception as e:
        st.error(f"Error in inter-article analysis: {e}")
        st.info("Inter-article analysis may be unavailable.")

def render_period_returns_analysis(inter_df):
    """Render period returns between articles."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_period = inter_df['period_days'].mean()
        st.metric("Avg Period (Days)", f"{avg_period:.1f}")
    
    with col2:
        avg_return = inter_df['period_return'].mean()
        st.metric("Avg Inter-Article Return", f"{avg_return:.2%}")
    
    with col3:
        best_return = inter_df['period_return'].max()
        st.metric("Best Period Return", f"{best_return:.2%}")
    
    with col4:
        worst_return = inter_df['period_return'].min()
        st.metric("Worst Period Return", f"{worst_return:.2%}")
    
    # Period returns chart
    fig = go.Figure()
    
    for asset in inter_df['asset'].unique():
        asset_data = inter_df[inter_df['asset'] == asset]
        
        fig.add_trace(go.Scatter(
            x=asset_data['period_start'],
            y=asset_data['period_return'] * 100,
            mode='markers+lines',
            name=f'{asset} Returns',
            text=[f"Period: {days}d<br>Return: {ret:.2%}" 
                  for days, ret in zip(asset_data['period_days'], asset_data['period_return'])],
            hovertemplate='%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Inter-Article Period Returns",
        xaxis_title="Period Start Date",
        yaxis_title="Period Return (%)",
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_theme_transition_analysis(inter_df):
    """Render theme transition analysis."""
    
    st.markdown("#### Theme Transition Performance")
    
    # Calculate performance by theme transition (handle None/NaN values)
    inter_df_clean = inter_df.copy()
    inter_df_clean['start_article_theme'] = inter_df_clean['start_article_theme'].fillna('Unknown')
    inter_df_clean['end_article_theme'] = inter_df_clean['end_article_theme'].fillna('Unknown')
    
    transition_perf = inter_df_clean.groupby(['start_article_theme', 'end_article_theme'])['period_return'].agg(['mean', 'count'])
    transition_perf = transition_perf.reset_index()
    transition_perf['transition'] = transition_perf['start_article_theme'] + ' → ' + transition_perf['end_article_theme']
    
    # Filter for transitions with at least 2 occurrences
    significant_transitions = transition_perf[transition_perf['count'] >= 2].sort_values('mean', ascending=False)
    
    if not significant_transitions.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=significant_transitions['transition'],
            y=significant_transitions['mean'] * 100,
            text=[f"{mean:.2%} ({count})" for mean, count in zip(significant_transitions['mean'], significant_transitions['count'])],
            textposition='auto',
            name='Avg Return'
        ))
        
        fig.update_layout(
            title="Performance by Theme Transition",
            xaxis_title="Theme Transition",
            yaxis_title="Average Return (%)",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No significant theme transitions found (minimum 2 occurrences).")

def render_signal_evolution_analysis(inter_df):
    """Render signal strength evolution analysis."""
    
    st.markdown("#### Signal Strength Evolution")
    
    # Calculate signal changes (handle NaN values explicitly)
    inter_df['signal_change'] = inter_df['end_signal'].sub(inter_df['start_signal'], fill_value=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal change distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=inter_df['signal_change'],
            nbinsx=20,
            name='Signal Change',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Distribution of Signal Strength Changes",
            xaxis_title="Signal Change",
            yaxis_title="Frequency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal change vs returns
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=inter_df['signal_change'],
            y=inter_df['period_return'] * 100,
            mode='markers',
            text=[f"Period: {days}d<br>Asset: {asset}" 
                  for days, asset in zip(inter_df['period_days'], inter_df['asset'])],
            hovertemplate='Signal Change: %{x:.2f}<br>Return: %{y:.2f}%<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Signal Change vs Period Returns",
            xaxis_title="Signal Strength Change",
            yaxis_title="Period Return (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_weekly_performance_attribution(analysis_engine, filters):
    """Render weekly performance attribution analysis."""
    
    st.subheader("📈 Weekly Performance Attribution")
    
    try:
        attribution = analysis_engine.calculate_performance_attribution(filters)
        
        if attribution:
            # Create tabs for different attribution views
            tab1, tab2 = st.tabs(["Theme Performance", "Signal Strength Analysis"])
            
            with tab1:
                if 'by_theme' in attribution:
                    render_theme_performance_chart(attribution['by_theme'])
            
            with tab2:
                if 'by_signal_strength' in attribution:
                    render_signal_strength_performance_chart(attribution['by_signal_strength'])
        
        else:
            st.info("Performance attribution data not available for current filters.")
    
    except Exception as e:
        st.error(f"Error in performance attribution: {e}")

def render_theme_performance_chart(theme_data):
    """Render theme performance attribution chart."""
    
    theme_df = pd.DataFrame.from_dict(theme_data, orient='index')
    theme_df = theme_df.sort_values('mean', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=theme_df.index,
        x=theme_df['mean'] * 100,
        orientation='h',
        text=[f"{mean:.2%} (n={count})" for mean, count in zip(theme_df['mean'], theme_df['count'])],
        textposition='auto',
        name='Avg Weekly Return'
    ))
    
    fig.update_layout(
        title="Weekly Performance by Article Theme",
        xaxis_title="Average Weekly Return (%)",
        yaxis_title="Article Theme",
        height=max(400, len(theme_df) * 30)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_signal_strength_performance_chart(signal_data):
    """Render signal strength performance attribution chart."""
    
    signal_df = pd.DataFrame.from_dict(signal_data, orient='index')
    
    if not signal_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=signal_df.index,
            y=signal_df['mean'] * 100,
            text=[f"{mean:.2%}" for mean in signal_df['mean']],
            textposition='auto',
            name='Avg Weekly Return'
        ))
        
        fig.update_layout(
            title="Weekly Performance by Signal Strength Quartile",
            xaxis_title="Signal Strength Quartile", 
            yaxis_title="Average Weekly Return (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Signal strength performance data not available.")

def render_performance_metrics_summary(analysis_engine, filters):
    """Render comprehensive performance metrics summary."""
    
    st.subheader("📊 Performance Metrics Summary")
    
    try:
        risk_metrics = analysis_engine.calculate_risk_metrics(filters)
        
        if not risk_metrics:
            st.warning("No performance data available for current filters.")
            return
        
        # Main performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📈 Mean Return", 
                f"{risk_metrics.get('mean_return', 0):.2%}",
                help="Average 7-day return following articles"
            )
            st.metric(
                "📊 Volatility", 
                f"{risk_metrics.get('volatility', 0):.2%}",
                help="Standard deviation of returns"
            )
        
        with col2:
            sharpe = risk_metrics.get('sharpe_ratio', 0)
            st.metric(
                "⚡ Sharpe Ratio", 
                f"{sharpe:.3f}",
                help="Risk-adjusted return (return/volatility)"
            )
            win_rate = risk_metrics.get('win_rate', 0)
            st.metric(
                "🎯 Win Rate", 
                f"{win_rate:.1%}",
                help="Percentage of profitable signals"
            )
        
        with col3:
            st.metric(
                "📈 Best Return", 
                f"{risk_metrics.get('max_return', 0):.2%}",
                help="Highest 7-day return"
            )
            st.metric(
                "📉 Worst Return", 
                f"{risk_metrics.get('min_return', 0):.2%}",
                help="Lowest 7-day return"
            )
        
        # Additional metrics
        st.markdown("**Additional Metrics**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_win = risk_metrics.get('average_win', 0)
            st.write(f"**Average Win**: {avg_win:.2%}")
        with col2:
            avg_loss = risk_metrics.get('average_loss', 0) 
            st.write(f"**Average Loss**: {avg_loss:.2%}")
        with col3:
            profit_factor = risk_metrics.get('profit_factor', 0)
            if profit_factor == float('inf'):
                st.write("**Profit Factor**: ∞ (no losses)")
            else:
                st.write(f"**Profit Factor**: {profit_factor:.2f}")
    
    except Exception as e:
        st.error(f"Error calculating performance metrics: {e}")

def render_risk_adjusted_metrics(analysis_engine, filters):
    """Render risk-adjusted performance metrics."""
    
    st.subheader("⚠️ Risk-Adjusted Analysis")
    
    try:
        risk_metrics = analysis_engine.calculate_risk_metrics(filters)
        
        if not risk_metrics:
            st.info("No risk data available for current filters.")
            return
        
        # Risk assessment
        max_dd = risk_metrics.get('max_drawdown', 0)
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        
        # Risk level determination
        if max_dd > 0.2:
            risk_level = "HIGH"
            risk_color = "red"
        elif max_dd > 0.1:
            risk_level = "MEDIUM" 
            risk_color = "orange"
        else:
            risk_level = "LOW"
            risk_color = "green"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Risk Level**: :{risk_color}[{risk_level}]")
            st.metric(
                "📉 Max Drawdown",
                f"{max_dd:.2%}",
                help="Maximum peak-to-trough decline"
            )
        
        with col2:
            # Sharpe ratio assessment
            if sharpe > 1.0:
                sharpe_assessment = "Excellent"
                sharpe_color = "green"
            elif sharpe > 0.5:
                sharpe_assessment = "Good"
                sharpe_color = "green"
            elif sharpe > 0:
                sharpe_assessment = "Fair"
                sharpe_color = "orange"
            else:
                sharpe_assessment = "Poor"
                sharpe_color = "red"
            
            st.markdown(f"**Risk-Adj Return**: :{sharpe_color}[{sharpe_assessment}]")
            st.metric(
                "📊 Sharpe Ratio",
                f"{sharpe:.3f}",
                help="Higher values indicate better risk-adjusted returns"
            )
        
        # Risk recommendations
        st.markdown("**Risk Management Recommendations**")
        
        recommendations = []
        if max_dd > 0.15:
            recommendations.append("Consider reducing position sizes")
        if sharpe < 0.5:
            recommendations.append("Review signal quality - low risk-adjusted returns")
        if risk_metrics.get('win_rate', 0) < 0.4:
            recommendations.append("Improve signal accuracy - low win rate")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("✅ Risk profile appears acceptable")
    
    except Exception as e:
        st.error(f"Error in risk analysis: {e}")

def render_temporal_pattern_analysis(data_processor, analysis_engine, filters):
    """Render temporal pattern analysis."""
    
    st.subheader("🕒 Temporal Pattern Analysis")
    
    try:
        correlation_data = data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            st.info("No temporal data available for analysis.")
            return
        
        # Add date/time features if article_date exists
        if 'article_date' in correlation_data.columns:
            correlation_data = correlation_data.copy()
            correlation_data['article_date'] = pd.to_datetime(correlation_data['article_date'])
            correlation_data['day_of_week'] = correlation_data['article_date'].dt.day_name()
            correlation_data['month'] = correlation_data['article_date'].dt.month_name()
            correlation_data['quarter'] = correlation_data['article_date'].dt.quarter
            
            tab1, tab2, tab3 = st.tabs(["📅 Day of Week", "📆 Monthly Patterns", "📊 Quarterly Trends"])
            
            with tab1:
                # Day of week analysis
                dow_performance = correlation_data.groupby('day_of_week')['period_return'].agg(['mean', 'count', 'std']).reset_index()
                dow_performance = dow_performance.sort_values('mean', ascending=False)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dow_performance['day_of_week'],
                    y=dow_performance['mean'] * 100,
                    text=[f"{mean:.2%} (n={count})" for mean, count in zip(dow_performance['mean'], dow_performance['count'])],
                    textposition='auto',
                    name='Avg Return',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Average Returns by Day of Week",
                    xaxis_title="Day of Week",
                    yaxis_title="Average Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best/worst days
                best_day = dow_performance.iloc[0]
                worst_day = dow_performance.iloc[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Best Day**: {best_day['day_of_week']} ({best_day['mean']:.2%})")
                with col2:
                    st.error(f"**Worst Day**: {worst_day['day_of_week']} ({worst_day['mean']:.2%})")
            
            with tab2:
                # Monthly analysis
                monthly_performance = correlation_data.groupby('month')['period_return'].agg(['mean', 'count']).reset_index()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_performance['month'],
                    y=monthly_performance['mean'] * 100,
                    text=[f"{mean:.2%}" for mean in monthly_performance['mean']],
                    textposition='auto',
                    name='Monthly Avg Return',
                    marker_color='lightgreen'
                ))
                
                fig.update_layout(
                    title="Average Returns by Month",
                    xaxis_title="Month",
                    yaxis_title="Average Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Quarterly analysis 
                quarterly_performance = correlation_data.groupby('quarter')['period_return'].agg(['mean', 'count', 'std']).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    for _, row in quarterly_performance.iterrows():
                        st.metric(
                            f"Q{int(row['quarter'])}",
                            f"{row['mean']:.2%}",
                            delta=f"±{row['std']:.2%}" if pd.notna(row['std']) else None
                        )
                
                with col2:
                    # Quarterly trend chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[f"Q{int(q)}" for q in quarterly_performance['quarter']],
                        y=quarterly_performance['mean'] * 100,
                        mode='lines+markers',
                        name='Quarterly Performance',
                        line=dict(color='purple', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title="Quarterly Performance Trend",
                        xaxis_title="Quarter",
                        yaxis_title="Average Return (%)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No date information available for temporal analysis.")
    
    except Exception as e:
        st.error(f"Error in temporal analysis: {e}")
        import traceback
        st.text(traceback.format_exc())
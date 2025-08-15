"""
Monthly Deep Dive Page Content
This module provides monthly trend analysis and performance attribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
from datetime import datetime, timedelta

def render_monthly_analysis_content():
    """Render the comprehensive monthly analysis page content."""
    
    st.title("📅 Monthly Deep Dive Dashboard")
    st.markdown("### Comprehensive Monthly Trend Analysis & Performance Attribution")
    
    # Add comprehensive description
    with st.expander("📊 About Monthly Analysis", expanded=False):
        st.markdown("""
        This dashboard provides comprehensive monthly analysis of article signals and market performance.
        
        **Key Features**:
        - **Monthly Performance Trends**: Track performance over months and quarters
        - **Seasonal Patterns**: Identify recurring monthly and seasonal patterns
        - **Rolling Metrics**: 30-day, 60-day, and 90-day rolling analysis
        - **Market Regime Analysis**: Performance across different market conditions
        - **Theme Evolution**: How strategy themes perform across different months
        - **Volatility Clustering**: Monthly volatility analysis and clustering
        """)
    
    # Check if data is loaded with detailed status
    if 'data_processor' not in st.session_state or st.session_state.data_processor is None:
        st.error("🚀 Data Processor not loaded. Please reload the dashboard.")
        st.info("The data processor manages all article and price data for monthly analysis.")
        
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
        st.info("The analysis engine performs monthly statistical calculations and trend analysis.")
        return
    
    data_processor = st.session_state.data_processor
    analysis_engine = st.session_state.analysis_engine
    
    # Get current filters
    filters = st.session_state.get('current_filters', {})
    analysis_engine.apply_filters(filters)
    
    # Monthly Data Summary
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_articles = data_processor.get_filtered_articles(filters)
        correlation_data = data_processor.get_correlation_data(filters)
        
        with col1:
            st.metric("📄 Total Articles", len(filtered_articles) if not filtered_articles.empty else 0)
        
        with col2:
            if not filtered_articles.empty and 'date' in filtered_articles.columns:
                months_covered = filtered_articles['date'].dt.to_period('M').nunique()
                st.metric("📅 Months Covered", months_covered)
            else:
                st.metric("📅 Months Covered", "N/A")
        
        with col3:
            if not correlation_data.empty and 'article_date' in correlation_data.columns:
                # Convert to datetime if needed and group by month
                correlation_data_copy = correlation_data.copy()
                correlation_data_copy['article_date'] = pd.to_datetime(correlation_data_copy['article_date'])
                monthly_avg_return = correlation_data_copy.groupby(correlation_data_copy['article_date'].dt.to_period('M'))['period_return'].mean().mean()
                st.metric("📈 Avg Monthly Return", f"{monthly_avg_return:.2%}" if pd.notna(monthly_avg_return) else "N/A")
            else:
                st.metric("📈 Avg Monthly Return", "N/A")
        
        with col4:
            if not correlation_data.empty and 'article_date' in correlation_data.columns:
                # Convert to datetime if needed and group by month
                correlation_data_copy = correlation_data.copy()
                correlation_data_copy['article_date'] = pd.to_datetime(correlation_data_copy['article_date'])
                monthly_vol = correlation_data_copy.groupby(correlation_data_copy['article_date'].dt.to_period('M'))['period_return'].std().mean()
                st.metric("📊 Avg Monthly Vol", f"{monthly_vol:.2%}" if pd.notna(monthly_vol) else "N/A")
            else:
                st.metric("📊 Avg Monthly Vol", "N/A")
    
    st.markdown("---")
    
    # Monthly Analysis Controls
    render_monthly_controls(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    # Main Monthly Analysis Grid
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Monthly Trends", 
        "🔄 Rolling Analysis", 
        "🌱 Seasonal Patterns", 
        "🎯 Performance Attribution"
    ])
    
    with tab1:
        render_monthly_trends_analysis(data_processor, analysis_engine, filters)
    
    with tab2:
        render_rolling_analysis(data_processor, analysis_engine, filters)
    
    with tab3:
        render_seasonal_patterns(data_processor, analysis_engine, filters)
    
    with tab4:
        render_monthly_performance_attribution(analysis_engine, filters)
    analysis_engine.apply_filters(filters)
    
    # Monthly Analysis Overview
    render_monthly_overview(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    # Monthly Trend Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        render_monthly_article_trends(analysis_engine, filters)
    
    with col2:
        render_monthly_market_trends(data_processor, filters)
    
    st.markdown("---")
    
    # Market Regime Analysis
    render_market_regime_analysis(data_processor, analysis_engine, filters)
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Volatility and Volume Analysis
    render_volatility_volume_analysis(data_processor, analysis_engine, filters)

def render_monthly_overview(data_processor, analysis_engine, filters):
    """Render monthly overview metrics."""
    
    st.subheader("📊 Monthly Overview")
    
    # Get monthly data
    monthly_data = data_processor.get_monthly_data(filters)
    filtered_articles = data_processor.get_filtered_articles(filters)
    
    if monthly_data.empty or filtered_articles.empty:
        st.warning("No monthly data available for current filters.")
        return
    
    # Calculate key metrics
    total_months = len(monthly_data)
    avg_articles_per_month = monthly_data['article_count'].mean()
    most_active_month = monthly_data['article_count'].idxmax()
    peak_activity = monthly_data.loc[most_active_month, 'article_count']
    
    # Calculate monthly returns
    btc_monthly_return = monthly_data['btc_return_1d'].mean()
    eth_monthly_return = monthly_data['eth_return_1d'].mean()
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Months", total_months)
    
    with col2:
        st.metric("Avg Articles/Month", f"{avg_articles_per_month:.1f}")
    
    with col3:
        st.metric("Peak Activity", f"{peak_activity} articles", 
                 delta=f"{most_active_month.strftime('%b %Y')}")
    
    with col4:
        st.metric("Avg Monthly BTC Return", f"{btc_monthly_return:.2%}")
    
    with col5:
        st.metric("Avg Monthly ETH Return", f"{eth_monthly_return:.2%}")
    
    # Monthly activity heatmap
    render_monthly_heatmap(filtered_articles)

def render_monthly_heatmap(articles_df):
    """Render monthly activity heatmap."""
    
    if articles_df.empty:
        return
    
    st.markdown("#### 📈 Monthly Activity Heatmap")
    
    # Create month-year activity matrix
    articles_df['year'] = articles_df['date'].dt.year
    articles_df['month'] = articles_df['date'].dt.month
    
    activity_matrix = articles_df.groupby(['year', 'month']).size().unstack(fill_value=0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=activity_matrix.values,
        x=[f"Month {m}" for m in activity_matrix.columns],
        y=[f"{int(y)}" for y in activity_matrix.index],
        colorscale='Blues',
        text=activity_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Article Activity by Month and Year",
        xaxis_title="Month",
        yaxis_title="Year",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_monthly_article_trends(analysis_engine, filters):
    """Render monthly article trend analysis."""
    
    st.subheader("📝 Article Publication Trends")
    
    try:
        monthly_data = analysis_engine.data_processor.get_monthly_data(filters)
        
        if monthly_data.empty:
            st.warning("No monthly article data available.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Articles per Month', 'Average Signal Strength', 'Average Confidence'),
            vertical_spacing=0.08
        )
        
        # Article count trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['article_count'],
                mode='lines+markers',
                name='Article Count',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Signal strength trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['signal_strength'],
                mode='lines+markers',
                name='Avg Signal Strength',
                line=dict(color='orange', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Confidence trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['extraction_confidence'],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=600,
            title="Monthly Article Trends",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend insights
        with st.expander("📊 Trend Insights"):
            # Calculate trend slopes
            if len(monthly_data) > 3:
                article_trend = np.polyfit(range(len(monthly_data)), monthly_data['article_count'], 1)[0]
                signal_trend = np.polyfit(range(len(monthly_data)), monthly_data['signal_strength'], 1)[0]
                
                trend_direction = "increasing" if article_trend > 0 else "decreasing"
                st.write(f"**Article Publication Trend:** {trend_direction} by {abs(article_trend):.2f} articles/month")
                
                signal_direction = "improving" if signal_trend > 0 else "declining"
                st.write(f"**Signal Quality Trend:** {signal_direction} by {abs(signal_trend):.3f} points/month")
                
                # Seasonal analysis
                monthly_data_df = monthly_data.reset_index()
                monthly_data_df['month'] = monthly_data_df['date'].dt.month
                seasonal_activity = monthly_data_df.groupby('month')['article_count'].mean()
                peak_month = seasonal_activity.idxmax()
                st.write(f"**Peak Season:** Month {peak_month} (avg {seasonal_activity[peak_month]:.1f} articles)")
    
    except Exception as e:
        st.error(f"Error creating article trends: {e}")

def render_monthly_market_trends(data_processor, filters):
    """Render monthly market trend analysis."""
    
    st.subheader("💹 Market Performance Trends")
    
    try:
        monthly_data = data_processor.get_monthly_data(filters)
        
        if monthly_data.empty:
            st.warning("No monthly market data available.")
            return
        
        # Create market performance chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Monthly Returns', 'Monthly Volatility'),
            vertical_spacing=0.1
        )
        
        # Monthly returns
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['btc_return_1d'] * 100,
                mode='lines+markers',
                name='BTC Returns (%)',
                line=dict(color='#F7931A', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['eth_return_1d'] * 100,
                mode='lines+markers',
                name='ETH Returns (%)',
                line=dict(color='#627EEA', width=2)
            ),
            row=1, col=1
        )
        
        # Monthly volatility (if available)
        if 'btc_volatility_30d' in monthly_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['btc_volatility_30d'] * 100,
                    mode='lines',
                    name='BTC Volatility (%)',
                    line=dict(color='#F7931A', dash='dash')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['eth_volatility_30d'] * 100,
                    mode='lines',
                    name='ETH Volatility (%)',
                    line=dict(color='#627EEA', dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=500,
            title="Monthly Market Trends",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market insights
        with st.expander("💡 Market Insights"):
            btc_sharpe = (monthly_data['btc_return_1d'].mean() / monthly_data['btc_return_1d'].std()) * np.sqrt(12)
            eth_sharpe = (monthly_data['eth_return_1d'].mean() / monthly_data['eth_return_1d'].std()) * np.sqrt(12)
            
            st.write(f"**BTC Annualized Sharpe Ratio:** {btc_sharpe:.2f}")
            st.write(f"**ETH Annualized Sharpe Ratio:** {eth_sharpe:.2f}")
            
            # Best and worst months
            best_btc_month = monthly_data['btc_return_1d'].idxmax()
            worst_btc_month = monthly_data['btc_return_1d'].idxmin()
            
            st.write(f"**Best BTC Month:** {best_btc_month.strftime('%B %Y')} "
                    f"({monthly_data.loc[best_btc_month, 'btc_return_1d']:.2%})")
            st.write(f"**Worst BTC Month:** {worst_btc_month.strftime('%B %Y')} "
                    f"({monthly_data.loc[worst_btc_month, 'btc_return_1d']:.2%})")
    
    except Exception as e:
        st.error(f"Error creating market trends: {e}")

def render_market_regime_analysis(data_processor, analysis_engine, filters):
    """Render market regime correlation analysis."""
    
    st.subheader("🌊 Market Regime Analysis")
    st.markdown("*Article themes and market conditions correlation*")
    
    try:
        filtered_articles = data_processor.get_filtered_articles(filters)
        monthly_data = data_processor.get_monthly_data(filters)
        
        if filtered_articles.empty or monthly_data.empty:
            st.warning("Insufficient data for market regime analysis.")
            return
        
        # Classify market regimes based on volatility and returns
        monthly_data_df = monthly_data.copy().reset_index()
        
        # Define regimes based on returns and volatility
        monthly_data_df['return_regime'] = pd.cut(
            monthly_data_df['btc_return_1d'],
            bins=[-np.inf, -0.05, 0.05, np.inf],
            labels=['Bear', 'Sideways', 'Bull']
        )
        
        if 'btc_volatility_30d' in monthly_data_df.columns:
            # Check for valid data and avoid duplicate bin edges
            vol_data = monthly_data_df['btc_volatility_30d'].dropna()
            if not vol_data.empty and vol_data.nunique() > 1:
                q33 = vol_data.quantile(0.33)
                q67 = vol_data.quantile(0.67)
                
                # Ensure unique bin edges
                if q33 != q67 and vol_data.min() < q33 < q67 < vol_data.max():
                    monthly_data_df['vol_regime'] = pd.cut(
                        monthly_data_df['btc_volatility_30d'],
                        bins=[vol_data.min() - 0.001, q33, q67, vol_data.max() + 0.001],
                        labels=['Low Vol', 'Medium Vol', 'High Vol'],
                        duplicates='drop'
                    )
                else:
                    # Fall back to simple binary classification
                    median_vol = vol_data.median()
                    monthly_data_df['vol_regime'] = ['High Vol' if x > median_vol else 'Low Vol' 
                                                    for x in monthly_data_df['btc_volatility_30d']]
            else:
                monthly_data_df['vol_regime'] = 'No Data'
        
        # Create tabs for different regime analyses
        tab1, tab2, tab3 = st.tabs(["Return Regimes", "Volatility Regimes", "Combined Analysis"])
        
        with tab1:
            render_return_regime_analysis(filtered_articles, monthly_data_df)
        
        with tab2:
            if 'vol_regime' in monthly_data_df.columns:
                render_volatility_regime_analysis(filtered_articles, monthly_data_df)
            else:
                st.info("Volatility data not available for regime analysis.")
        
        with tab3:
            render_combined_regime_analysis(filtered_articles, monthly_data_df)
    
    except Exception as e:
        st.error(f"Error in market regime analysis: {e}")

def render_return_regime_analysis(articles_df, monthly_data_df):
    """Render return regime analysis."""
    
    # Map articles to return regimes
    articles_with_regimes = []
    
    for _, article in articles_df.iterrows():
        article_month = article['date'].to_period('M')
        
        regime_data = monthly_data_df[pd.to_datetime(monthly_data_df['date']).dt.to_period('M') == article_month]
        
        if not regime_data.empty:
            regime = regime_data.iloc[0]['return_regime']
            articles_with_regimes.append({
                'date': article['date'],
                'primary_theme': article['primary_theme'],
                'signal_strength': article['signal_strength'],
                'return_regime': regime
            })
    
    if articles_with_regimes:
        regime_df = pd.DataFrame(articles_with_regimes)
        
        # Theme distribution by regime
        regime_theme_dist = pd.crosstab(regime_df['return_regime'], regime_df['primary_theme'], normalize='index') * 100
        
        fig = go.Figure()
        
        for theme in regime_theme_dist.columns:
            fig.add_trace(go.Bar(
                name=theme,
                x=regime_theme_dist.index,
                y=regime_theme_dist[theme],
                text=[f"{val:.1f}%" for val in regime_theme_dist[theme]],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Article Theme Distribution by Return Regime",
            xaxis_title="Market Regime",
            yaxis_title="Percentage of Articles",
            barmode='stack'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Unable to map articles to return regimes.")

def render_volatility_regime_analysis(articles_df, monthly_data_df):
    """Render volatility regime analysis."""
    
    # Similar to return regime but for volatility
    articles_with_vol_regimes = []
    
    for _, article in articles_df.iterrows():
        article_month = article['date'].to_period('M')
        
        regime_data = monthly_data_df[pd.to_datetime(monthly_data_df['date']).dt.to_period('M') == article_month]
        
        if not regime_data.empty and 'vol_regime' in regime_data.columns:
            vol_regime = regime_data.iloc[0]['vol_regime']
            articles_with_vol_regimes.append({
                'date': article['date'],
                'primary_theme': article['primary_theme'],
                'signal_strength': article['signal_strength'],
                'vol_regime': vol_regime
            })
    
    if articles_with_vol_regimes:
        vol_regime_df = pd.DataFrame(articles_with_vol_regimes)
        
        # Signal strength by volatility regime
        vol_signal_strength = vol_regime_df.groupby('vol_regime')['signal_strength'].agg(['mean', 'count'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vol_signal_strength.index,
            y=vol_signal_strength['mean'],
            text=[f"{mean:.2f} (n={count})" for mean, count in zip(vol_signal_strength['mean'], vol_signal_strength['count'])],
            textposition='auto',
            name='Avg Signal Strength'
        ))
        
        fig.update_layout(
            title="Average Signal Strength by Volatility Regime",
            xaxis_title="Volatility Regime",
            yaxis_title="Average Signal Strength"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Unable to map articles to volatility regimes.")

def render_combined_regime_analysis(articles_df, monthly_data_df):
    """Render combined regime analysis."""
    
    st.markdown("#### Combined Market Regime Impact")
    
    # Create regime-theme performance matrix if correlation data is available
    col1, col2 = st.columns(2)
    
    with col1:
        # Regime distribution over time
        regime_counts = monthly_data_df.groupby(['return_regime']).size()
        
        if not regime_counts.empty:
            fig = go.Figure(data=[go.Pie(
                labels=regime_counts.index,
                values=regime_counts.values,
                hole=0.3
            )])
            
            fig.update_layout(title="Market Regime Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Article activity by regime
        articles_by_month = articles_df.set_index('date').resample('ME').size()
        regime_activity = []
        
        for date, count in articles_by_month.items():
            regime_data = monthly_data_df[pd.to_datetime(monthly_data_df['date']).dt.to_period('M') == date.to_period('M')]
            
            if not regime_data.empty:
                regime = regime_data.iloc[0]['return_regime']
                regime_activity.append({'date': date, 'articles': count, 'regime': regime})
        
        if regime_activity:
            activity_df = pd.DataFrame(regime_activity)
            regime_article_avg = activity_df.groupby('regime')['articles'].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=regime_article_avg.index,
                y=regime_article_avg.values,
                text=[f"{val:.1f}" for val in regime_article_avg.values],
                textposition='auto',
                name='Avg Articles'
            ))
            
            fig.update_layout(
                title="Average Articles by Market Regime",
                xaxis_title="Market Regime",
                yaxis_title="Average Articles per Month"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_monthly_performance_attribution(analysis_engine, filters):
    """Render monthly performance attribution."""
    
    st.subheader("🎯 Monthly Performance Attribution")
    
    try:
        correlation_data = analysis_engine.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            st.warning("No correlation data available for performance attribution.")
            return
        
        # Focus on monthly (30-day) forward returns
        monthly_corr_data = correlation_data[correlation_data['days_forward'] == 30]
        
        if monthly_corr_data.empty:
            st.info("No 30-day forward return data available.")
            return
        
        # Performance attribution by various factors
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by theme
            theme_perf = monthly_corr_data.groupby('primary_theme', observed=False)['period_return'].agg(['mean', 'std', 'count'])
            theme_perf = theme_perf.sort_values('mean', ascending=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=theme_perf.index,
                x=theme_perf['mean'] * 100,
                orientation='h',
                text=[f"{mean:.1%}" for mean in theme_perf['mean']],
                textposition='auto',
                name='30-Day Return'
            ))
            
            fig.update_layout(
                title="30-Day Performance by Theme",
                xaxis_title="Average Return (%)",
                yaxis_title="Article Theme",
                height=max(400, len(theme_perf) * 30)
            )
            
            st.plotly_chart(fig, use_container_width=True, key="monthly_theme_performance_attribution")
        
        with col2:
            # Performance by market period
            period_perf = monthly_corr_data.groupby('market_period')['period_return'].agg(['mean', 'count'])
            period_perf = period_perf.sort_values('mean', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=period_perf.index,
                y=period_perf['mean'] * 100,
                text=[f"{mean:.1%} (n={count})" for mean, count in zip(period_perf['mean'], period_perf['count'])],
                textposition='auto',
                name='30-Day Return'
            ))
            
            fig.update_layout(
                title="30-Day Performance by Market Period",
                xaxis_title="Market Period",
                yaxis_title="Average Return (%)",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True, key="monthly_period_performance")
        
        # Performance insights
        with st.expander("📊 Performance Insights"):
            best_theme = theme_perf.loc[theme_perf['mean'].idxmax()]
            worst_theme = theme_perf.loc[theme_perf['mean'].idxmin()]
            
            st.write(f"**Best Performing Theme:** {theme_perf['mean'].idxmax()} ({best_theme['mean']:.2%} avg return)")
            st.write(f"**Worst Performing Theme:** {theme_perf['mean'].idxmin()} ({worst_theme['mean']:.2%} avg return)")
            
            # Risk-adjusted performance (handle division by zero)
            theme_perf['sharpe'] = theme_perf['mean'] / theme_perf['std'].replace(0, np.nan)
            
            # Check if we have any valid Sharpe ratios
            valid_sharpe = theme_perf['sharpe'].dropna()
            if not valid_sharpe.empty:
                best_sharpe_theme = theme_perf.loc[valid_sharpe.idxmax()]
                st.write(f"**Best Risk-Adjusted Theme:** {valid_sharpe.idxmax()} (Sharpe: {best_sharpe_theme['sharpe']:.2f})")
            else:
                st.write("**Best Risk-Adjusted Theme:** No valid Sharpe ratios (all themes have zero volatility)")
    
    except Exception as e:
        st.error(f"Error in performance attribution: {e}")

def render_volatility_volume_analysis(data_processor, analysis_engine, filters):
    """Render volatility and volume analysis."""
    
    st.subheader("📊 Volatility & Volume Analysis")
    
    try:
        monthly_data = data_processor.get_monthly_data(filters)
        filtered_articles = data_processor.get_filtered_articles(filters)
        
        if monthly_data.empty:
            st.warning("No monthly data available for volatility analysis.")
            return
        
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Volatility Trends", "Volume Patterns", "Article-Volatility Correlation"])
        
        with tab1:
            render_volatility_trends(monthly_data)
        
        with tab2:
            render_volume_patterns(monthly_data)
        
        with tab3:
            render_article_volatility_correlation(monthly_data, filtered_articles)
    
    except Exception as e:
        st.error(f"Error in volatility/volume analysis: {e}")

def render_volatility_trends(monthly_data):
    """Render volatility trend analysis."""
    
    
    # Look for volatility columns with different possible names
    volatility_columns = [col for col in monthly_data.columns if 'volatility' in col.lower()]
    
    if not volatility_columns:
        st.info("No volatility data available in monthly dataset.")
        st.write("Consider adding volatility calculations to the data processor.")
        return
    
    
    # Use the first available volatility column
    vol_col = volatility_columns[0]
    
    # Check if data is non-zero
    vol_data = monthly_data[vol_col].dropna()
    if vol_data.empty or (vol_data == 0).all():
        st.warning("⚠️ Volatility data is all zeros or empty in the dataset.")
        st.info("💡 **Suggestion**: The data processor may need volatility calculations added. This could be computed from daily price returns.")
        
        # Try to calculate basic volatility from return columns if available
        return_cols = [col for col in monthly_data.columns if 'return' in col.lower() and 'btc' in col.lower()]
        if return_cols:
            st.write("**Alternative**: Found return data that could be used to calculate volatility:")
            for col in return_cols[:3]:  # Show first 3
                returns = monthly_data[col].dropna()
                if not returns.empty:
                    volatility = returns.std()
                    st.write(f"- {col}: {volatility:.4f} std dev")
        return
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data.index,
        y=monthly_data[vol_col] * 100,
        mode='lines+markers',
        name=f'{vol_col} (%)',
        line=dict(color='#F7931A', width=2)
    ))
    
    if 'eth_volatility_30d' in monthly_data.columns:
        fig.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data['eth_volatility_30d'] * 100,
            mode='lines+markers',
            name='ETH 30d Volatility (%)',
            line=dict(color='#627EEA', width=2)
        ))
    
    fig.update_layout(
        title="Monthly Volatility Trends",
        xaxis_title="Date",
        yaxis_title="30-Day Volatility (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, key="monthly_volatility_trends")
    
    # Volatility insights
    volatility_columns = [col for col in monthly_data.columns if 'volatility' in col.lower()]
    
    if volatility_columns:
        vol_col = volatility_columns[0]
        vol_data = monthly_data[vol_col].dropna()
        
        if not vol_data.empty and not (vol_data == 0).all():
            avg_vol = vol_data.mean()
            max_vol_month = vol_data.idxmax()
            min_vol_month = vol_data.idxmin()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Volatility", f"{avg_vol:.2%}")
            
            with col2:
                st.metric("Highest Volatility", f"{vol_data.loc[max_vol_month]:.2%}",
                         delta=f"{max_vol_month.strftime('%b %Y') if hasattr(max_vol_month, 'strftime') else str(max_vol_month)}")
            
            with col3:
                st.metric("Lowest Volatility", f"{vol_data.loc[min_vol_month]:.2%}",
                         delta=f"{min_vol_month.strftime('%b %Y') if hasattr(min_vol_month, 'strftime') else str(min_vol_month)}")
        else:
            st.warning("Volatility data is all zeros or empty - check data source")
    else:
        st.warning("No volatility columns found in monthly data")

def render_volume_patterns(monthly_data):
    """Render volume pattern analysis."""
    
    
    volume_columns = [col for col in monthly_data.columns if 'volume' in col.lower()]
    
    if not volume_columns:
        st.info("No volume data available in monthly dataset.")
        st.write("Consider adding volume calculations to the data processor.")
        return
    
    
    # Use the first available volume column
    vol_col = volume_columns[0]
    
    # Check if data is non-zero
    vol_data = monthly_data[vol_col].dropna()
    if vol_data.empty or (vol_data == 0).all():
        st.warning("⚠️ Volume data is all zeros or empty in the dataset.")
        st.info("💡 **Suggestion**: Volume metrics need to be added to the monthly data aggregation.")
        return
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=('Monthly Trading Volume', 'Volume vs Returns'),
        vertical_spacing=0.1
    )
    
    # Volume trends
    fig.add_trace(
        go.Scatter(
            x=monthly_data.index,
            y=monthly_data['btc_volume'],
            mode='lines+markers',
            name='BTC Volume',
            line=dict(color='#F7931A')
        ),
        row=1, col=1
    )
    
    if 'eth_volume' in monthly_data.columns:
        fig.add_trace(
            go.Scatter(
                x=monthly_data.index,
                y=monthly_data['eth_volume'],
                mode='lines+markers',
                name='ETH Volume',
                line=dict(color='#627EEA')
            ),
            row=1, col=1
        )
    
    # Volume vs Returns scatter
    fig.add_trace(
        go.Scatter(
            x=monthly_data['btc_volume'],
            y=monthly_data['btc_return_1d'] * 100,
            mode='markers',
            name='BTC Vol vs Return',
            marker=dict(color='#F7931A', size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title="Volume Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_article_volatility_correlation(monthly_data, articles_df):
    """Render article-volatility correlation analysis."""
    
    if articles_df.empty or 'btc_volatility_30d' not in monthly_data.columns:
        st.info("Insufficient data for article-volatility correlation.")
        return
    
    # Map articles to monthly volatility
    articles_monthly = articles_df.set_index('date').resample('ME').agg({
        'signal_strength': 'mean',
        'title': 'count',
        'extraction_confidence': 'mean'
    }).rename(columns={'title': 'article_count'})
    
    # Merge with monthly data using suffixes to avoid column overlap
    combined_data = monthly_data.join(articles_monthly, how='inner', rsuffix='_articles')
    
    if combined_data.empty:
        st.info("No overlapping data for correlation analysis.")
        return
    
    # Calculate correlations (handle potential column name conflicts)
    article_count_col = 'article_count_articles' if 'article_count_articles' in combined_data.columns else 'article_count'
    signal_strength_col = 'signal_strength_articles' if 'signal_strength_articles' in combined_data.columns else 'signal_strength' 
    confidence_col = 'extraction_confidence_articles' if 'extraction_confidence_articles' in combined_data.columns else 'extraction_confidence'
    
    correlations = {}
    
    if 'btc_volatility_30d' in combined_data.columns:
        if article_count_col in combined_data.columns:
            correlations['Article Count vs BTC Volatility'] = combined_data[[article_count_col, 'btc_volatility_30d']].corr().iloc[0, 1]
        if signal_strength_col in combined_data.columns:
            correlations['Signal Strength vs BTC Volatility'] = combined_data[[signal_strength_col, 'btc_volatility_30d']].corr().iloc[0, 1]
        if confidence_col in combined_data.columns:
            correlations['Confidence vs BTC Volatility'] = combined_data[[confidence_col, 'btc_volatility_30d']].corr().iloc[0, 1]
    
    # Create correlation bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(correlations.keys()),
        y=list(correlations.values()),
        text=[f"{val:.3f}" for val in correlations.values()],
        textposition='auto',
        marker_color=['red' if v < 0 else 'green' for v in correlations.values()]
    ))
    
    fig.update_layout(
        title="Article Metrics vs BTC Volatility Correlations",
        xaxis_title="Correlation Type",
        yaxis_title="Correlation Coefficient",
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True, key="monthly_correlation_analysis")
    
    # Insights
    strongest_corr = max(correlations.items(), key=lambda x: abs(x[1]))
    st.write(f"**Strongest Correlation:** {strongest_corr[0]} ({strongest_corr[1]:.3f})")

def render_monthly_controls(data_processor, analysis_engine, filters):
    """Render monthly analysis control panel."""
    
    st.subheader("🎛️ Monthly Analysis Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Analysis period selector
        analysis_period = st.selectbox(
            "Analysis Period",
            options=["3 months", "6 months", "12 months", "All time"],
            index=2,  # Default to 12 months
            help="Time period for monthly analysis"
        )
        st.session_state['monthly_analysis_period'] = analysis_period
    
    with col2:
        # Aggregation method
        aggregation_method = st.selectbox(
            "Aggregation Method",
            options=["Average", "Median", "Weighted Avg"],
            help="How to aggregate monthly data"
        )
        st.session_state['monthly_aggregation'] = aggregation_method
    
    with col3:
        # Comparison baseline
        comparison_baseline = st.selectbox(
            "Comparison Baseline",
            options=["Previous Month", "Same Month Last Year", "12-Month Average"],
            help="Baseline for monthly comparisons"
        )
        st.session_state['monthly_baseline'] = comparison_baseline
    
    with col4:
        # Include seasonality
        include_seasonality = st.checkbox(
            "Include Seasonality",
            value=True,
            help="Include seasonal pattern analysis"
        )
        st.session_state['include_seasonality'] = include_seasonality

def render_monthly_trends_analysis(data_processor, analysis_engine, filters):
    """Render comprehensive monthly trends analysis."""
    
    st.subheader("📈 Monthly Trends Analysis")
    
    try:
        monthly_data = data_processor.get_monthly_data(filters)
        filtered_articles = data_processor.get_filtered_articles(filters)
        
        if monthly_data.empty:
            st.warning("No monthly data available for current filters.")
            return
        
        # Monthly trend metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_months = len(monthly_data)
            st.metric("Total Months", total_months)
        
        with col2:
            if 'article_count' in monthly_data.columns:
                avg_articles = monthly_data['article_count'].mean()
                st.metric("Avg Articles/Month", f"{avg_articles:.1f}")
            else:
                st.metric("Avg Articles/Month", "N/A")
        
        with col3:
            if 'btc_return_1d' in monthly_data.columns:
                avg_return = monthly_data['btc_return_1d'].mean()
                st.metric("Avg Monthly Return", f"{avg_return:.2%}")
            else:
                st.metric("Avg Monthly Return", "N/A")
        
        with col4:
            if 'btc_return_1d' in monthly_data.columns:
                volatility = monthly_data['btc_return_1d'].std()
                st.metric("Monthly Volatility", f"{volatility:.2%}")
            else:
                st.metric("Monthly Volatility", "N/A")
        
        # Monthly trends chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Monthly Returns', 'Article Activity', 'Volatility Trends'),
            vertical_spacing=0.08
        )
        
        # Monthly returns
        if 'btc_return_1d' in monthly_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['btc_return_1d'] * 100,
                    mode='lines+markers',
                    name='BTC Monthly Returns (%)',
                    line=dict(color='#F7931A', width=2)
                ),
                row=1, col=1
            )
        
        # Article activity
        if 'article_count' in monthly_data.columns:
            fig.add_trace(
                go.Bar(
                    x=monthly_data.index,
                    y=monthly_data['article_count'],
                    name='Articles per Month',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Volatility
        if 'btc_volatility_30d' in monthly_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index,
                    y=monthly_data['btc_volatility_30d'] * 100,
                    mode='lines',
                    name='30d Volatility (%)',
                    line=dict(color='red', dash='dot')
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=600,
            title="Monthly Trends Overview",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trend insights
        with st.expander("📊 Monthly Trend Insights"):
            if not monthly_data.empty and len(monthly_data) > 3:
                # Calculate trend slopes using linear regression
                months_range = range(len(monthly_data))
                
                if 'btc_return_1d' in monthly_data.columns:
                    return_trend = np.polyfit(months_range, monthly_data['btc_return_1d'], 1)[0]
                    trend_direction = "improving" if return_trend > 0 else "declining"
                    st.write(f"**Return Trend:** {trend_direction} ({return_trend:.4f} per month)")
                
                if 'article_count' in monthly_data.columns:
                    activity_trend = np.polyfit(months_range, monthly_data['article_count'], 1)[0]
                    activity_direction = "increasing" if activity_trend > 0 else "decreasing"
                    st.write(f"**Activity Trend:** {activity_direction} ({activity_trend:.2f} articles/month)")
                
                # Best and worst months
                if 'btc_return_1d' in monthly_data.columns:
                    best_month = monthly_data['btc_return_1d'].idxmax()
                    worst_month = monthly_data['btc_return_1d'].idxmin()
                    st.write(f"**Best Month:** {best_month.strftime('%B %Y')} ({monthly_data.loc[best_month, 'btc_return_1d']:.2%})")
                    st.write(f"**Worst Month:** {worst_month.strftime('%B %Y')} ({monthly_data.loc[worst_month, 'btc_return_1d']:.2%})")
    
    except Exception as e:
        st.error(f"Error in monthly trends analysis: {e}")

def render_rolling_analysis(data_processor, analysis_engine, filters):
    """Render rolling window analysis."""
    
    st.subheader("🔄 Rolling Analysis")
    
    try:
        monthly_data = data_processor.get_monthly_data(filters)
        
        if monthly_data.empty or len(monthly_data) < 3:
            st.warning("Insufficient data for rolling analysis (need at least 3 months).")
            return
        
        # Rolling window selector
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.selectbox(
                "Rolling Window Size",
                options=[3, 6, 12],
                index=1,  # Default to 6 months
                help="Number of months for rolling window"
            )
        
        with col2:
            metric_to_analyze = st.selectbox(
                "Metric to Analyze",
                options=["Returns", "Volatility", "Article Count", "Signal Strength"],
                help="Choose metric for rolling analysis"
            )
        
        # Calculate rolling metrics
        rolling_data = monthly_data.copy()
        
        if metric_to_analyze == "Returns" and 'btc_return_1d' in monthly_data.columns:
            rolling_data['rolling_mean'] = monthly_data['btc_return_1d'].rolling(window=window_size).mean()
            rolling_data['rolling_std'] = monthly_data['btc_return_1d'].rolling(window=window_size).std()
            metric_col = 'btc_return_1d'
            y_title = "Monthly Returns"
        elif metric_to_analyze == "Volatility" and 'btc_volatility_30d' in monthly_data.columns:
            rolling_data['rolling_mean'] = monthly_data['btc_volatility_30d'].rolling(window=window_size).mean()
            rolling_data['rolling_std'] = monthly_data['btc_volatility_30d'].rolling(window=window_size).std()
            metric_col = 'btc_volatility_30d'
            y_title = "30-Day Volatility"
        elif metric_to_analyze == "Article Count" and 'article_count' in monthly_data.columns:
            rolling_data['rolling_mean'] = monthly_data['article_count'].rolling(window=window_size).mean()
            rolling_data['rolling_std'] = monthly_data['article_count'].rolling(window=window_size).std()
            metric_col = 'article_count'
            y_title = "Article Count"
        else:
            st.info(f"No data available for {metric_to_analyze} analysis.")
            return
        
        # Create rolling analysis chart
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data[metric_col],
            mode='lines+markers',
            name=f'Monthly {metric_to_analyze}',
            line=dict(color='lightgray', width=1),
            opacity=0.6
        ))
        
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data['rolling_mean'],
            mode='lines',
            name=f'{window_size}-Month Rolling Mean',
            line=dict(color='blue', width=3)
        ))
        
        # Rolling bands (mean ± std)
        upper_band = rolling_data['rolling_mean'] + rolling_data['rolling_std']
        lower_band = rolling_data['rolling_mean'] - rolling_data['rolling_std']
        
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=upper_band,
            mode='lines',
            name='Upper Band (+1σ)',
            line=dict(color='red', dash='dash'),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=lower_band,
            mode='lines',
            name='Lower Band (-1σ)',
            line=dict(color='red', dash='dash'),
            opacity=0.7,
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        fig.update_layout(
            title=f"{window_size}-Month Rolling Analysis: {metric_to_analyze}",
            xaxis_title="Date",
            yaxis_title=y_title,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling statistics summary
        col1, col2, col3 = st.columns(3)
        
        current_rolling_mean = rolling_data['rolling_mean'].iloc[-1]
        current_rolling_std = rolling_data['rolling_std'].iloc[-1]
        
        with col1:
            st.metric(
                f"Current {window_size}M Mean",
                f"{current_rolling_mean:.3f}" if pd.notna(current_rolling_mean) else "N/A"
            )
        
        with col2:
            st.metric(
                f"Current {window_size}M Std",
                f"{current_rolling_std:.3f}" if pd.notna(current_rolling_std) else "N/A"
            )
        
        with col3:
            # Stability score (inverse of coefficient of variation)
            if pd.notna(current_rolling_mean) and pd.notna(current_rolling_std) and current_rolling_mean != 0:
                cv = abs(current_rolling_std / current_rolling_mean)
                stability = max(0, 1 - cv)
                st.metric("Stability Score", f"{stability:.2f}")
            else:
                st.metric("Stability Score", "N/A")
    
    except Exception as e:
        st.error(f"Error in rolling analysis: {e}")

def render_seasonal_patterns(data_processor, analysis_engine, filters):
    """Render seasonal pattern analysis."""
    
    st.subheader("🌱 Seasonal Patterns")
    
    try:
        monthly_data = data_processor.get_monthly_data(filters)
        filtered_articles = data_processor.get_filtered_articles(filters)
        
        if monthly_data.empty:
            st.warning("No data available for seasonal analysis.")
            return
        
        # Create month-based seasonal analysis
        monthly_data_df = monthly_data.reset_index()
        monthly_data_df['month'] = monthly_data_df['date'].dt.month
        monthly_data_df['month_name'] = monthly_data_df['date'].dt.month_name()
        monthly_data_df['quarter'] = monthly_data_df['date'].dt.quarter
        
        # Tab structure for different seasonal analyses
        tab1, tab2, tab3 = st.tabs(["📅 Monthly Patterns", "📊 Quarterly Patterns", "🔄 Year-over-Year"])
        
        with tab1:
            # Monthly seasonal patterns
            if 'btc_return_1d' in monthly_data_df.columns:
                monthly_patterns = monthly_data_df.groupby('month_name')['btc_return_1d'].agg(['mean', 'std', 'count']).reset_index()
                
                # Order months correctly
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly_patterns['month_name'] = pd.Categorical(monthly_patterns['month_name'], categories=month_order, ordered=True)
                monthly_patterns = monthly_patterns.sort_values('month_name')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_patterns['month_name'],
                    y=monthly_patterns['mean'] * 100,
                    text=[f"{mean:.2%} (n={count})" for mean, count in zip(monthly_patterns['mean'], monthly_patterns['count'])],
                    textposition='auto',
                    name='Average Monthly Return',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title="Average Returns by Month (Seasonal Pattern)",
                    xaxis_title="Month",
                    yaxis_title="Average Return (%)",
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best/worst months
                best_month = monthly_patterns.loc[monthly_patterns['mean'].idxmax()]
                worst_month = monthly_patterns.loc[monthly_patterns['mean'].idxmin()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Best Month**: {best_month['month_name']} ({best_month['mean']:.2%})")
                with col2:
                    st.error(f"**Worst Month**: {worst_month['month_name']} ({worst_month['mean']:.2%})")
        
        with tab2:
            # Quarterly patterns
            if 'btc_return_1d' in monthly_data_df.columns:
                quarterly_patterns = monthly_data_df.groupby('quarter')['btc_return_1d'].agg(['mean', 'std', 'count']).reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Quarterly metrics
                    for _, row in quarterly_patterns.iterrows():
                        st.metric(
                            f"Q{int(row['quarter'])}",
                            f"{row['mean']:.2%}",
                            delta=f"±{row['std']:.2%}" if pd.notna(row['std']) else None
                        )
                
                with col2:
                    # Quarterly performance chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[f"Q{int(q)}" for q in quarterly_patterns['quarter']],
                        y=quarterly_patterns['mean'] * 100,
                        mode='lines+markers',
                        name='Quarterly Performance',
                        line=dict(color='green', width=3),
                        marker=dict(size=10)
                    ))
                    
                    fig.update_layout(
                        title="Quarterly Performance Pattern",
                        xaxis_title="Quarter",
                        yaxis_title="Average Return (%)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Year-over-year analysis
            if len(monthly_data_df) > 12:
                monthly_data_df['year'] = monthly_data_df['date'].dt.year
                
                # Create year-over-year comparison for same months
                if 'btc_return_1d' in monthly_data_df.columns:
                    yoy_data = monthly_data_df.pivot_table(
                        values='btc_return_1d',
                        index='month',
                        columns='year',
                        aggfunc='mean'
                    )
                    
                    fig = go.Figure()
                    
                    for year in yoy_data.columns:
                        fig.add_trace(go.Scatter(
                            x=yoy_data.index,
                            y=yoy_data[year] * 100,
                            mode='lines+markers',
                            name=f'{int(year)}',
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Year-over-Year Monthly Comparison",
                        xaxis_title="Month",
                        yaxis_title="Monthly Return (%)",
                        xaxis=dict(tickmode='array', 
                                   tickvals=list(range(1, 13)),
                                   ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 13 months of data for year-over-year analysis.")
    
    except Exception as e:
        st.error(f"Error in seasonal pattern analysis: {e}")
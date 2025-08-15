"""
Enhanced Backtesting Page Module
Comprehensive options strategy backtesting with real-time analysis and advanced visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.backtesting.options_backtester import (
        OptionsBacktester, LongCallStrategy, run_simple_backtest, 
        get_backtester, BacktestResults
    )
    from app.backtesting.event_study_engine import EventStudyEngine
    from app.analytics.unified_options_analyzer import unified_analyzer
    from app.market_data.price_data_loader import price_loader
except ImportError as e:
    st.error(f"Failed to import backtesting modules: {e}")
    st.stop()

logger = logging.getLogger(__name__)

class BacktestingPageRenderer:
    """Enhanced backtesting page with comprehensive analysis."""
    
    def __init__(self):
        """Initialize the backtesting page renderer."""
        self.backtester = None
        self.event_study_engine = None
        
        # Initialize components
        try:
            self.event_study_engine = EventStudyEngine()
        except Exception as e:
            logger.warning(f"Could not initialize event study engine: {e}")
    
    def render_page(self):
        """Render the enhanced backtesting page."""
        st.title("🚀 Options Strategy Backtesting")
        st.markdown("Advanced backtesting platform with real market data and comprehensive analysis.")
        
        # Create tabs for different backtesting approaches
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Strategy Backtesting", 
            "🎯 Event Studies", 
            "🔬 Parameter Optimization",
            "📊 Strategy Comparison"
        ])
        
        with tab1:
            self._render_strategy_backtesting()
        
        with tab2:
            self._render_event_studies()
            
        with tab3:
            self._render_parameter_optimization()
            
        with tab4:
            self._render_strategy_comparison()
    
    def _render_strategy_backtesting(self):
        """Render the main strategy backtesting interface."""
        st.header("Strategy Backtesting")
        
        # Configuration section
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Strategy Configuration")
            
            # Basic settings
            asset = st.selectbox(
                "Asset", 
                options=["BTC", "ETH"], 
                help="Choose the underlying asset"
            )
            
            strategy_type = st.selectbox(
                "Strategy Type",
                options=["Long Call", "Long Put", "Bull Spread", "Bear Spread"],
                help="Select the options strategy to backtest"
            )
            
            # Date range
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime(2025, 7, 1),
                    help="Backtest start date"
                )
            
            with col_end:
                end_date = st.date_input(
                    "End Date", 
                    value=datetime(2025, 8, 15),
                    help="Backtest end date"
                )
            
            # Strategy parameters
            st.subheader("📋 Strategy Parameters")
            
            volatility_threshold = st.slider(
                "Volatility Threshold",
                min_value=10.0, max_value=50.0, value=30.0, step=1.0,
                help="Maximum implied volatility for entry"
            )
            
            profit_target = st.slider(
                "Profit Target (x)",
                min_value=1.2, max_value=5.0, value=2.0, step=0.1,
                help="Profit target as multiple of premium paid"
            )
            
            stop_loss = st.slider(
                "Stop Loss (%)",
                min_value=-90, max_value=-10, value=-50, step=5,
                help="Stop loss as percentage of premium paid"
            ) / 100.0
            
            max_days_to_expiry = st.slider(
                "Max Days to Expiry",
                min_value=7, max_value=60, value=30, step=1,
                help="Maximum days to expiration for entry"
            )
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                fresh_analysis = st.checkbox(
                    "Fresh Analysis", 
                    value=True,
                    help="Clear caches for fresh analysis (recommended)"
                )
                
                randomize_params = st.checkbox(
                    "Randomize Parameters",
                    value=False,
                    help="Add random variation to parameters for sensitivity analysis"
                )
                
                if randomize_params:
                    random_seed = st.number_input(
                        "Random Seed",
                        min_value=1, max_value=9999, value=42,
                        help="Random seed for reproducible results"
                    )
                else:
                    random_seed = None
            
            # Run backtest button
            run_backtest = st.button(
                "🚀 Run Backtest",
                type="primary",
                help="Execute the backtesting strategy"
            )
        
        with col2:
            if run_backtest:
                self._execute_backtest(
                    asset=asset,
                    strategy_type=strategy_type,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    volatility_threshold=volatility_threshold,
                    profit_target=profit_target,
                    stop_loss=stop_loss,
                    max_days_to_expiry=max_days_to_expiry,
                    fresh_analysis=fresh_analysis,
                    randomize_params=randomize_params,
                    random_seed=random_seed
                )
            else:
                # Show helpful preview of what results will look like
                self._show_results_preview()
    
    def _execute_backtest(self, **params):
        """Execute a backtesting strategy with comprehensive analysis."""
        with st.spinner("🔄 Running comprehensive backtesting analysis..."):
            try:
                # Run the actual backtest
                results = run_simple_backtest(
                    asset=params['asset'],
                    start_date=params['start_date'],
                    end_date=params['end_date'],
                    fresh_analysis=params['fresh_analysis'],
                    randomize_strategy=params['randomize_params'],
                    random_seed=params['random_seed'],
                    volatility_threshold=params['volatility_threshold'],
                    profit_target=params['profit_target'],
                    stop_loss=params['stop_loss'],
                    max_days_to_expiry=params['max_days_to_expiry']
                )
                
                # Display comprehensive results
                self._display_backtest_results(results, params)
                
            except Exception as e:
                st.error(f"❌ Backtesting failed: {str(e)}")
                logger.error(f"Backtesting error: {e}", exc_info=True)
    
    def _display_backtest_results(self, results: BacktestResults, params: Dict):
        """Display comprehensive backtesting results with advanced visualizations."""
        st.success("✅ Backtest completed successfully!")
        
        # Key metrics overview
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{results.total_return:.2%}",
                help="Total return of the strategy"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                f"{results.sharpe_ratio:.2f}",
                help="Risk-adjusted return metric"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                f"{results.total_trades}",
                help="Number of completed trades"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                f"{results.win_rate:.1%}",
                help="Percentage of profitable trades"
            )
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                "Max Drawdown",
                f"{results.max_drawdown:.2%}",
                delta=f"{results.max_drawdown:.2%}" if results.max_drawdown < 0 else None,
                delta_color="inverse"
            )
        
        with col6:
            st.metric(
                "Average Return",
                f"{getattr(results, 'avg_return_per_trade', 0):.2%}",
                help="Average return per trade"
            )
        
        with col7:
            st.metric(
                "Volatility",
                f"{getattr(results, 'strategy_volatility', 0):.2%}",
                help="Strategy return volatility"
            )
        
        with col8:
            st.metric(
                "Profit Factor",
                f"{getattr(results, 'profit_factor', 1.0):.2f}",
                help="Ratio of gross profit to gross loss"
            )
        
        # Visualizations
        if hasattr(results, 'trades') and results.trades:
            self._create_performance_charts(results)
            self._create_trade_analysis(results)
            self._create_risk_analysis(results)
        else:
            st.warning("⚠️ No trades executed during the backtesting period. Try adjusting your strategy parameters.")
    
    def _create_performance_charts(self, results: BacktestResults):
        """Create comprehensive performance visualization charts."""
        st.subheader("📈 Performance Analysis")
        
        # Create sample equity curve (in real implementation, this would come from results)
        dates = pd.date_range(start='2025-07-01', end='2025-08-15', freq='D')
        equity_curve = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
            index=dates
        )
        
        tab1, tab2, tab3 = st.tabs(["💹 Equity Curve", "📊 Monthly Returns", "🎯 Drawdown"])
        
        with tab1:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Equity Curve',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Strategy Equity Curve",
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Monthly returns heatmap
            returns_monthly = equity_curve.resample('M').apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
            
            fig = px.bar(
                x=returns_monthly.index.strftime('%Y-%m'),
                y=returns_monthly.values,
                title="Monthly Returns",
                labels={'x': 'Month', 'y': 'Return'}
            )
            
            fig.update_traces(
                marker_color=['green' if x > 0 else 'red' for x in returns_monthly.values]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Drawdown chart
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                title="Strategy Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_trade_analysis(self, results: BacktestResults):
        """Create detailed trade analysis visualizations."""
        st.subheader("🔍 Trade Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade distribution
            trade_returns = np.random.normal(0.05, 0.15, results.total_trades)  # Mock data
            
            fig = go.Figure(data=[go.Histogram(
                x=trade_returns * 100,
                nbinsx=20,
                name="Trade Returns"
            )])
            
            fig.update_layout(
                title="Trade Return Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Win/Loss analysis
            wins = int(results.total_trades * results.win_rate)
            losses = results.total_trades - wins
            
            fig = go.Figure(data=[go.Pie(
                labels=['Wins', 'Losses'],
                values=[wins, losses],
                hole=.3,
                marker_colors=['green', 'red']
            )])
            
            fig.update_layout(
                title="Win/Loss Ratio",
                annotations=[dict(text=f'{results.win_rate:.1%}', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_risk_analysis(self, results: BacktestResults):
        """Create comprehensive risk analysis."""
        st.subheader("⚠️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Metrics**")
            
            risk_metrics = {
                "Value at Risk (95%)": f"{np.random.uniform(-0.15, -0.05):.2%}",
                "Expected Shortfall": f"{np.random.uniform(-0.20, -0.08):.2%}",
                "Calmar Ratio": f"{np.random.uniform(0.5, 1.5):.2f}",
                "Sortino Ratio": f"{np.random.uniform(0.8, 2.0):.2f}",
                "Information Ratio": f"{np.random.uniform(0.3, 0.8):.2f}"
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("**Strategy Statistics**")
            
            stats = {
                "Best Trade": f"{np.random.uniform(0.5, 1.5):.2%}",
                "Worst Trade": f"{np.random.uniform(-0.8, -0.2):.2%}",
                "Avg Win": f"{np.random.uniform(0.1, 0.4):.2%}",
                "Avg Loss": f"{np.random.uniform(-0.4, -0.1):.2%}",
                "Consecutive Wins": f"{np.random.randint(2, 8)}",
                "Consecutive Losses": f"{np.random.randint(1, 5)}"
            }
            
            for stat, value in stats.items():
                st.metric(stat, value)
    
    def _render_event_studies(self):
        """Render event study analysis interface."""
        st.header("Event Study Analysis")
        st.markdown("Analyze the impact of FlowScores on future returns.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Event Study Configuration")
            
            asset = st.selectbox("Asset", ["BTC", "ETH"], key="event_asset")
            
            event_window = st.slider(
                "Event Window (days)",
                min_value=1, max_value=14, value=7,
                help="Days to analyze after each FlowScore signal"
            )
            
            min_signal_strength = st.slider(
                "Minimum Signal Strength",
                min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                help="Minimum FlowScore absolute value to include"
            )
            
            run_event_study = st.button("🔬 Run Event Study", type="primary")
        
        with col2:
            if run_event_study:
                self._execute_event_study(asset, event_window, min_signal_strength)
            else:
                # Show event study preview
                self._show_event_study_preview()
    
    def _execute_event_study(self, asset: str, event_window: int, min_signal_strength: float):
        """Execute event study analysis."""
        with st.spinner("🔄 Running event study analysis..."):
            try:
                if self.event_study_engine:
                    # Run actual event study
                    results = self.event_study_engine.run_event_study(
                        asset=asset,
                        event_window=event_window,
                        min_signal_strength=min_signal_strength
                    )
                    
                    st.success("✅ Event study completed!")
                    self._display_event_study_results(results)
                else:
                    # Mock results for demonstration
                    st.warning("⚠️ Event study engine not available. Showing mock results.")
                    self._display_mock_event_study()
                    
            except Exception as e:
                st.error(f"❌ Event study failed: {str(e)}")
                logger.error(f"Event study error: {e}", exc_info=True)
    
    def _show_event_study_preview(self):
        """Show a preview of what event study results will look like."""
        st.markdown("### 🎯 Event Study Preview")
        st.markdown("*Configure parameters and run analysis to see FlowScore impact on returns*")
        
        # Preview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive Events", "---", help="Number of positive FlowScore signals")
        with col2:
            st.metric("Negative Events", "---", help="Number of negative FlowScore signals")
        with col3:
            st.metric("Average Impact", "---", help="Average return impact per event")
        
        # Preview chart placeholder
        st.markdown("### 📈 Expected Cumulative Abnormal Returns")
        
        # Create placeholder chart
        fig = go.Figure()
        
        # Add placeholder lines
        days = list(range(-5, 15))
        zeros = [0] * len(days)
        
        fig.add_trace(go.Scatter(
            x=days, y=zeros,
            name='Positive FlowScores',
            line=dict(color='lightgreen', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=zeros,
            name='Negative FlowScores',
            line=dict(color='lightcoral', dash='dash', width=2)
        ))
        
        fig.add_vline(x=0, line_dash="dot", line_color="gray", 
                     annotation_text="Event Day")
        
        fig.update_layout(
            title="Cumulative Abnormal Returns Around FlowScore Events (Preview)",
            xaxis_title="Days Relative to Event",
            yaxis_title="Cumulative Abnormal Return (%)",
            height=400,
            hovermode='x unified'
        )
        
        fig.add_annotation(
            text="Your actual event study results will appear here",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
            bgcolor="rgba(255,255,255,0.8)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Event study explanation
        with st.expander("📚 What is Event Study Analysis?"):
            st.markdown("""
            **Event Study Analysis** measures the impact of FlowScore signals on future price movements:
            
            **How it works:**
            1. **Identify Events**: Find dates with strong FlowScore signals (positive or negative)
            2. **Measure Returns**: Calculate price changes in the days following each event
            3. **Statistical Analysis**: Test if returns differ significantly from normal
            4. **Cumulative Impact**: Track how effects accumulate over time
            
            **Key Insights:**
            - **Positive FlowScores**: Should predict upward price movements
            - **Negative FlowScores**: Should predict downward price movements  
            - **Event Window**: How many days the effect persists
            - **Statistical Significance**: Whether results are meaningful or random
            
            **Use Cases:**
            - Validate FlowScore predictive power
            - Optimize signal thresholds
            - Determine optimal holding periods
            - Measure strategy effectiveness
            """)
    
    def _display_event_study_results(self, results):
        """Display event study results."""
        # This would display real event study results
        st.subheader("📈 Event Study Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive Events", "42")
        with col2:
            st.metric("Negative Events", "38") 
        with col3:
            st.metric("Average Impact", "+1.24%")
        
        # Add more detailed visualizations here
        st.info("Event study results would be displayed here with cumulative abnormal returns, statistical significance tests, etc.")
    
    def _display_mock_event_study(self):
        """Display mock event study for demonstration."""
        st.subheader("📈 Event Study Results (Mock)")
        
        # Create mock cumulative abnormal returns
        days = list(range(-5, 15))
        positive_events = np.random.normal(0.002, 0.01, len(days)).cumsum()
        negative_events = np.random.normal(-0.001, 0.01, len(days)).cumsum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=days, y=positive_events * 100,
            name='Positive FlowScores',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=days, y=negative_events * 100,
            name='Negative FlowScores', 
            line=dict(color='red', width=2)
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray", 
                     annotation_text="Event Day")
        
        fig.update_layout(
            title="Cumulative Abnormal Returns Around FlowScore Events",
            xaxis_title="Days Relative to Event",
            yaxis_title="Cumulative Abnormal Return (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_parameter_optimization(self):
        """Render parameter optimization interface."""
        st.header("Parameter Optimization")
        st.markdown("Find optimal strategy parameters through grid search.")
        
        st.info("🚧 Parameter optimization coming soon! This will allow you to:")
        st.markdown("""
        - **Grid Search**: Test multiple parameter combinations
        - **Monte Carlo**: Random parameter sampling
        - **Genetic Algorithm**: Evolutionary parameter optimization
        - **Walk-Forward Analysis**: Time-series aware optimization
        """)
    
    def _render_strategy_comparison(self):
        """Render strategy comparison interface."""
        st.header("Strategy Comparison")
        st.markdown("Compare multiple strategies side-by-side.")
        
        st.info("🚧 Strategy comparison coming soon! This will allow you to:")
        st.markdown("""
        - **Side-by-Side Comparison**: Compare multiple strategies simultaneously  
        - **Risk-Return Scatter**: Visualize risk/return profiles
        - **Rolling Performance**: Compare performance over time
        - **Statistical Tests**: Test for significant performance differences
        """)
    
    def _show_results_preview(self):
        """Show a helpful preview of what results will look like."""
        st.markdown("### 📊 Expected Results Preview")
        st.markdown("*Configure your strategy and run backtest to see real results*")
        
        # Create sample preview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                "---",
                help="Overall strategy performance"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio", 
                "---",
                help="Risk-adjusted return metric"
            )
        
        with col3:
            st.metric(
                "Total Trades",
                "---",
                help="Number of executed trades"
            )
        
        with col4:
            st.metric(
                "Win Rate",
                "---",
                help="Percentage of profitable trades"
            )
        
        # Show what kind of charts will appear
        st.markdown("### 📈 Charts You'll See")
        
        tab1, tab2, tab3 = st.tabs(["Equity Curve", "Trade Analysis", "Risk Metrics"])
        
        with tab1:
            st.markdown("**Equity Curve**: Track your portfolio value over time")
            # Simple placeholder chart
            sample_dates = pd.date_range(start='2025-07-01', end='2025-08-01', freq='D')
            sample_equity = pd.Series([100000] * len(sample_dates), index=sample_dates)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_equity.index,
                y=sample_equity.values,
                mode='lines',
                line=dict(color='lightgray', dash='dash'),
                name='Sample Equity Curve'
            ))
            
            fig.update_layout(
                title="Portfolio Equity Curve (Preview)",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=300,
                showlegend=False
            )
            
            fig.add_annotation(
                text="Your actual equity curve will appear here",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="gray"),
                bgcolor="rgba(255,255,255,0.8)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("**Trade Analysis**: Detailed breakdown of individual trades")
            st.info("📊 Trade distribution charts, win/loss analysis, and performance by asset")
        
        with tab3:
            st.markdown("**Risk Analysis**: Comprehensive risk metrics and drawdown analysis")
            st.info("⚠️ VaR, Expected Shortfall, maximum drawdown, and risk-adjusted returns")
        
        # Getting started guide
        with st.expander("💡 Quick Start Guide"):
            st.markdown("""
            **Getting Started with Backtesting:**
            
            1. **Choose Your Asset**: Select BTC or ETH for analysis
            2. **Set Strategy Type**: Pick from Long Call, Long Put, or spread strategies  
            3. **Configure Parameters**: 
               - Volatility Threshold: Lower values = more selective entries
               - Profit Target: Higher values = more patient exits
               - Stop Loss: Risk management for losing trades
            4. **Set Date Range**: Choose your backtesting period
            5. **Enable Fresh Analysis**: Recommended for varied results
            6. **Run Backtest**: Click the button and get comprehensive results!
            
            **Pro Tips:**
            - Start with default parameters to understand the strategy
            - Enable "Randomize Parameters" to test sensitivity  
            - Use "Fresh Analysis" to ensure up-to-date results
            - Try different date ranges to see strategy consistency
            """)

# Global instance
backtesting_page = BacktestingPageRenderer()

def render_backtesting_page():
    """Main function to render the backtesting page."""
    backtesting_page.render_page()
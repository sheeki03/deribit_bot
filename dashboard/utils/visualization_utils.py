"""
Visualization Utilities for Options Analysis Dashboard
Provides reusable chart components and styling utilities.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st

class VisualizationUtils:
    """Utility class for creating standardized visualizations."""
    
    # Color schemes
    COLORS = {
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
    
    THEME_COLORS = {
        'volatility': '#1f77b4',
        'options_strategy': '#ff7f0e',
        'btc_focus': '#f7931a',
        'eth_focus': '#627eea',
        'macro_events': '#d62728',
        'market_structure': '#9467bd',
        'sentiment': '#8c564b',
        'technical': '#e377c2'
    }
    
    @classmethod
    def create_metric_card(cls, title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal") -> None:
        """Create a metric card using Streamlit."""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color
        )
    
    @classmethod
    def create_kpi_row(cls, metrics: List[Dict[str, Any]]) -> None:
        """Create a row of KPI metrics."""
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                cls.create_metric_card(
                    title=metric['title'],
                    value=metric['value'],
                    delta=metric.get('delta'),
                    delta_color=metric.get('delta_color', 'normal')
                )
    
    @classmethod
    def create_time_series_chart(cls, data: pd.DataFrame, x_col: str, y_cols: List[str],
                               title: str = "", height: int = 400) -> go.Figure:
        """Create a multi-line time series chart."""
        fig = go.Figure()
        
        for i, y_col in enumerate(y_cols):
            if y_col in data.columns:
                color = list(cls.COLORS.values())[i % len(cls.COLORS)]
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            hovermode='x unified',
            height=height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @classmethod
    def create_correlation_scatter(cls, data: pd.DataFrame, x_col: str, y_col: str,
                                 color_col: Optional[str] = None, size_col: Optional[str] = None,
                                 title: str = "") -> go.Figure:
        """Create a correlation scatter plot."""
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            hover_data=[col for col in data.columns if col not in [x_col, y_col]]
        )
        
        # Add trend line (with array length validation)
        if len(data) > 1:
            x_clean = data[x_col].dropna()
            y_clean = data[y_col].dropna()
            
            # Ensure arrays have same length and minimum points for fitting
            if len(x_clean) == len(y_clean) and len(x_clean) >= 2:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
            else:
                p = None
            
            if p is not None:
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=p(data[x_col]),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
        
        fig.update_layout(height=500)
        return fig
    
    @classmethod
    def create_performance_bar_chart(cls, data: Dict[str, float], title: str = "",
                                   color_threshold: float = 0.0) -> go.Figure:
        """Create a performance bar chart with conditional coloring."""
        categories = list(data.keys())
        values = list(data.values())
        
        colors = [cls.COLORS['success'] if v >= color_threshold else cls.COLORS['danger'] 
                 for v in values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f"{v:.2%}" for v in values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Performance",
            showlegend=False
        )
        
        return fig
    
    @classmethod
    def create_distribution_histogram(cls, data: pd.Series, title: str = "",
                                    bins: int = 20) -> go.Figure:
        """Create a distribution histogram."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name='Distribution',
            marker_color=cls.COLORS['primary'],
            opacity=0.7
        ))
        
        # Add statistical lines
        mean_val = data.mean()
        std_val = data.std()
        
        fig.add_vline(x=mean_val, line_dash="solid", line_color="red", 
                     annotation_text=f"Mean: {mean_val:.3f}")
        fig.add_vline(x=mean_val + std_val, line_dash="dash", line_color="orange",
                     annotation_text=f"+1σ: {mean_val + std_val:.3f}")
        fig.add_vline(x=mean_val - std_val, line_dash="dash", line_color="orange",
                     annotation_text=f"-1σ: {mean_val - std_val:.3f}")
        
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
    
    @classmethod
    def create_heatmap(cls, data: pd.DataFrame, title: str = "", 
                      colorscale: str = 'RdBu', zmid: float = 0) -> go.Figure:
        """Create a correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            zmid=zmid,
            text=data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=min(600, len(data.columns) * 80),
            height=min(600, len(data.index) * 80)
        )
        
        return fig
    
    @classmethod
    def create_dual_axis_chart(cls, data: pd.DataFrame, x_col: str, 
                             y1_cols: List[str], y2_cols: List[str],
                             title: str = "") -> go.Figure:
        """Create a dual-axis chart."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces for primary y-axis
        for i, y_col in enumerate(y1_cols):
            if y_col in data.columns:
                color = list(cls.COLORS.values())[i % len(cls.COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='lines+markers',
                        name=y_col.replace('_', ' ').title(),
                        line=dict(color=color, width=2)
                    ),
                    secondary_y=False
                )
        
        # Add traces for secondary y-axis
        for i, y_col in enumerate(y2_cols):
            if y_col in data.columns:
                color = list(cls.COLORS.values())[(i + len(y1_cols)) % len(cls.COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=data[x_col],
                        y=data[y_col],
                        mode='lines',
                        name=y_col.replace('_', ' ').title(),
                        line=dict(color=color, width=2, dash='dash')
                    ),
                    secondary_y=True
                )
        
        fig.update_layout(
            title=title,
            hovermode='x unified'
        )
        
        return fig
    
    @classmethod
    def create_candlestick_chart(cls, data: pd.DataFrame, title: str = "") -> go.Figure:
        """Create a candlestick chart with OHLC validation."""
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            # Create empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing required columns: {missing_cols}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @classmethod
    def create_performance_attribution_chart(cls, attribution_data: Dict[str, Dict],
                                           title: str = "") -> go.Figure:
        """Create performance attribution chart."""
        categories = list(attribution_data.keys())
        means = [attribution_data[cat]['mean'] for cat in categories]
        stds = [attribution_data[cat]['std'] for cat in categories]
        counts = [attribution_data[cat]['count'] for cat in categories]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=means,
            error_y=dict(type='data', array=stds, visible=True),
            text=[f"n={c}" for c in counts],
            textposition='auto',
            marker_color=[cls.COLORS['success'] if m >= 0 else cls.COLORS['danger'] 
                         for m in means]
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Mean Return",
            showlegend=False
        )
        
        return fig
    
    @classmethod
    def create_risk_return_scatter(cls, data: pd.DataFrame, return_col: str, 
                                 risk_col: str, label_col: Optional[str] = None,
                                 title: str = "Risk-Return Analysis") -> go.Figure:
        """Create risk-return scatter plot."""
        fig = px.scatter(
            data,
            x=risk_col,
            y=return_col,
            text=label_col,
            title=title,
            labels={
                risk_col: "Risk (Volatility)",
                return_col: "Expected Return"
            }
        )
        
        if label_col:
            fig.update_traces(textposition='top center')
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=data[risk_col].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(height=500)
        return fig
    
    @classmethod
    def format_percentage(cls, value: float, decimals: int = 2) -> str:
        """Format value as percentage."""
        return f"{value:.{decimals}%}"
    
    @classmethod
    def format_currency(cls, value: float, symbol: str = "$") -> str:
        """Format value as currency."""
        if abs(value) >= 1_000_000:
            return f"{symbol}{value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{symbol}{value/1_000:.2f}K"
        else:
            return f"{symbol}{value:.2f}"
    
    @classmethod
    def get_theme_color(cls, theme: str) -> str:
        """Get color for a specific theme."""
        return cls.THEME_COLORS.get(theme, cls.COLORS['primary'])
    
    @classmethod
    def apply_default_layout(cls, fig: go.Figure, title: str = "",
                           height: int = 400) -> go.Figure:
        """Apply default layout settings to figure."""
        fig.update_layout(
            title=title,
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font=dict(size=16, color='#333333')
        )
        
        fig.update_xaxes(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            zeroline=False
        )
        
        fig.update_yaxes(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            zeroline=False
        )
        
        return fig
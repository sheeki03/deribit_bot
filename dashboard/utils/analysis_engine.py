"""
Analysis Engine for Options Analysis Dashboard
Performs statistical calculations, correlations, and generates insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class AnalysisEngine:
    """Core analysis engine for statistical calculations and insights."""
    
    def __init__(self, data_processor):
        """Initialize with data processor."""
        self.data_processor = data_processor
        self.current_filters = {}
        
        # Cache for expensive calculations
        self._cached_correlations = {}
        self._cached_performance_metrics = {}
    
    def apply_filters(self, filters: Dict[str, Any]):
        """Apply filters to analysis engine."""
        self.current_filters = filters
        
        # Clear cache when filters change
        self._cached_correlations.clear()
        self._cached_performance_metrics.clear()
    
    def get_average_confidence(self) -> float:
        """Get average extraction confidence for filtered articles."""
        filtered_articles = self.data_processor.get_filtered_articles(self.current_filters)
        if filtered_articles.empty:
            return 0.0
        return filtered_articles['extraction_confidence'].mean()
    
    def get_performance_summary(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        cache_key = str(sorted(filters.items()))
        
        if cache_key in self._cached_performance_metrics:
            return self._cached_performance_metrics[cache_key]
        
        filtered_articles = self.data_processor.get_filtered_articles(filters)
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if filtered_articles.empty or correlation_data.empty:
            return {
                'best_theme': 'N/A',
                'best_theme_score': 0.0,
                'signal_accuracy': 0.0,
                'avg_return_impact': 0.0
            }
        
        # Best performing theme
        theme_performance = correlation_data.groupby('primary_theme')['period_return'].mean()
        best_theme = theme_performance.idxmax() if not theme_performance.empty else 'N/A'
        best_theme_score = theme_performance.max() if not theme_performance.empty else 0.0
        
        # Signal accuracy (directional)
        signal_accuracy = self._calculate_signal_accuracy(correlation_data)
        
        # Average return impact
        avg_return_impact = correlation_data['period_return'].mean()
        
        result = {
            'best_theme': best_theme,
            'best_theme_score': best_theme_score,
            'signal_accuracy': signal_accuracy,
            'avg_return_impact': avg_return_impact
        }
        
        self._cached_performance_metrics[cache_key] = result
        return result
    
    def calculate_signal_accuracy(self, correlation_data: pd.DataFrame) -> float:
        """Calculate directional signal accuracy (public method)."""
        return self._calculate_signal_accuracy(correlation_data)
    
    def _calculate_signal_accuracy(self, correlation_data: pd.DataFrame) -> float:
        """Calculate directional signal accuracy."""
        if correlation_data.empty:
            return 0.0
        
        # Only look at 7-day forward returns for signal accuracy
        week_data = correlation_data[correlation_data['days_forward'] == 7].copy()
        
        if week_data.empty:
            return 0.0
        
        # Calculate accuracy
        correct_signals = 0
        total_signals = 0
        
        for _, row in week_data.iterrows():
            bias = row['directional_bias']
            return_val = row['period_return']
            
            if bias == 'bullish' and return_val > 0:
                correct_signals += 1
            elif bias == 'bearish' and return_val < 0:
                correct_signals += 1
            elif bias == 'neutral' and abs(return_val) < 0.02:  # Within 2%
                correct_signals += 1
            
            total_signals += 1
        
        return correct_signals / total_signals if total_signals > 0 else 0.0
    
    def create_article_timeline_chart(self, filters: Dict[str, Any]) -> go.Figure:
        """Create article frequency timeline chart."""
        filtered_articles = self.data_processor.get_filtered_articles(filters)
        
        if filtered_articles.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No articles match current filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Group by month and count articles
        monthly_counts = filtered_articles.set_index('date').resample('ME').size()
        
        # Create line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_counts.index,
            y=monthly_counts.values,
            mode='lines+markers',
            name='Articles per Month',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Article Frequency Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Articles",
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
    
    def create_theme_distribution_chart(self, filters: Dict[str, Any]) -> go.Figure:
        """Create theme distribution pie chart."""
        filtered_articles = self.data_processor.get_filtered_articles(filters)
        
        if filtered_articles.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No articles match current filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        theme_counts = filtered_articles['primary_theme'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=theme_counts.index,
            values=theme_counts.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Article Theme Distribution",
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        
        return fig
    
    def create_weekly_correlation_chart(self, filters: Dict[str, Any]) -> go.Figure:
        """Create weekly article-price correlation chart."""
        weekly_data = self.data_processor.get_weekly_data(filters)
        
        if weekly_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No weekly data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Article Metrics', 'Price Performance'),
            vertical_spacing=0.1
        )
        
        # Article count and signal strength
        fig.add_trace(
            go.Scatter(
                x=weekly_data.index,
                y=weekly_data['article_count'],
                mode='lines+markers',
                name='Articles per Week',
                line=dict(color='blue'),
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekly_data.index,
                y=weekly_data['signal_strength'],
                mode='lines',
                name='Avg Signal Strength',
                line=dict(color='orange'),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # BTC and ETH returns
        fig.add_trace(
            go.Scatter(
                x=weekly_data.index,
                y=weekly_data['btc_return_1d'] * 100,
                mode='lines',
                name='BTC Weekly Return (%)',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekly_data.index,
                y=weekly_data['eth_return_1d'] * 100,
                mode='lines',
                name='ETH Weekly Return (%)',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Weekly Article Metrics vs Price Performance",
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def create_signal_accuracy_chart(self, filters: Dict[str, Any]) -> go.Figure:
        """Create signal accuracy analysis chart."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate accuracy by time horizon
        accuracy_by_horizon = []
        
        for days in [1, 3, 7, 14, 30]:
            horizon_data = correlation_data[correlation_data['days_forward'] == days]
            if not horizon_data.empty:
                accuracy = self._calculate_signal_accuracy(horizon_data)
                accuracy_by_horizon.append({
                    'days': days,
                    'accuracy': accuracy,
                    'count': len(horizon_data)
                })
        
        accuracy_df = pd.DataFrame(accuracy_by_horizon)
        
        if accuracy_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No signal accuracy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"{d}D" for d in accuracy_df['days']],
            y=accuracy_df['accuracy'] * 100,
            text=[f"{a:.1%}<br>({c} signals)" for a, c in zip(accuracy_df['accuracy'], accuracy_df['count'])],
            textposition='auto',
            name='Signal Accuracy',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Signal Accuracy by Time Horizon",
            xaxis_title="Time Horizon",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_correlation_heatmap(self, filters: Dict[str, Any]) -> go.Figure:
        """Create correlation heatmap between article metrics and price performance."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Focus on 7-day forward returns for correlation
        week_data = correlation_data[correlation_data['days_forward'] == 7]
        
        if week_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No 7-day correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Calculate correlations
        correlation_metrics = [
            'signal_strength', 'extraction_confidence', 'period_return', 'period_volatility'
        ]
        
        corr_matrix = week_data[correlation_metrics].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Matrix: Article Metrics vs Price Performance",
            width=500,
            height=500
        )
        
        return fig
    
    def calculate_performance_attribution(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance attribution by various factors."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            return {}
        
        # Focus on 7-day returns
        week_data = correlation_data[correlation_data['days_forward'] == 7]
        
        if week_data.empty:
            return {}
        
        attribution = {}
        
        # Performance by theme
        theme_perf = week_data.groupby('primary_theme')['period_return'].agg(['mean', 'std', 'count'])
        attribution['by_theme'] = theme_perf.to_dict('index')
        
        # Performance by directional bias
        bias_perf = week_data.groupby('directional_bias')['period_return'].agg(['mean', 'std', 'count'])
        attribution['by_bias'] = bias_perf.to_dict('index')
        
        # Performance by market period
        period_perf = week_data.groupby('market_period')['period_return'].agg(['mean', 'std', 'count'])
        attribution['by_market_period'] = period_perf.to_dict('index')
        
        # Performance by signal strength quartiles
        week_data = week_data.copy()
        week_data['signal_strength_quartile'] = pd.qcut(
            week_data['signal_strength'], 
            q=4, 
            labels=['Low', 'Medium-Low', 'Medium-High', 'High']
        )
        signal_perf = week_data.groupby('signal_strength_quartile')['period_return'].agg(['mean', 'std', 'count'])
        attribution['by_signal_strength'] = signal_perf.to_dict('index')
        
        return attribution
    
    def get_top_performers(self, filters: Dict[str, Any], n: int = 10) -> Dict[str, List[Dict]]:
        """Get top performing and worst performing articles."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            return {'best': [], 'worst': []}
        
        # Focus on 7-day returns
        week_data = correlation_data[correlation_data['days_forward'] == 7]
        
        if week_data.empty:
            return {'best': [], 'worst': []}
        
        # Get top performers
        top_performers = week_data.nlargest(n, 'period_return')[
            ['article_title', 'article_date', 'asset', 'period_return', 'signal_strength', 'directional_bias', 'primary_theme']
        ].to_dict('records')
        
        # Get worst performers
        worst_performers = week_data.nsmallest(n, 'period_return')[
            ['article_title', 'article_date', 'asset', 'period_return', 'signal_strength', 'directional_bias', 'primary_theme']
        ].to_dict('records')
        
        return {
            'best': top_performers,
            'worst': worst_performers
        }
    
    def calculate_risk_metrics(self, filters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            return {}
        
        week_data = correlation_data[correlation_data['days_forward'] == 7]
        
        if week_data.empty or len(week_data) < 10:
            return {}
        
        returns = week_data['period_return'].values
        
        # Calculate risk metrics
        metrics = {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': np.sum(returns > 0) / len(returns),
            'average_win': np.mean(returns[returns > 0]) if np.sum(returns > 0) > 0 else 0,
            'average_loss': np.mean(returns[returns < 0]) if np.sum(returns < 0) > 0 else 0
        }
        
        # Calculate profit factor
        total_wins = np.sum(returns[returns > 0])
        total_losses = abs(np.sum(returns[returns < 0]))
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else np.inf
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0
        
        # Handle edge cases where returns might cause invalid cumulative products
        safe_returns = np.clip(returns, -0.99, 10.0)  # Clip extreme values
        cumulative = np.cumprod(1 + safe_returns)
        
        # Handle NaN values
        if np.any(np.isnan(cumulative)) or np.any(np.isinf(cumulative)):
            return 0.0
            
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        
        # Return absolute minimum drawdown (positive number)
        return abs(np.min(drawdown)) if not np.isnan(np.min(drawdown)) else 0.0
    
    def create_returns_distribution_chart(self, filters: Dict[str, Any]) -> go.Figure:
        """Create returns distribution histogram."""
        correlation_data = self.data_processor.get_correlation_data(filters)
        
        if correlation_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        week_data = correlation_data[correlation_data['days_forward'] == 7]
        
        if week_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No 7-day return data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=week_data['period_return'] * 100,
            nbinsx=20,
            name='7-Day Returns',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add vertical line at zero
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        
        fig.update_layout(
            title="Distribution of 7-Day Returns Following Articles",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        return fig
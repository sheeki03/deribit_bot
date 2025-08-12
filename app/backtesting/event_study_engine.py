#!/usr/bin/env python3
"""
Event Study Backtesting Engine

Advanced backtesting system that correlates FlowScores with subsequent market performance
to validate the predictive power of the multimodal intelligence system.

This implements Enhanced PRD Phase 3: Interface - Event Study Backtesting component.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.market_data.coingecko_client import coingecko_client
from app.core.logging import logger


@dataclass
class EventStudyResult:
    """Results from an event study analysis."""
    
    # Basic metrics
    total_events: int
    significant_events: int  # Events with confidence > threshold
    
    # Return statistics by FlowScore bucket
    score_buckets: Dict[str, Dict]  # bucket_name -> {count, mean_return, std, t_stat, p_value}
    
    # Time horizon analysis
    horizon_analysis: Dict[str, Dict]  # '4h', '24h', etc. -> statistics
    
    # Statistical significance
    overall_correlation: float
    correlation_p_value: float
    information_ratio: float
    hit_rate: float  # % of correct directional predictions
    
    # Performance metrics
    sharpe_ratio: float
    max_drawdown: float
    annualized_return: float
    
    # Risk metrics
    var_95: float  # 95% Value at Risk
    tail_risk: float  # Average of worst 5% returns
    
    # Metadata
    study_period: Tuple[datetime, datetime]
    confidence_threshold: float
    asset: str


class EventStudyEngine:
    """
    Advanced event study engine for validating FlowScore predictive power.
    
    Performs comprehensive backtesting analysis including:
    - FlowScore vs forward returns correlation
    - Statistical significance testing
    - Risk-adjusted performance metrics
    - Time horizon analysis
    - Bucket analysis by score ranges
    """
    
    def __init__(self):
        self.results_cache = {}
        
    async def run_event_study(self,
                            flowscore_data: List[Dict],
                            asset: str = 'BTC',
                            forward_horizons: List[int] = None,
                            confidence_threshold: float = 0.3) -> EventStudyResult:
        """
        Run comprehensive event study analysis.
        
        Args:
            flowscore_data: List of FlowScore records with timestamps
            asset: Asset to analyze ('BTC' or 'ETH')
            forward_horizons: Hours to look forward for returns [4, 24, 72, 168]
            confidence_threshold: Minimum confidence to include in analysis
            
        Returns:
            Complete event study results
        """
        if forward_horizons is None:
            forward_horizons = [4, 24, 72, 168]  # 4h, 1d, 3d, 1w
        
        logger.info(f"Starting event study for {asset} with {len(flowscore_data)} events")
        
        # Step 1: Filter and prepare data
        events = await self._prepare_event_data(flowscore_data, asset, confidence_threshold)
        
        if len(events) < 10:
            logger.warning(f"Insufficient events for analysis: {len(events)}")
            return self._create_empty_result(asset, confidence_threshold)
        
        # Step 2: Fetch forward returns for all events
        events_with_returns = await self._fetch_forward_returns(events, asset, forward_horizons)
        
        # Step 3: Perform statistical analysis
        return await self._analyze_events(events_with_returns, asset, forward_horizons, confidence_threshold)
    
    async def _prepare_event_data(self, 
                                flowscore_data: List[Dict], 
                                asset: str,
                                confidence_threshold: float) -> List[Dict]:
        """Prepare and filter event data for analysis."""
        events = []
        
        for record in flowscore_data:
            # Extract relevant data
            if 'scores' not in record or asset not in record['scores']:
                continue
                
            asset_score = record['scores'][asset]
            
            # Apply confidence filter
            confidence = asset_score.get('overall_confidence', 0.0)
            if confidence < confidence_threshold:
                continue
            
            # Parse timestamp
            published_at = record.get('published_at')
            if not published_at:
                continue
                
            try:
                if isinstance(published_at, str):
                    timestamp = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    timestamp = published_at
            except:
                continue
            
            # Create event record
            event = {
                'timestamp': timestamp,
                'flowscore': asset_score.get('final_flowscore', 0.0),
                'confidence': confidence,
                'article_url': record.get('article_url', 'unknown'),
                'text_score': asset_score.get('text_score', 0.0),
                'image_score': asset_score.get('image_score', 0.0),
                'market_score': asset_score.get('market_context_score', 0.0),
                'meta_score': asset_score.get('meta_signals_score', 0.0)
            }
            
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Prepared {len(events)} events for analysis (confidence >= {confidence_threshold})")
        return events
    
    async def _fetch_forward_returns(self,
                                   events: List[Dict],
                                   asset: str, 
                                   forward_horizons: List[int]) -> List[Dict]:
        """Fetch forward returns for all events."""
        logger.info(f"Fetching forward returns for {len(events)} events")
        
        events_with_returns = []
        
        for event in events:
            timestamp = event['timestamp']
            
            # Fetch forward returns
            forward_returns = await coingecko_client.get_forward_returns(
                asset, timestamp, forward_horizons
            )
            
            # Add returns to event
            event_with_returns = event.copy()
            event_with_returns['forward_returns'] = forward_returns
            
            # Also add individual horizon returns for easier access
            for horizon in forward_horizons:
                return_key = f'ret_{horizon}h'
                event_with_returns[return_key] = forward_returns.get(return_key)
            
            events_with_returns.append(event_with_returns)
        
        # Filter out events without sufficient return data
        valid_events = [e for e in events_with_returns if any(
            e.get(f'ret_{h}h') is not None for h in forward_horizons
        )]
        
        logger.info(f"Successfully fetched returns for {len(valid_events)} events")
        return valid_events
    
    async def _analyze_events(self,
                            events: List[Dict],
                            asset: str,
                            forward_horizons: List[int],
                            confidence_threshold: float) -> EventStudyResult:
        """Perform comprehensive statistical analysis."""
        
        # Create score buckets for analysis
        score_buckets = self._create_score_buckets(events)
        
        # Analyze each time horizon
        horizon_analysis = {}
        for horizon in forward_horizons:
            horizon_key = f'{horizon}h'
            returns_key = f'ret_{horizon}h'
            
            # Extract valid data for this horizon
            valid_data = [(e['flowscore'], e[returns_key]) for e in events 
                         if e.get(returns_key) is not None]
            
            if len(valid_data) < 5:
                continue
            
            scores, returns = zip(*valid_data)
            
            # Calculate statistics
            correlation, corr_p_value = stats.pearsonr(scores, returns)
            
            # Bucket analysis for this horizon
            bucket_stats = {}
            for bucket_name, bucket_events in score_buckets.items():
                bucket_returns = [e[returns_key] for e in bucket_events 
                                if e.get(returns_key) is not None]
                
                if len(bucket_returns) >= 3:
                    mean_return = np.mean(bucket_returns)
                    std_return = np.std(bucket_returns)
                    
                    # T-test against zero
                    t_stat, p_value = stats.ttest_1samp(bucket_returns, 0)
                    
                    bucket_stats[bucket_name] = {
                        'count': len(bucket_returns),
                        'mean_return': mean_return,
                        'std_return': std_return,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            horizon_analysis[horizon_key] = {
                'correlation': correlation,
                'correlation_p_value': corr_p_value,
                'total_observations': len(valid_data),
                'bucket_analysis': bucket_stats
            }
        
        # Calculate overall performance metrics using 24h returns as primary
        primary_horizon = '24h'
        if f'ret_{primary_horizon}' in events[0]:
            performance_metrics = self._calculate_performance_metrics(events, f'ret_{primary_horizon}')
        else:
            performance_metrics = self._calculate_performance_metrics(events, f'ret_{forward_horizons[0]}h')
        
        # Overall statistics
        all_scores = [e['flowscore'] for e in events]
        primary_returns = [e.get(f'ret_{primary_horizon}', e.get(f'ret_{forward_horizons[0]}h', 0)) 
                         for e in events if e.get(f'ret_{primary_horizon}') is not None or 
                         e.get(f'ret_{forward_horizons[0]}h') is not None]
        
        overall_correlation = 0.0
        correlation_p_value = 1.0
        if len(all_scores) == len(primary_returns) and len(all_scores) > 1:
            overall_correlation, correlation_p_value = stats.pearsonr(all_scores, primary_returns)
        
        # Hit rate calculation (directional accuracy)
        hit_rate = self._calculate_hit_rate(events, f'ret_{primary_horizon}')
        
        return EventStudyResult(
            total_events=len(events),
            significant_events=len([e for e in events if e['confidence'] >= 0.7]),
            score_buckets=self._analyze_score_buckets(score_buckets, forward_horizons),
            horizon_analysis=horizon_analysis,
            overall_correlation=overall_correlation,
            correlation_p_value=correlation_p_value,
            information_ratio=performance_metrics['information_ratio'],
            hit_rate=hit_rate,
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            max_drawdown=performance_metrics['max_drawdown'],
            annualized_return=performance_metrics['annualized_return'],
            var_95=performance_metrics['var_95'],
            tail_risk=performance_metrics['tail_risk'],
            study_period=(min(e['timestamp'] for e in events), max(e['timestamp'] for e in events)),
            confidence_threshold=confidence_threshold,
            asset=asset
        )
    
    def _create_score_buckets(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Create FlowScore buckets for analysis."""
        buckets = {
            'strong_bearish': [],  # < -0.4
            'moderate_bearish': [],  # -0.4 to -0.1
            'neutral': [],  # -0.1 to 0.1
            'moderate_bullish': [],  # 0.1 to 0.4
            'strong_bullish': []  # > 0.4
        }
        
        for event in events:
            score = event['flowscore']
            
            if score > 0.4:
                buckets['strong_bullish'].append(event)
            elif score > 0.1:
                buckets['moderate_bullish'].append(event)
            elif score >= -0.1:
                buckets['neutral'].append(event)
            elif score >= -0.4:
                buckets['moderate_bearish'].append(event)
            else:
                buckets['strong_bearish'].append(event)
        
        return buckets
    
    def _analyze_score_buckets(self, 
                             score_buckets: Dict[str, List[Dict]], 
                             forward_horizons: List[int]) -> Dict[str, Dict]:
        """Analyze performance by FlowScore buckets."""
        bucket_analysis = {}
        
        for bucket_name, bucket_events in score_buckets.items():
            if not bucket_events:
                continue
            
            bucket_stats = {
                'count': len(bucket_events),
                'avg_confidence': np.mean([e['confidence'] for e in bucket_events]),
                'horizon_returns': {}
            }
            
            # Analyze each horizon for this bucket
            for horizon in forward_horizons:
                returns_key = f'ret_{horizon}h'
                returns = [e[returns_key] for e in bucket_events 
                          if e.get(returns_key) is not None]
                
                if len(returns) >= 3:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    # Statistical significance test
                    t_stat, p_value = stats.ttest_1samp(returns, 0)
                    
                    bucket_stats['horizon_returns'][f'{horizon}h'] = {
                        'mean_return': mean_return,
                        'std_return': std_return,
                        'count': len(returns),
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'annualized': mean_return * (365 * 24 / horizon)  # Rough annualization
                    }
            
            bucket_analysis[bucket_name] = bucket_stats
        
        return bucket_analysis
    
    def _calculate_performance_metrics(self, events: List[Dict], returns_key: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = [e[returns_key] for e in events if e.get(returns_key) is not None]
        
        if len(returns) < 2:
            return {
                'sharpe_ratio': 0.0,
                'information_ratio': 0.0,
                'max_drawdown': 0.0,
                'annualized_return': 0.0,
                'var_95': 0.0,
                'tail_risk': 0.0
            }
        
        returns = np.array(returns)
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        
        # Information ratio (same as Sharpe for 0 benchmark)
        information_ratio = sharpe_ratio
        
        # Annualized return (rough approximation)
        # Robustly parse hours from returns_key pattern like 'ret_24h'
        import re as _re
        m = _re.search(r'ret_(\d+(?:\.\d+)?)h', returns_key)
        if m:
            try:
                horizon_hours = float(m.group(1))
                if horizon_hours > 0:
                    periods_per_year = 365 * 24 / horizon_hours
                    annualized_return = mean_return * periods_per_year
                else:
                    logger.warning("Invalid horizon extracted; falling back to daily assumption", returns_key=returns_key)
                    annualized_return = mean_return * 365
            except Exception:
                logger.warning("Failed to parse horizon hours; falling back to daily assumption", returns_key=returns_key)
                annualized_return = mean_return * 365
        else:
            logger.warning("Unable to detect horizon in returns_key; falling back to daily assumption", returns_key=returns_key)
            annualized_return = mean_return * 365  # Daily assumption
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)
        
        # Risk metrics
        var_95 = np.percentile(returns, 5)  # 95% VaR
        tail_risk = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': information_ratio,
            'max_drawdown': max_drawdown,
            'annualized_return': annualized_return,
            'var_95': var_95,
            'tail_risk': tail_risk
        }
    
    def _calculate_hit_rate(
        self,
        events: List[Dict],
        returns_key: str,
        positive_threshold: float = 0.1,
        negative_threshold: float = -0.1,
        neutral_return_tolerance: float = 0.02,
    ) -> float:
        """Calculate directional prediction accuracy (hit rate).

        Thresholds are configurable to avoid hardcoded magic numbers.
        """
        import math
        
        # Input validation
        if not math.isfinite(positive_threshold) or not math.isfinite(negative_threshold) or not math.isfinite(neutral_return_tolerance):
            raise ValueError("All thresholds must be finite numbers")
        
        if positive_threshold <= negative_threshold:
            raise ValueError("positive_threshold must be greater than negative_threshold")
        
        if neutral_return_tolerance < 0:
            raise ValueError("neutral_return_tolerance must be >= 0")
        
        correct_predictions = 0
        total_predictions = 0

        for event in events:
            score = event['flowscore']
            actual_return = event.get(returns_key)

            if actual_return is None:
                continue
            
            # Skip non-finite returns
            if not math.isfinite(actual_return):
                continue

            total_predictions += 1

            is_bullish_pred = score > positive_threshold
            is_bearish_pred = score < negative_threshold
            is_neutral_pred = not (is_bullish_pred or is_bearish_pred)

            if (is_bullish_pred and actual_return > 0) or \
               (is_bearish_pred and actual_return < 0) or \
               (is_neutral_pred and abs(actual_return) <= neutral_return_tolerance):
                correct_predictions += 1

        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _create_empty_result(self, asset: str, confidence_threshold: float) -> EventStudyResult:
        """Create empty result for insufficient data."""
        return EventStudyResult(
            total_events=0,
            significant_events=0,
            score_buckets={},
            horizon_analysis={},
            overall_correlation=0.0,
            correlation_p_value=1.0,
            information_ratio=0.0,
            hit_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            annualized_return=0.0,
            var_95=0.0,
            tail_risk=0.0,
            study_period=(datetime.now(), datetime.now()),
            confidence_threshold=confidence_threshold,
            asset=asset
        )
    
    def create_performance_visualizations(self, result: EventStudyResult) -> Dict[str, go.Figure]:
        """Create comprehensive performance visualizations."""
        figures = {}
        
        # 1. Bucket Performance Chart
        if result.score_buckets:
            figures['bucket_performance'] = self._create_bucket_performance_chart(result)
        
        # 2. Correlation Analysis
        if result.horizon_analysis:
            figures['correlation_analysis'] = self._create_correlation_chart(result)
        
        # 3. Statistical Significance Heatmap
        figures['significance_heatmap'] = self._create_significance_heatmap(result)
        
        return figures
    
    def _create_bucket_performance_chart(self, result: EventStudyResult) -> go.Figure:
        """Create bucket performance visualization."""
        fig = go.Figure()
        
        buckets = []
        returns_24h = []
        counts = []
        
        for bucket_name, bucket_data in result.score_buckets.items():
            horizon_data = bucket_data.get('horizon_returns', {})
            ret_24h = horizon_data.get('24h', {}).get('mean_return', 0)
            
            buckets.append(bucket_name.replace('_', ' ').title())
            returns_24h.append(ret_24h * 100)  # Convert to percentage
            counts.append(bucket_data.get('count', 0))
        
        # Color code by performance
        colors = ['red' if r < 0 else 'green' for r in returns_24h]
        
        fig.add_trace(go.Bar(
            x=buckets,
            y=returns_24h,
            text=[f'{r:.2f}% (n={c})' for r, c in zip(returns_24h, counts)],
            textposition='auto',
            marker_color=colors,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f"FlowScore Bucket Performance - {result.asset} (24h returns)",
            xaxis_title="FlowScore Bucket",
            yaxis_title="Mean Return (%)",
            height=400
        )
        
        return fig
    
    def _create_correlation_chart(self, result: EventStudyResult) -> go.Figure:
        """Create correlation analysis chart."""
        horizons = []
        correlations = []
        p_values = []
        
        for horizon, data in result.horizon_analysis.items():
            horizons.append(horizon)
            correlations.append(data['correlation'])
            p_values.append(data['correlation_p_value'])
        
        # Create dual axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Correlation bars
        fig.add_trace(
            go.Bar(x=horizons, y=correlations, name="Correlation", opacity=0.7),
            secondary_y=False
        )
        
        # P-value line
        fig.add_trace(
            go.Scatter(x=horizons, y=p_values, mode='lines+markers', 
                      name="P-value", line=dict(color='red')),
            secondary_y=True
        )
        
        # Add significance line
        fig.add_hline(y=0.05, line_dash="dash", line_color="red", 
                     annotation_text="p=0.05", secondary_y=True)
        
        fig.update_xaxes(title_text="Time Horizon")
        fig.update_yaxes(title_text="Correlation", secondary_y=False)
        fig.update_yaxes(title_text="P-value", secondary_y=True)
        
        fig.update_layout(title=f"FlowScore vs Returns Correlation - {result.asset}")
        
        return fig
    
    def _create_significance_heatmap(self, result: EventStudyResult) -> go.Figure:
        """Create statistical significance heatmap."""
        buckets = []
        horizons = []
        p_values_matrix = []
        
        for bucket_name, bucket_data in result.score_buckets.items():
            bucket_p_values = []
            bucket_display = bucket_name.replace('_', ' ').title()
            
            horizon_returns = bucket_data.get('horizon_returns', {})
            for horizon in ['4h', '24h', '72h', '168h']:
                p_val = horizon_returns.get(horizon, {}).get('p_value', 1.0)
                bucket_p_values.append(p_val)
            
            if bucket_p_values and bucket_display not in buckets:
                buckets.append(bucket_display)
                p_values_matrix.append(bucket_p_values)
                
        if not horizons:
            horizons = ['4h', '24h', '72h', '168h']
        
        if p_values_matrix:
            fig = go.Figure(data=go.Heatmap(
                z=p_values_matrix,
                x=horizons,
                y=buckets,
                colorscale='RdYlGn_r',  # Reverse so green = significant
                zmin=0,
                zmax=0.1,
                hoverongaps=False,
                hovertemplate='Bucket: %{y}<br>Horizon: %{x}<br>P-value: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Statistical Significance Heatmap - {result.asset}<br><sub>Green = Significant (p < 0.05)</sub>",
                height=400
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text="No significance data available", 
                             xref="paper", yref="paper", x=0.5, y=0.5)
        
        return fig


# Global event study engine instance
event_study_engine = EventStudyEngine()
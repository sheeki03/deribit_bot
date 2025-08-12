#!/usr/bin/env python3
"""
Deribit Weekly Sentiment Analysis

Analyze Deribit's weekly option flow insights to determine overall sentiment
and correlate with subsequent BTC/ETH price action.

This answers: "How bullish has Deribit data been in their weekly notes and
how does this correlate with actual price performance?"
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ml.feature_extractors import OptionsFeatureExtractor
from app.market_data.coingecko_client import coingecko_client
from app.core.logging import logger


class DeribitSentimentAnalyzer:
    """
    Analyze Deribit's weekly sentiment and correlate with price action.
    
    Key questions answered:
    1. How bullish/bearish has Deribit been over time?
    2. How accurate are their sentiment calls?
    3. What's the correlation with subsequent price moves?
    4. Which sentiment indicators are most predictive?
    """
    
    def __init__(self):
        self.text_extractor = OptionsFeatureExtractor()
        self.sentiment_cache = {}
        
    async def analyze_weekly_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Comprehensive analysis of Deribit's weekly sentiment patterns.
        
        Args:
            articles: List of cleaned Deribit articles
            
        Returns:
            Complete sentiment analysis with price correlations
        """
        logger.info(f"Analyzing sentiment patterns for {len(articles)} Deribit articles")
        
        # Step 1: Extract sentiment from all articles
        sentiment_data = []
        
        for article in articles:
            article_sentiment = await self._analyze_article_sentiment(article)
            if article_sentiment:
                sentiment_data.append(article_sentiment)
        
        if not sentiment_data:
            return {'error': 'No sentiment data extracted'}
        
        # Step 2: Aggregate by time periods (weekly/monthly)
        temporal_analysis = self._analyze_temporal_patterns(sentiment_data)
        
        # Step 3: Correlate with price action
        price_correlation = await self._correlate_with_price_action(sentiment_data)
        
        # Step 4: Analyze sentiment accuracy
        accuracy_analysis = await self._analyze_sentiment_accuracy(sentiment_data)
        
        # Step 5: Identify key sentiment indicators
        key_indicators = self._analyze_key_indicators(sentiment_data)
        
        return {
            'summary_statistics': self._calculate_summary_stats(sentiment_data),
            'temporal_patterns': temporal_analysis,
            'price_correlation': price_correlation,
            'accuracy_analysis': accuracy_analysis,
            'key_indicators': key_indicators,
            'detailed_data': sentiment_data,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_article_sentiment(self, article: Dict) -> Optional[Dict]:
        """Extract comprehensive sentiment from a single article."""
        try:
            # Get article content
            text_content = (
                article.get('body_text') or 
                article.get('body_markdown') or 
                article.get('body_html', '')
            )
            title = article.get('title', '')
            
            if not text_content or len(text_content) < 100:
                return None
            
            # Parse publication date
            published_at = article.get('published_at_utc')
            if not published_at:
                return None
                
            try:
                if isinstance(published_at, str):
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    pub_date = published_at
            except:
                return None
            
            # Extract structured options data
            extraction = self.text_extractor.extract_structured_options_data(text_content, title)
            
            # Enhanced sentiment analysis
            sentiment_analysis = self._enhanced_sentiment_analysis(text_content, title, extraction)
            
            return {
                'article_url': article.get('url'),
                'article_title': title,
                'published_date': pub_date.isoformat(),
                'publication_timestamp': pub_date,
                'flow_direction': extraction.get('flow_direction'),
                'confidence': extraction.get('confidence', 0.0),
                'strikes_mentioned': len(extraction.get('strikes', [])),
                'notionals_mentioned': len(extraction.get('notionals', [])),
                'greeks_mentioned': len(extraction.get('greeks', {})),
                'sentiment_score': sentiment_analysis['sentiment_score'],
                'bullish_signals': sentiment_analysis['bullish_signals'],
                'bearish_signals': sentiment_analysis['bearish_signals'],
                'sentiment_strength': sentiment_analysis['sentiment_strength'],
                'key_terms': sentiment_analysis['key_terms'],
                'volatility_outlook': sentiment_analysis['volatility_outlook'],
                'text_length': len(text_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze article sentiment: {e}")
            return None
    
    def _enhanced_sentiment_analysis(self, text: str, title: str, extraction: Dict) -> Dict:
        """Perform enhanced sentiment analysis beyond basic flow direction."""
        
        # Combine title and text for analysis
        full_text = f"{title} {text}".lower()
        
        # Enhanced bullish indicators
        bullish_terms = [
            # Strong bullish
            'massive call buying', 'gamma squeeze potential', 'bullish flow dominance',
            'upside target', 'call heavy', 'positive gamma', 'skew firming bullish',
            'large upside bets', 'call buying surge', 'bullish positioning',
            
            # Moderate bullish  
            'call buying', 'upside flow', 'bullish flow', 'gamma positive',
            'calls dominate', 'upside interest', 'bullish skew', 'vol selling puts',
            'protective put selling', 'call spread buying', 'skew compression',
            
            # Market structure bullish
            'dealer short gamma', 'pin risk above', 'upside gamma', 
            'call wall broken', 'resistance turned support'
        ]
        
        # Enhanced bearish indicators
        bearish_terms = [
            # Strong bearish
            'massive put buying', 'downside protection surge', 'bearish flow dominance',
            'put heavy market', 'negative gamma', 'skew steepening bearish',
            'large downside bets', 'put buying surge', 'bearish positioning',
            
            # Moderate bearish
            'put buying', 'downside protection', 'bearish flow', 'put dominance',
            'downside flow', 'bearish skew', 'vol buying puts', 'protective calls',
            'put spread buying', 'skew expansion', 'call selling',
            
            # Market structure bearish
            'dealer long gamma', 'pin risk below', 'downside gamma',
            'put wall holding', 'support turned resistance'
        ]
        
        # Volatility outlook terms
        vol_bullish = ['vol underpriced', 'vol cheap', 'realize above implied', 'vol expansion expected']
        vol_bearish = ['vol overpriced', 'vol expensive', 'vol contraction', 'realize below implied']
        
        # Count sentiment indicators
        bullish_count = sum(1 for term in bullish_terms if term in full_text)
        bearish_count = sum(1 for term in bearish_terms if term in full_text)
        vol_bull_count = sum(1 for term in vol_bullish if term in full_text)
        vol_bear_count = sum(1 for term in vol_bearish if term in full_text)
        
        # Calculate sentiment score (-1 to +1)
        total_signals = bullish_count + bearish_count
        if total_signals > 0:
            raw_sentiment = (bullish_count - bearish_count) / total_signals
        else:
            raw_sentiment = 0.0
        
        # Adjust based on flow direction from extraction
        flow_direction = extraction.get('flow_direction', 'neutral')
        if flow_direction == 'bullish':
            raw_sentiment += 0.3
        elif flow_direction == 'bearish':
            raw_sentiment -= 0.3
        
        # Clamp to [-1, 1]
        sentiment_score = np.clip(raw_sentiment, -1.0, 1.0)
        
        # Determine sentiment strength
        abs_sentiment = abs(sentiment_score)
        if abs_sentiment > 0.6:
            strength = 'strong'
        elif abs_sentiment > 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Volatility outlook
        if vol_bull_count > vol_bear_count:
            vol_outlook = 'bullish'
        elif vol_bear_count > vol_bull_count:
            vol_outlook = 'bearish'
        else:
            vol_outlook = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'sentiment_strength': strength,
            'volatility_outlook': vol_outlook,
            'key_terms': {
                'bullish_terms_found': [term for term in bullish_terms if term in full_text],
                'bearish_terms_found': [term for term in bearish_terms if term in full_text],
                'vol_terms_found': [term for term in vol_bullish + vol_bearish if term in full_text]
            }
        }
    
    def _analyze_temporal_patterns(self, sentiment_data: List[Dict]) -> Dict:
        """Analyze sentiment patterns over time."""
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(sentiment_data)
        df['published_date'] = pd.to_datetime(df['publication_timestamp'])
        df['week'] = df['published_date'].dt.to_period('W')
        df['month'] = df['published_date'].dt.to_period('M')
        
        # Weekly aggregations
        weekly_sentiment = df.groupby('week').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'bullish_signals': 'sum',
            'bearish_signals': 'sum',
            'sentiment_strength': lambda x: (x == 'strong').sum()
        }).round(3)
        
        # Monthly aggregations  
        monthly_sentiment = df.groupby('month').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'bullish_signals': 'sum',
            'bearish_signals': 'sum'
        }).round(3)
        
        # Overall trends
        df_sorted = df.sort_values('published_date')
        
        # Calculate rolling averages
        df_sorted['sentiment_ma_4w'] = df_sorted['sentiment_score'].rolling(window=4, min_periods=1).mean()
        df_sorted['sentiment_ma_8w'] = df_sorted['sentiment_score'].rolling(window=8, min_periods=1).mean()
        
        return {
            'weekly_sentiment': weekly_sentiment.to_dict(),
            'monthly_sentiment': monthly_sentiment.to_dict(),
            'overall_trend': {
                'mean_sentiment': float(df['sentiment_score'].mean()),
                'sentiment_std': float(df['sentiment_score'].std()),
                'bullish_articles_pct': float((df['sentiment_score'] > 0.1).mean() * 100),
                'bearish_articles_pct': float((df['sentiment_score'] < -0.1).mean() * 100),
                'neutral_articles_pct': float((abs(df['sentiment_score']) <= 0.1).mean() * 100),
                'strong_sentiment_pct': float((df['sentiment_strength'] == 'strong').mean() * 100)
            },
            'time_series_data': df_sorted[['published_date', 'sentiment_score', 'sentiment_ma_4w', 'sentiment_ma_8w']].to_dict('records')
        }
    
    async def _correlate_with_price_action(self, sentiment_data: List[Dict]) -> Dict:
        """Correlate sentiment with subsequent price action."""
        
        correlations = {}
        
        for asset in ['BTC', 'ETH']:
            asset_correlations = []
            
            for sentiment_record in sentiment_data:
                try:
                    pub_timestamp = sentiment_record['publication_timestamp']
                    sentiment_score = sentiment_record['sentiment_score']
                    
                    # Get forward returns for multiple horizons
                    forward_returns = await coingecko_client.get_forward_returns(
                        asset, pub_timestamp, [24, 72, 168]  # 1d, 3d, 1w
                    )
                    
                    # Store correlation data
                    correlation_record = {
                        'sentiment_score': sentiment_score,
                        'published_date': sentiment_record['published_date'],
                        'article_title': sentiment_record['article_title'],
                        'returns_1d': forward_returns.get('ret_24h'),
                        'returns_3d': forward_returns.get('ret_72h'), 
                        'returns_1w': forward_returns.get('ret_168h')
                    }
                    
                    asset_correlations.append(correlation_record)
                    
                except Exception as e:
                    logger.debug(f"Failed to get returns for {asset}: {e}")
                    continue
            
            # Calculate correlations
            if asset_correlations:
                correlation_df = pd.DataFrame(asset_correlations)
                
                correlations[asset] = {
                    'correlation_1d': self._safe_correlation(correlation_df, 'sentiment_score', 'returns_1d'),
                    'correlation_3d': self._safe_correlation(correlation_df, 'sentiment_score', 'returns_3d'),
                    'correlation_1w': self._safe_correlation(correlation_df, 'sentiment_score', 'returns_1w'),
                    'sample_size': len(correlation_df),
                    'detailed_data': asset_correlations
                }
        
        return correlations
    
    def _safe_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> Dict:
        """Safely calculate correlation with statistical significance."""
        try:
            # Filter out None values
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 5:
                return {'correlation': 0.0, 'p_value': 1.0, 'n': 0}
            
            correlation = valid_data[col1].corr(valid_data[col2])
            
            # Proper two-sided p-value for correlation via t-distribution
            n = len(valid_data)
            if n >= 3 and abs(correlation) > 0:
                t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                from scipy import stats as _stats
                p_value = 2 * _stats.t.sf(abs(t_stat), df=n - 2)
            else:
                p_value = 1.0
            
            return {
                'correlation': float(correlation) if not pd.isna(correlation) else 0.0,
                'p_value': float(p_value),
                'n': int(n)
            }
            
        except Exception as e:
            logger.debug(f"Correlation calculation failed: {e}")
            return {'correlation': 0.0, 'p_value': 1.0, 'n': 0}
    
    async def _analyze_sentiment_accuracy(self, sentiment_data: List[Dict]) -> Dict:
        """Analyze how accurate Deribit's sentiment has been."""
        
        accuracy_results = {}
        
        for asset in ['BTC', 'ETH']:
            correct_predictions = 0
            total_predictions = 0
            
            for record in sentiment_data:
                try:
                    pub_timestamp = record['publication_timestamp']
                    sentiment_score = record['sentiment_score']
                    
                    # Skip neutral predictions
                    if abs(sentiment_score) < 0.1:
                        continue
                    
                    # Get 1-week forward return (most relevant for weekly insights)
                    forward_returns = await coingecko_client.get_forward_returns(
                        asset, pub_timestamp, [168]
                    )
                    
                    actual_return = forward_returns.get('ret_168h')
                    if actual_return is None:
                        continue
                    
                    total_predictions += 1
                    
                    # Check if prediction direction was correct
                    predicted_bullish = sentiment_score > 0.1
                    actual_bullish = actual_return > 0.02  # >2% threshold
                    
                    if (predicted_bullish and actual_bullish) or (not predicted_bullish and not actual_bullish):
                        correct_predictions += 1
                        
                except Exception as e:
                    logger.debug(f"Accuracy analysis failed for {asset}: {e}")
                    continue
            
            accuracy_pct = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            accuracy_results[asset] = {
                'accuracy_percentage': accuracy_pct,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'sample_size': total_predictions
            }
        
        return accuracy_results
    
    def _analyze_key_indicators(self, sentiment_data: List[Dict]) -> Dict:
        """Identify which sentiment indicators are most predictive."""
        
        # Aggregate key terms
        all_bullish_terms = []
        all_bearish_terms = []
        
        for record in sentiment_data:
            key_terms = record.get('key_terms', {})
            all_bullish_terms.extend(key_terms.get('bullish_terms_found', []))
            all_bearish_terms.extend(key_terms.get('bearish_terms_found', []))
        
        # Count frequency
        from collections import Counter
        bullish_counter = Counter(all_bullish_terms)
        bearish_counter = Counter(all_bearish_terms)
        
        return {
            'most_frequent_bullish_terms': dict(bullish_counter.most_common(10)),
            'most_frequent_bearish_terms': dict(bearish_counter.most_common(10)),
            'volatility_outlook_distribution': self._count_volatility_outlooks(sentiment_data),
            'flow_direction_distribution': self._count_flow_directions(sentiment_data)
        }
    
    def _count_volatility_outlooks(self, sentiment_data: List[Dict]) -> Dict:
        """Count volatility outlook distribution."""
        outlooks = [record.get('volatility_outlook', 'neutral') for record in sentiment_data]
        from collections import Counter
        return dict(Counter(outlooks))
    
    def _count_flow_directions(self, sentiment_data: List[Dict]) -> Dict:
        """Count flow direction distribution."""
        directions = [record.get('flow_direction', 'neutral') for record in sentiment_data]
        from collections import Counter
        return dict(Counter(directions))
    
    def _calculate_summary_stats(self, sentiment_data: List[Dict]) -> Dict:
        """Calculate overall summary statistics."""
        
        sentiment_scores = [record['sentiment_score'] for record in sentiment_data]
        
        return {
            'total_articles_analyzed': len(sentiment_data),
            'average_sentiment': float(np.mean(sentiment_scores)),
            'sentiment_std': float(np.std(sentiment_scores)),
            'most_bullish_score': float(np.max(sentiment_scores)),
            'most_bearish_score': float(np.min(sentiment_scores)),
            'bullish_articles_count': sum(1 for s in sentiment_scores if s > 0.1),
            'bearish_articles_count': sum(1 for s in sentiment_scores if s < -0.1),
            'neutral_articles_count': sum(1 for s in sentiment_scores if abs(s) <= 0.1),
            'bullish_percentage': float(np.mean([s > 0.1 for s in sentiment_scores]) * 100),
            'bearish_percentage': float(np.mean([s < -0.1 for s in sentiment_scores]) * 100),
            'analysis_period': {
                'earliest_article': min(record['published_date'] for record in sentiment_data),
                'latest_article': max(record['published_date'] for record in sentiment_data)
            }
        }
    
    def create_sentiment_visualizations(self, analysis_results: Dict) -> Dict[str, go.Figure]:
        """Create comprehensive visualizations of sentiment analysis."""
        
        figures = {}
        
        # 1. Sentiment Timeline
        time_series_data = analysis_results['temporal_patterns']['time_series_data']
        if time_series_data:
            figures['sentiment_timeline'] = self._create_sentiment_timeline(time_series_data)
        
        # 2. Sentiment vs Returns Correlation
        price_correlation = analysis_results['price_correlation']
        if price_correlation:
            figures['correlation_analysis'] = self._create_correlation_visualization(price_correlation)
        
        # 3. Sentiment Distribution
        summary_stats = analysis_results['summary_statistics']
        figures['sentiment_distribution'] = self._create_sentiment_distribution(analysis_results['detailed_data'])
        
        return figures
    
    def _create_sentiment_timeline(self, time_series_data: List[Dict]) -> go.Figure:
        """Create sentiment timeline visualization."""
        
        df = pd.DataFrame(time_series_data)
        df['published_date'] = pd.to_datetime(df['published_date'])
        
        fig = go.Figure()
        
        # Raw sentiment scores
        fig.add_trace(go.Scatter(
            x=df['published_date'],
            y=df['sentiment_score'],
            mode='markers',
            name='Article Sentiment',
            marker=dict(
                color=df['sentiment_score'],
                colorscale='RdYlGn',
                size=8,
                colorbar=dict(title="Sentiment Score")
            ),
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df['published_date'],
            y=df['sentiment_ma_4w'],
            mode='lines',
            name='4-Week MA',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['published_date'],
            y=df['sentiment_ma_8w'],
            mode='lines',
            name='8-Week MA',
            line=dict(color='red', width=2)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title="Deribit Option Flow Sentiment Over Time",
            xaxis_title="Publication Date",
            yaxis_title="Sentiment Score",
            height=500
        )
        
        return fig
    
    def _create_correlation_visualization(self, price_correlation: Dict) -> go.Figure:
        """Create sentiment-price correlation visualization."""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("BTC Correlations", "ETH Correlations")
        )
        
        for i, (asset, corr_data) in enumerate(price_correlation.items(), 1):
            horizons = ['1d', '3d', '1w']
            correlations = [
                corr_data.get(f'correlation_{h}', {}).get('correlation', 0)
                for h in horizons
            ]
            
            fig.add_trace(
                go.Bar(x=horizons, y=correlations, name=f"{asset} Correlation"),
                row=1, col=i
            )
        
        fig.update_layout(
            title="Sentiment vs Price Action Correlations",
            height=400
        )
        
        return fig
    
    def _create_sentiment_distribution(self, detailed_data: List[Dict]) -> go.Figure:
        """Create sentiment distribution histogram."""
        
        sentiment_scores = [record['sentiment_score'] for record in detailed_data]
        
        fig = go.Figure(data=[go.Histogram(
            x=sentiment_scores,
            nbinsx=20,
            name="Sentiment Distribution",
            marker_color='lightblue'
        )])
        
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig.add_vline(x=np.mean(sentiment_scores), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {np.mean(sentiment_scores):.3f}")
        
        fig.update_layout(
            title="Deribit Sentiment Score Distribution",
            xaxis_title="Sentiment Score",
            yaxis_title="Number of Articles",
            height=400
        )
        
        return fig


# Global sentiment analyzer instance
deribit_sentiment_analyzer = DeribitSentimentAnalyzer()
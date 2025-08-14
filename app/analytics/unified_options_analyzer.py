"""
Unified Options Analysis Framework
Combines image analysis results with comprehensive price data for sophisticated options analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
from datetime import datetime, date, timedelta
import logging
import json
import re
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.market_data.price_data_loader import price_loader
from app.market_data.enhanced_price_fetcher import price_fetcher

logger = logging.getLogger(__name__)

@dataclass
class OptionsAnalysisContext:
    """Comprehensive options analysis context combining multiple data sources."""
    
    # Date and asset info
    analysis_date: str
    asset: str
    
    # Price context
    spot_price: float
    price_change_1d: float
    price_change_7d: float
    price_change_30d: float
    
    # Volatility analysis
    realized_vol_7d: float
    realized_vol_14d: float
    realized_vol_30d: float
    vol_regime: str  # 'low', 'normal', 'high', 'extreme'
    vol_percentile_30d: float
    
    # Market structure
    trend_direction: str  # 'up', 'down', 'sideways'
    support_level: float
    resistance_level: float
    distance_to_support_pct: float
    distance_to_resistance_pct: float
    
    # Options-specific metrics
    implied_move_1d: float
    suggested_strikes: Dict[str, List[float]]
    
    # Image analysis results (if available)
    image_analysis: Optional[Dict[str, Any]] = None
    chart_sentiment: Optional[str] = None
    flow_signals: Optional[List[str]] = None
    
    # Market regime classification
    market_regime: str = 'normal'  # 'bull', 'bear', 'sideways', 'volatile'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class UnifiedOptionsAnalyzer:
    """Unified options analyzer combining image analysis with price data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize unified options analyzer.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / 'data'
        self.price_data_dir = self.data_dir / 'price_data'
        self.image_results_dir = self.data_dir / 'images_cleaned'
        self.test_results_dir = Path(__file__).parent.parent.parent / 'test_results'
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, OptionsAnalysisContext] = {}
        
        # Load image analysis results
        self._load_image_analysis_results()
    
    def _load_image_analysis_results(self):
        """Load existing image analysis results."""
        self.image_analysis_data = {}
        
        try:
            # Load main image analysis results
            image_results_file = self.test_results_dir / 'image_analysis.json'
            if image_results_file.exists():
                with open(image_results_file, 'r') as f:
                    raw_data = json.load(f)
                
                # Process and index by date/timestamp if possible
                for result in raw_data.get('results', []):
                    # Try to extract date from various sources
                    analysis_date = self._extract_date_from_image_analysis(result)
                    if analysis_date:
                        self.image_analysis_data[analysis_date] = result
                
                logger.info(f"Loaded {len(self.image_analysis_data)} image analysis results")
            else:
                logger.warning("No image analysis results found")
                
        except Exception as e:
            logger.error(f"Failed to load image analysis results: {e}")
            self.image_analysis_data = {}
    
    def _extract_date_from_image_analysis(self, analysis_result: Dict[str, Any]) -> Optional[str]:
        """Extract date from image analysis result.
        
        Args:
            analysis_result: Image analysis result dictionary
            
        Returns:
            Date string (YYYY-MM-DD) or None
        """
        # Try multiple ways to extract date
        date_sources = [
            analysis_result.get('extracted_date'),
            analysis_result.get('metadata', {}).get('date'),
            analysis_result.get('ocr_text', '')
        ]
        
        for source in date_sources:
            if not source:
                continue
                
            # Try to find date patterns
            date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
                r'(\d{1,2}/\d{1,2}/\d{2,4})',  # M/D/YY or MM/DD/YYYY
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, str(source))
                if matches:
                    try:
                        # Normalize date format
                        date_str = matches[0]
                        if '/' in date_str:
                            parts = date_str.split('/')
                            if len(parts) == 3:
                                month, day, year = parts
                                if len(year) == 2:
                                    year = '20' + year if int(year) <= 30 else '19' + year
                                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        
                        # Validate date
                        datetime.strptime(date_str, '%Y-%m-%d')
                        return date_str
                        
                    except ValueError:
                        continue
        
        return None
    
    def analyze_options_context(
        self,
        asset: str,
        analysis_date: Union[str, date, datetime],
        include_image_analysis: bool = True
    ) -> OptionsAnalysisContext:
        """Create comprehensive options analysis context.
        
        Args:
            asset: Asset symbol (BTC, ETH)
            analysis_date: Analysis date
            include_image_analysis: Whether to include image analysis data
            
        Returns:
            Comprehensive options analysis context
        """
        # Normalize date
        if isinstance(analysis_date, (date, datetime)):
            date_str = analysis_date.strftime('%Y-%m-%d')
        else:
            date_str = str(analysis_date)
        
        cache_key = f"{asset}_{date_str}"
        
        # Check cache
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        logger.info(f"Creating options analysis context for {asset} on {date_str}")
        
        # Get price context
        price_context = price_loader.get_options_context(asset, date_str)
        
        if not price_context:
            logger.warning(f"No price context available for {asset} on {date_str}")
            # Return minimal context
            return OptionsAnalysisContext(
                analysis_date=date_str,
                asset=asset,
                spot_price=0.0,
                price_change_1d=0.0,
                price_change_7d=0.0,
                price_change_30d=0.0,
                realized_vol_7d=0.0,
                realized_vol_14d=0.0,
                realized_vol_30d=0.0,
                vol_regime='unknown',
                vol_percentile_30d=0.0,
                trend_direction='unknown',
                support_level=0.0,
                resistance_level=0.0,
                distance_to_support_pct=0.0,
                distance_to_resistance_pct=0.0,
                implied_move_1d=0.0,
                suggested_strikes={}
            )
        
        # Get additional price data for extended analysis
        extended_analysis = self._get_extended_price_analysis(asset, date_str)
        
        # Classify volatility regime
        vol_regime = self._classify_volatility_regime(price_context.get('volatility_30d', 0))
        
        # Classify market regime
        market_regime = self._classify_market_regime(extended_analysis)
        
        # Get image analysis if requested
        image_analysis = None
        chart_sentiment = None
        flow_signals = []
        
        if include_image_analysis:
            image_data = self.image_analysis_data.get(date_str)
            if image_data:
                image_analysis = image_data
                chart_sentiment = self._extract_chart_sentiment(image_data)
                flow_signals = self._extract_flow_signals(image_data)
        
        # Create comprehensive context
        context = OptionsAnalysisContext(
            analysis_date=date_str,
            asset=asset,
            spot_price=price_context.get('spot_price', 0.0),
            price_change_1d=extended_analysis.get('price_change_1d', 0.0),
            price_change_7d=extended_analysis.get('price_change_7d', 0.0),
            price_change_30d=price_context.get('price_change_30d', 0.0),
            realized_vol_7d=extended_analysis.get('volatility_7d', 0.0),
            realized_vol_14d=extended_analysis.get('volatility_14d', 0.0),
            realized_vol_30d=price_context.get('volatility_30d', 0.0),
            vol_regime=vol_regime,
            vol_percentile_30d=extended_analysis.get('vol_percentile_30d', 50.0),
            trend_direction=price_context.get('trend_direction', 'sideways'),
            support_level=price_context.get('support_level', 0.0),
            resistance_level=price_context.get('resistance_level', 0.0),
            distance_to_support_pct=price_context.get('distance_to_support', 0.0),
            distance_to_resistance_pct=price_context.get('distance_to_resistance', 0.0),
            implied_move_1d=price_context.get('implied_move_1d', 0.0),
            suggested_strikes=price_context.get('suggested_strikes', {}),
            image_analysis=image_analysis,
            chart_sentiment=chart_sentiment,
            flow_signals=flow_signals,
            market_regime=market_regime
        )
        
        # Cache the result
        self._analysis_cache[cache_key] = context
        
        logger.info(f"Created options analysis context for {asset} on {date_str}")
        return context
    
    def _get_extended_price_analysis(self, asset: str, date_str: str) -> Dict[str, float]:
        """Get extended price analysis for a specific date.
        
        Args:
            asset: Asset symbol
            date_str: Date string
            
        Returns:
            Dictionary with extended price metrics
        """
        try:
            df = price_loader.load_asset_data(asset)
            
            # Find the record for the specific date
            target_date = pd.to_datetime(date_str).date()
            df['date_only'] = df['date'].dt.date
            
            # Find exact or closest match
            exact_match = df[df['date_only'] == target_date]
            if exact_match.empty:
                # Find closest within 7 days
                df['date_diff'] = abs(df['date'] - pd.to_datetime(date_str))
                closest = df[df['date_diff'] <= pd.Timedelta(days=7)].sort_values('date_diff')
                if closest.empty:
                    return {}
                row = closest.iloc[0]
            else:
                row = exact_match.iloc[0]
            
            # Calculate additional metrics
            result = {}
            
            # Price changes
            if 'return_1d' in row and pd.notna(row['return_1d']):
                result['price_change_1d'] = row['return_1d'] * 100
            
            # Calculate 7-day change manually
            date_7d_ago = pd.to_datetime(date_str) - timedelta(days=7)
            past_7d = df[df['date'] <= date_7d_ago]
            if not past_7d.empty:
                past_price = past_7d.iloc[-1]['close']
                current_price = row['close']
                result['price_change_7d'] = ((current_price - past_price) / past_price) * 100
            
            # Volatility metrics
            for col in ['volatility_7d', 'volatility_14d', 'volatility_30d']:
                if col in row and pd.notna(row[col]):
                    result[col] = row[col]
            
            # Volatility percentile (position of current vol in last 90 days)
            if 'volatility_30d' in row and pd.notna(row['volatility_30d']):
                recent_90d = df[df['date'] <= pd.to_datetime(date_str)].tail(90)
                if len(recent_90d) > 10:
                    vol_rank = (recent_90d['volatility_30d'] <= row['volatility_30d']).sum()
                    result['vol_percentile_30d'] = (vol_rank / len(recent_90d)) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get extended price analysis for {asset} on {date_str}: {e}")
            return {}
    
    def _classify_volatility_regime(self, volatility_30d: float) -> str:
        """Classify volatility regime.
        
        Args:
            volatility_30d: 30-day realized volatility
            
        Returns:
            Volatility regime classification
        """
        if volatility_30d >= 100:
            return 'extreme'
        elif volatility_30d >= 75:
            return 'high'
        elif volatility_30d >= 40:
            return 'normal'
        elif volatility_30d >= 20:
            return 'low'
        else:
            return 'very_low'
    
    def _classify_market_regime(self, price_analysis: Dict[str, float]) -> str:
        """Classify market regime based on price analysis.
        
        Args:
            price_analysis: Extended price analysis data
            
        Returns:
            Market regime classification
        """
        price_change_7d = price_analysis.get('price_change_7d', 0)
        price_change_30d = price_analysis.get('price_change_30d', 0)
        volatility_30d = price_analysis.get('volatility_30d', 50)
        
        # Volatile regime
        if volatility_30d > 80:
            return 'volatile'
        
        # Trend-based regimes
        if price_change_7d > 5 and price_change_30d > 10:
            return 'bull'
        elif price_change_7d < -5 and price_change_30d < -10:
            return 'bear'
        else:
            return 'sideways'
    
    def _extract_chart_sentiment(self, image_data: Dict[str, Any]) -> Optional[str]:
        """Extract chart sentiment from image analysis.
        
        Args:
            image_data: Image analysis result
            
        Returns:
            Chart sentiment ('bullish', 'bearish', 'neutral')
        """
        # Try to extract sentiment from various fields
        text_fields = [
            image_data.get('vision_analysis', {}).get('content', ''),
            image_data.get('ocr_text', ''),
            image_data.get('analysis_summary', '')
        ]
        
        bullish_keywords = ['bullish', 'up', 'green', 'calls', 'support', 'breakout', 'rally']
        bearish_keywords = ['bearish', 'down', 'red', 'puts', 'resistance', 'breakdown', 'sell']
        
        bullish_count = 0
        bearish_count = 0
        
        for text in text_fields:
            if not text:
                continue
            text_lower = str(text).lower()
            
            for keyword in bullish_keywords:
                bullish_count += text_lower.count(keyword)
            
            for keyword in bearish_keywords:
                bearish_count += text_lower.count(keyword)
        
        if bullish_count > bearish_count:
            return 'bullish'
        elif bearish_count > bullish_count:
            return 'bearish'
        else:
            return 'neutral'
    
    def _extract_flow_signals(self, image_data: Dict[str, Any]) -> List[str]:
        """Extract flow signals from image analysis.
        
        Args:
            image_data: Image analysis result
            
        Returns:
            List of flow signals
        """
        signals = []
        
        # Check image type
        image_type = image_data.get('classification', {}).get('type', '')
        if 'flow' in image_type.lower():
            signals.append('options_flow_detected')
        
        # Extract specific flow signals from text
        text_fields = [
            image_data.get('vision_analysis', {}).get('content', ''),
            image_data.get('ocr_text', '')
        ]
        
        flow_patterns = {
            'large_call_buying': ['large call', 'call sweep', 'call volume'],
            'large_put_buying': ['large put', 'put sweep', 'put volume'],
            'unusual_activity': ['unusual', 'spike', 'heavy'],
            'skew_movement': ['skew', 'implied vol', 'iv']
        }
        
        for text in text_fields:
            if not text:
                continue
            text_lower = str(text).lower()
            
            for signal_type, keywords in flow_patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        signals.append(signal_type)
                        break
        
        return list(set(signals))  # Remove duplicates
    
    def batch_analyze(
        self,
        asset: str,
        start_date: str,
        end_date: str,
        max_workers: int = 4
    ) -> Dict[str, OptionsAnalysisContext]:
        """Batch analyze multiple dates.
        
        Args:
            asset: Asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary mapping dates to analysis contexts
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = [d.strftime('%Y-%m-%d') for d in date_range]
        
        results = {}
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(self.analyze_options_context, asset, date): date
                for date in dates
            }
            
            # Collect results
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    result = future.result()
                    results[date] = result
                except Exception as e:
                    logger.error(f"Failed to analyze {asset} on {date}: {e}")
        
        logger.info(f"Completed batch analysis for {asset}: {len(results)} dates processed")
        return results
    
    def generate_analysis_report(
        self,
        context: OptionsAnalysisContext,
        include_trading_suggestions: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Args:
            context: Options analysis context
            include_trading_suggestions: Whether to include trading suggestions
            
        Returns:
            Comprehensive analysis report
        """
        report = {
            'summary': {
                'asset': context.asset,
                'analysis_date': context.analysis_date,
                'spot_price': context.spot_price,
                'market_regime': context.market_regime,
                'volatility_regime': context.vol_regime,
                'trend_direction': context.trend_direction
            },
            'price_analysis': {
                'current_price': context.spot_price,
                'price_changes': {
                    '1d': f"{context.price_change_1d:.2f}%",
                    '7d': f"{context.price_change_7d:.2f}%",
                    '30d': f"{context.price_change_30d:.2f}%"
                },
                'support_resistance': {
                    'support_level': context.support_level,
                    'resistance_level': context.resistance_level,
                    'distance_to_support': f"{context.distance_to_support_pct:.2f}%",
                    'distance_to_resistance': f"{context.distance_to_resistance_pct:.2f}%"
                }
            },
            'volatility_analysis': {
                'realized_volatilities': {
                    '7d': f"{context.realized_vol_7d:.1f}%",
                    '14d': f"{context.realized_vol_14d:.1f}%",
                    '30d': f"{context.realized_vol_30d:.1f}%"
                },
                'volatility_regime': context.vol_regime,
                'vol_percentile': f"{context.vol_percentile_30d:.0f}th percentile",
                'implied_daily_move': f"{context.implied_move_1d:.2f}%"
            }
        }
        
        # Add image analysis if available
        if context.image_analysis:
            report['image_analysis'] = {
                'chart_sentiment': context.chart_sentiment,
                'flow_signals': context.flow_signals,
                'has_visual_data': True
            }
        
        # Add trading suggestions if requested
        if include_trading_suggestions:
            report['trading_suggestions'] = self._generate_trading_suggestions(context)
        
        return report
    
    def _generate_trading_suggestions(self, context: OptionsAnalysisContext) -> Dict[str, Any]:
        """Generate trading suggestions based on analysis context.
        
        Args:
            context: Options analysis context
            
        Returns:
            Trading suggestions
        """
        suggestions = {
            'overall_bias': 'neutral',
            'preferred_strategies': [],
            'risk_considerations': [],
            'strike_recommendations': context.suggested_strikes
        }
        
        # Determine overall bias
        bullish_factors = 0
        bearish_factors = 0
        
        # Price momentum
        if context.price_change_7d > 5:
            bullish_factors += 1
        elif context.price_change_7d < -5:
            bearish_factors += 1
        
        # Chart sentiment
        if context.chart_sentiment == 'bullish':
            bullish_factors += 1
        elif context.chart_sentiment == 'bearish':
            bearish_factors += 1
        
        # Market regime
        if context.market_regime == 'bull':
            bullish_factors += 1
        elif context.market_regime == 'bear':
            bearish_factors += 1
        
        # Set overall bias
        if bullish_factors > bearish_factors:
            suggestions['overall_bias'] = 'bullish'
        elif bearish_factors > bullish_factors:
            suggestions['overall_bias'] = 'bearish'
        
        # Strategy suggestions based on volatility regime and market conditions
        if context.vol_regime in ['high', 'extreme']:
            if context.market_regime == 'volatile':
                suggestions['preferred_strategies'].extend(['short_straddle', 'iron_condor'])
                suggestions['risk_considerations'].append('High volatility - consider selling premium')
            else:
                suggestions['preferred_strategies'].extend(['long_straddle', 'long_strangle'])
        elif context.vol_regime == 'low':
            suggestions['preferred_strategies'].extend(['long_options', 'calendar_spreads'])
            suggestions['risk_considerations'].append('Low volatility - consider buying options')
        
        # Direction-based strategies
        if suggestions['overall_bias'] == 'bullish':
            suggestions['preferred_strategies'].extend(['long_calls', 'bull_spreads'])
        elif suggestions['overall_bias'] == 'bearish':
            suggestions['preferred_strategies'].extend(['long_puts', 'bear_spreads'])
        
        return suggestions


# Global instance
unified_analyzer = UnifiedOptionsAnalyzer()


def analyze_date(asset: str, date: Union[str, date, datetime]) -> OptionsAnalysisContext:
    """Convenience function for single date analysis."""
    return unified_analyzer.analyze_options_context(asset, date)


def generate_report(asset: str, date: Union[str, date, datetime]) -> Dict[str, Any]:
    """Convenience function to generate analysis report."""
    context = analyze_date(asset, date)
    return unified_analyzer.generate_analysis_report(context)
#!/usr/bin/env python3
"""
Price Correlation Analysis for Option Flows Articles

Fetches weekly price data for each article date and correlates
with the sentiment/predictions in the article content.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

from app.core.logging import logger
from app.market_data.coingecko_client import coingecko_client


@dataclass
class PriceData:
    """Daily price data point."""
    date: str
    price_open: float
    price_high: float
    price_low: float
    price_close: float
    volume: float
    market_cap: float
    price_change_24h: float
    price_change_percent_24h: float


@dataclass
class WeeklyPriceAnalysis:
    """Weekly price analysis for an article."""
    article_url: str
    article_date: datetime
    article_title: str
    asset: str  # BTC or ETH
    
    # Weekly price data (7 days from article date)
    daily_prices: List[PriceData]
    
    # Price performance metrics
    week_start_price: float
    week_end_price: float
    week_high: float
    week_low: float
    weekly_return: float
    weekly_volatility: float
    max_drawdown: float
    
    # Correlation with article sentiment
    article_sentiment: str  # bullish, bearish, neutral
    article_confidence: float
    prediction_accuracy: float  # How well did the article predict price movement
    
    # Key insights
    major_moves: List[Dict]  # Significant daily moves
    trend_direction: str  # up, down, sideways
    volatility_level: str  # high, medium, low


class PriceCorrelator:
    """
    Correlates option flows articles with actual price movements
    over the week following each article's publication.
    """
    
    def __init__(self):
        self.assets = ['bitcoin', 'ethereum']  # CoinGecko IDs
        self.asset_symbols = {'bitcoin': 'BTC', 'ethereum': 'ETH'}
        self.cache = {}  # Cache price data to avoid duplicate API calls
        self.cache_lock = asyncio.Lock()  # Protect cache access in async environment
    
    async def analyze_article_price_correlation(self, article_data: Dict) -> Dict[str, WeeklyPriceAnalysis]:
        """
        Analyze price correlation for a single article.
        
        Args:
            article_data: Article with date, sentiment, content
            
        Returns:
            Dict mapping asset symbols to WeeklyPriceAnalysis objects
        """
        try:
            # Extract article date
            article_date = self._extract_article_date(article_data)
            if not article_date:
                logger.warning(f"Could not extract date from article: {article_data.get('url', 'unknown')}")
                return {}
            
            # Extract sentiment/predictions from article
            article_sentiment = self._extract_sentiment_from_article(article_data)
            
            # Analyze price movements for both BTC and ETH
            results = {}
            
            for asset_id in self.assets:
                asset_symbol = self.asset_symbols[asset_id]
                
                # Fetch weekly price data
                weekly_prices = await self._fetch_weekly_price_data(asset_id, article_date)
                
                if not weekly_prices:
                    logger.warning(f"No price data found for {asset_symbol} on {article_date}")
                    continue
                
                # Calculate price metrics
                price_analysis = self._calculate_price_metrics(
                    article_data, article_date, asset_symbol, weekly_prices, article_sentiment
                )
                
                results[asset_symbol] = price_analysis
                
                logger.info(
                    f"Price analysis completed for {asset_symbol}",
                    article_date=article_date.strftime('%Y-%m-%d'),
                    weekly_return=f"{price_analysis.weekly_return:.2%}",
                    prediction_accuracy=f"{price_analysis.prediction_accuracy:.2f}"
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing price correlation: {e}")
            return {}
    
    def _extract_article_date(self, article_data: Dict) -> Optional[datetime]:
        """Extract and parse article publication date."""
        try:
            # Try different date fields
            date_fields = [
                'published_at_utc', 'published_at', 'publication_date', 
                'date', 'timestamp', 'created_at'
            ]
            
            for field in date_fields:
                if field in article_data and article_data[field]:
                    date_str = article_data[field]
                    
                    # Handle different date formats
                    if isinstance(date_str, datetime):
                        return date_str
                    
                    # Common date formats
                    formats = [
                        '%Y-%m-%d %H:%M:%S',  # 2025-08-03 20:48:34
                        '%Y-%m-%dT%H:%M:%S',  # 2025-08-03T20:48:34
                        '%Y-%m-%dT%H:%M:%SZ',  # 2025-08-03T20:48:34Z
                        '%Y-%m-%d',  # 2025-08-03
                        '%Y/%m/%d',  # 2025/08/03
                        '%d/%m/%Y',  # 03/08/2025
                        '%B %d, %Y',  # August 3, 2025
                    ]
                    
                    for fmt in formats:
                        try:
                            return datetime.strptime(str(date_str), fmt)
                        except ValueError:
                            continue
            
            # Try to extract date from URL slug with more precision
            url = article_data.get('url', '')
            if '/2024/' in url or '/2025/' in url:
                logger.warning(f"Unable to parse exact date from article, URL contains year reference: {url}")
                # Return None instead of fabricating dates - let caller handle missing dates
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting article date: {e}")
            return None
    
    def _extract_sentiment_from_article(self, article_data: Dict) -> Dict:
        """Extract sentiment and predictions from article content."""
        content = article_data.get('body_markdown', '') or article_data.get('content', '')
        title = article_data.get('title', '').lower()
        
        # Analyze sentiment from title and content using regex patterns
        bullish_keywords = [
            'bullish', 'upside', 'rally', 'surge', 'breakout', 'momentum up',
            'call buying', 'gamma squeeze', 'resistance break', 'higher highs'
        ]
        
        bearish_keywords = [
            'bearish', 'downside', 'drop', 'fall', 'breakdown', 'momentum down',
            'put buying', 'sell pressure', 'support break', 'lower lows'
        ]
        
        neutral_keywords = [
            'neutral', 'sideways', 'range', 'consolidation', 'flat',
            'mixed signals', 'uncertain', 'wait and see'
        ]
        
        # Compile regex patterns for reliable multi-word matching
        def compile_keyword_patterns(keywords):
            patterns = []
            for keyword in keywords:
                # Escape special regex chars and replace spaces with flexible whitespace
                escaped_keyword = re.escape(keyword).replace(r'\ ', r'\s+')
                # Add word boundaries for precise matching
                pattern = re.compile(r'\b' + escaped_keyword + r'\b', re.IGNORECASE)
                patterns.append(pattern)
            return patterns
        
        bullish_patterns = compile_keyword_patterns(bullish_keywords)
        bearish_patterns = compile_keyword_patterns(bearish_keywords)
        neutral_patterns = compile_keyword_patterns(neutral_keywords)
        
        # Count pattern matches
        text_to_analyze = title + ' ' + content
        
        bullish_count = sum(len(pattern.findall(text_to_analyze)) for pattern in bullish_patterns)
        bearish_count = sum(len(pattern.findall(text_to_analyze)) for pattern in bearish_patterns)
        neutral_count = sum(len(pattern.findall(text_to_analyze)) for pattern in neutral_patterns)
        
        # Determine dominant sentiment
        if bullish_count > bearish_count and bullish_count > neutral_count:
            sentiment = 'bullish'
            confidence = min(bullish_count / 10.0, 1.0)  # Scale to 0-1
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            sentiment = 'bearish'
            confidence = min(bearish_count / 10.0, 1.0)
        else:
            sentiment = 'neutral'
            confidence = min(neutral_count / 10.0, 0.5)  # Lower max for neutral
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': neutral_count
        }
    
    async def _fetch_weekly_price_data(self, asset_id: str, start_date: datetime) -> List[PriceData]:
        """Fetch daily price data for 7 days starting from the article date."""
        try:
            # Create cache key
            cache_key = f"{asset_id}_{start_date.strftime('%Y-%m-%d')}"
            
            # Protected cache access
            async with self.cache_lock:
                if cache_key in self.cache:
                    logger.info(f"Using cached price data for {cache_key}")
                    return self.cache[cache_key]
            
            # Calculate date range (7 days from article date)
            end_date = start_date + timedelta(days=7)
            
            # Format dates for CoinGecko API (DD-MM-YYYY)
            from_date = start_date.strftime('%d-%m-%Y')
            to_date = end_date.strftime('%d-%m-%Y')
            
            logger.info(f"Fetching price data for {asset_id} from {from_date} to {to_date}")
            
            # Fetch historical prices from CoinGecko
            price_data = await coingecko_client.get_historical_price_range(
                asset_id, from_date, to_date
            )
            
            if not price_data:
                logger.warning(f"No price data returned for {asset_id}")
                return []
            
            # Convert to PriceData objects
            daily_prices = []
            
            for i, price_point in enumerate(price_data):
                try:
                    # CoinGecko returns [timestamp, price] pairs
                    if len(price_point) >= 2:
                        timestamp = price_point[0] / 1000  # Convert from milliseconds
                        price = float(price_point[1])
                        
                        # Use actual timestamp from API instead of fabricated dates
                        actual_date = datetime.fromtimestamp(timestamp)
                        
                        # Use the price as close, but don't fabricate OHLC spread
                        # Set all OHLC values to the same price to indicate we only have close data
                        price_data_point = PriceData(
                            date=actual_date.strftime('%Y-%m-%d'),
                            price_open=price,  # Same as close - no fabricated spread
                            price_high=price,  # Same as close - no fabricated spread
                            price_low=price,   # Same as close - no fabricated spread
                            price_close=price,
                            volume=0,  # Would need separate API call - mark as unavailable
                            market_cap=0,  # Would need separate API call - mark as unavailable
                            price_change_24h=0,  # Calculate from previous day
                            price_change_percent_24h=0  # Calculate from previous day
                        )
                        
                        # Calculate daily changes
                        if i > 0:
                            prev_price = daily_prices[i-1].price_close
                            price_data_point.price_change_24h = price - prev_price
                            price_data_point.price_change_percent_24h = (
                                (price - prev_price) / prev_price * 100 if prev_price > 0 else 0
                            )
                        
                        daily_prices.append(price_data_point)
                        
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Error processing price point {i}: {e}")
                    continue
            
            # Cache the results with lock protection
            async with self.cache_lock:
                self.cache[cache_key] = daily_prices
            
            logger.info(f"Fetched {len(daily_prices)} daily price points for {asset_id}")
            return daily_prices
            
        except Exception as e:
            logger.error(f"Error fetching weekly price data for {asset_id}: {e}")
            return []
    
    def _calculate_price_metrics(self, article_data: Dict, article_date: datetime, 
                                asset_symbol: str, daily_prices: List[PriceData], 
                                article_sentiment: Dict) -> WeeklyPriceAnalysis:
        """Calculate comprehensive price metrics for the week following the article."""
        
        if not daily_prices:
            return self._create_empty_analysis(article_data, article_date, asset_symbol, article_sentiment)
        
        # Basic price metrics with safety guards
        prices = [p.price_close for p in daily_prices]
        
        # Safety guard: ensure prices list is not empty
        if not prices:
            logger.warning("Prices list is empty, cannot compute price metrics")
            return self._create_empty_analysis(article_data, article_date, asset_symbol, article_sentiment)
        
        week_start_price = prices[0]
        week_end_price = prices[-1]
        week_high = max(p.price_high for p in daily_prices)
        week_low = min(p.price_low for p in daily_prices)
        
        # Calculate weekly return
        weekly_return = (week_end_price - week_start_price) / week_start_price if week_start_price > 0 else 0
        
        # Calculate volatility (standard deviation of daily returns)
        daily_returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
            daily_returns.append(daily_return)
        
        weekly_volatility = pd.Series(daily_returns).std() if daily_returns else 0
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = week_start_price
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Identify major moves (>5% daily changes)
        major_moves = []
        for price_point in daily_prices:
            if abs(price_point.price_change_percent_24h) > 5:
                major_moves.append({
                    'date': price_point.date,
                    'change_percent': price_point.price_change_percent_24h,
                    'direction': 'up' if price_point.price_change_percent_24h > 0 else 'down',
                    'price': price_point.price_close
                })
        
        # Determine trend direction
        if weekly_return > 0.05:  # >5% gain
            trend_direction = 'up'
        elif weekly_return < -0.05:  # >5% loss
            trend_direction = 'down'
        else:
            trend_direction = 'sideways'
        
        # Determine volatility level
        if weekly_volatility > 0.05:  # >5% daily volatility
            volatility_level = 'high'
        elif weekly_volatility > 0.02:  # >2% daily volatility
            volatility_level = 'medium'
        else:
            volatility_level = 'low'
        
        # Calculate prediction accuracy
        prediction_accuracy = self._calculate_prediction_accuracy(
            article_sentiment, weekly_return, trend_direction
        )
        
        return WeeklyPriceAnalysis(
            article_url=article_data.get('url', ''),
            article_date=article_date,
            article_title=article_data.get('title', ''),
            asset=asset_symbol,
            daily_prices=daily_prices,
            week_start_price=week_start_price,
            week_end_price=week_end_price,
            week_high=week_high,
            week_low=week_low,
            weekly_return=weekly_return,
            weekly_volatility=weekly_volatility,
            max_drawdown=max_drawdown,
            article_sentiment=article_sentiment.get('sentiment', 'neutral'),
            article_confidence=article_sentiment.get('confidence', 0),
            prediction_accuracy=prediction_accuracy,
            major_moves=major_moves,
            trend_direction=trend_direction,
            volatility_level=volatility_level
        )

    def _create_empty_analysis(self, article_data: Dict, article_date: datetime, asset_symbol: str, article_sentiment: Dict) -> WeeklyPriceAnalysis:
        """Create a default empty WeeklyPriceAnalysis object when price data is unavailable."""
        return WeeklyPriceAnalysis(
            article_url=article_data.get('url', ''),
            article_date=article_date,
            article_title=article_data.get('title', ''),
            asset=asset_symbol,
            daily_prices=[],
            week_start_price=0,
            week_end_price=0,
            week_high=0,
            week_low=0,
            weekly_return=0,
            weekly_volatility=0,
            max_drawdown=0,
            article_sentiment=article_sentiment.get('sentiment', 'neutral'),
            article_confidence=article_sentiment.get('confidence', 0),
            prediction_accuracy=0,
            major_moves=[],
            trend_direction='sideways',
            volatility_level='low'
        )
    
    def _calculate_prediction_accuracy(self, article_sentiment: Dict, 
                                     weekly_return: float, trend_direction: str) -> float:
        """Calculate how accurately the article predicted price movement."""
        
        sentiment = article_sentiment.get('sentiment', 'neutral')
        confidence = article_sentiment.get('confidence', 0)
        
        # Base accuracy on sentiment vs actual price movement
        accuracy = 0.5  # Neutral baseline
        
        if sentiment == 'bullish' and weekly_return > 0:
            # Correctly predicted upward movement
            accuracy = 0.7 + (weekly_return * 2)  # Higher returns = higher accuracy
            accuracy = min(accuracy, 1.0)
        elif sentiment == 'bearish' and weekly_return < 0:
            # Correctly predicted downward movement  
            accuracy = 0.7 + (abs(weekly_return) * 2)  # Larger drops = higher accuracy
            accuracy = min(accuracy, 1.0)
        elif sentiment == 'neutral' and abs(weekly_return) < 0.05:
            # Correctly predicted sideways movement
            accuracy = 0.8
        elif sentiment == 'bullish' and weekly_return < -0.05:
            # Incorrectly predicted bullish but price dropped significantly
            accuracy = 0.2
        elif sentiment == 'bearish' and weekly_return > 0.05:
            # Incorrectly predicted bearish but price rose significantly
            accuracy = 0.2
        else:
            # Partial accuracy for minor mismatches
            accuracy = 0.4
        
        # Weight by confidence
        final_accuracy = accuracy * confidence + 0.5 * (1 - confidence)
        
        return min(max(final_accuracy, 0.0), 1.0)  # Clamp to 0-1
    
    def _safe_avg(self, values: List[float]) -> float:
        """Helper to safely calculate average, returns 0 if empty list."""
        return sum(values) / len(values) if values else 0.0
    
    def calculate_correlation_statistics(self, analyses: List[WeeklyPriceAnalysis]) -> Dict:
        """Calculate overall correlation statistics across all articles."""
        
        if not analyses:
            return {}
        
        # Overall prediction accuracy
        accuracies = [a.prediction_accuracy for a in analyses]
        avg_accuracy = self._safe_avg(accuracies)
        
        # Accuracy by sentiment
        bullish_analyses = [a for a in analyses if a.article_sentiment == 'bullish']
        bearish_analyses = [a for a in analyses if a.article_sentiment == 'bearish']
        neutral_analyses = [a for a in analyses if a.article_sentiment == 'neutral']
        
        bullish_accuracy = self._safe_avg([a.prediction_accuracy for a in bullish_analyses])
        bearish_accuracy = self._safe_avg([a.prediction_accuracy for a in bearish_analyses])
        neutral_accuracy = self._safe_avg([a.prediction_accuracy for a in neutral_analyses])
        
        # Average weekly returns with safe division
        avg_weekly_return = self._safe_avg([a.weekly_return for a in analyses])
        avg_volatility = self._safe_avg([a.weekly_volatility for a in analyses])
        
        # Hit rates with safe division
        correct_predictions = len([a for a in analyses if a.prediction_accuracy > 0.6])
        hit_rate = correct_predictions / len(analyses) if analyses else 0.0
        
        return {
            'total_articles_analyzed': len(analyses),
            'overall_accuracy': avg_accuracy,
            'sentiment_accuracy': {
                'bullish': bullish_accuracy,
                'bearish': bearish_accuracy,
                'neutral': neutral_accuracy
            },
            'hit_rate': hit_rate,
            'avg_weekly_return': avg_weekly_return,
            'avg_weekly_volatility': avg_volatility,
            'sentiment_distribution': {
                'bullish': len(bullish_analyses),
                'bearish': len(bearish_analyses),
                'neutral': len(neutral_analyses)
            }
        }


# Global instance
price_correlator = PriceCorrelator()
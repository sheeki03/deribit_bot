import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import uuid

from app.core.config import settings
from app.core.logging import logger
from app.ml.ensemble_scorer import ensemble_scorer, FlowScoreComponents
from app.market_data.coingecko_client import coingecko_client


class FlowScorer:
    """
    Main interface for calculating FlowScores for Deribit option flows articles.
    
    Orchestrates the entire scoring pipeline:
    1. Content validation and preprocessing
    2. Market data enrichment
    3. Multimodal ensemble scoring
    4. Database persistence
    5. Alert generation
    """
    
    def __init__(self):
        self.processing_queue = []
        self.processed_articles = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'extreme': settings.extreme_threshold,  # ±0.5
            'significant': settings.alert_threshold,  # ±0.3
            'confidence_minimum': settings.min_confidence_threshold  # 0.7
        }
    
    async def score_article(self, 
                           article_data: Dict,
                           assets: List[str] = None,
                           enrich_market_data: bool = True) -> Dict[str, FlowScoreComponents]:
        """
        Score an article for sentiment across specified assets.
        
        Args:
            article_data: Complete article data with content, images, metadata
            assets: List of assets to score ['BTC', 'ETH'] (default: both)
            enrich_market_data: Whether to fetch current market data
            
        Returns:
            Dictionary mapping asset -> FlowScoreComponents
        """
        if assets is None:
            assets = ['BTC', 'ETH']
        
        article_url = article_data.get('url', 'unknown')
        article_title = article_data.get('title', 'Untitled')
        
        logger.info(f"Scoring article: {article_title}")
        
        try:
            # Map cleaned field if present
            if not article_data.get('body_markdown') and not article_data.get('body_html'):
                if article_data.get('body_text'):
                    article_data['body_markdown'] = article_data.get('body_text')

            # Validate article data
            if not self._validate_article_data(article_data):
                logger.warning(f"Article validation failed: {article_url}")
                return self._create_neutral_scores(assets, "Validation failed")
            
            # Enrich with market data if requested
            if enrich_market_data:
                article_data = await self._enrich_market_data(article_data)
            
            # Score for each asset in parallel
            scoring_tasks = [
                ensemble_scorer.calculate_flowscore(article_data, asset)
                for asset in assets
            ]
            
            scores = await asyncio.gather(*scoring_tasks, return_exceptions=True)
            
            # Process results
            asset_scores = {}
            for i, asset in enumerate(assets):
                if isinstance(scores[i], Exception):
                    logger.error(f"Scoring failed for {asset}: {scores[i]}")
                    asset_scores[asset] = ensemble_scorer.create_neutral_score(f"Error: {scores[i]}")
                else:
                    asset_scores[asset] = scores[i]
            
            # Store results
            self.processed_articles[article_url] = {
                'timestamp': datetime.now(),
                'scores': asset_scores,
                'article_data': {
                    'title': article_title,
                    'url': article_url,
                    'published_at': article_data.get('published_at_utc')
                }
            }
            
            logger.info(f"Article scored successfully: {article_url}")
            
            # Generate alerts if scores are significant
            await self._check_alert_conditions(article_data, asset_scores)
            
            return asset_scores
            
        except Exception as e:
            logger.error(f"Article scoring failed: {article_url}", error=str(e))
            return self._create_neutral_scores(assets, f"Scoring error: {str(e)}")
    
    async def score_articles_batch(self, 
                                  articles_data: List[Dict],
                                  assets: List[str] = None,
                                  max_concurrent: int = 5,
                                  enrich_market_data: bool = True) -> Dict[str, Dict[str, FlowScoreComponents]]:
        """
        Score multiple articles efficiently with concurrency control.
        
        Args:
            articles_data: List of article data dictionaries
            assets: List of assets to score
            max_concurrent: Maximum concurrent scoring operations
            
        Returns:
            Dictionary mapping article_url -> {asset -> FlowScoreComponents}
        """
        if assets is None:
            assets = ['BTC', 'ETH']
        
        logger.info(f"Batch scoring {len(articles_data)} articles")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def score_with_semaphore(article_data):
            async with semaphore:
                return await self.score_article(article_data, assets, enrich_market_data=enrich_market_data)
        
        # Execute batch scoring
        scoring_tasks = [score_with_semaphore(article) for article in articles_data]
        results = await asyncio.gather(*scoring_tasks, return_exceptions=True)
        
        # Process results
        batch_results = {}
        for i, article_data in enumerate(articles_data):
            article_url = article_data.get('url', f'unknown_{i}')
            
            if isinstance(results[i], Exception):
                logger.error(f"Batch scoring failed for {article_url}: {results[i]}")
                batch_results[article_url] = self._create_neutral_scores(assets, f"Batch error: {results[i]}")
            else:
                batch_results[article_url] = results[i]
        
        successful_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Batch scoring completed: {successful_count}/{len(articles_data)} successful")
        
        return batch_results
    
    def _validate_article_data(self, article_data: Dict) -> bool:
        """Validate article data has minimum required fields."""
        required_fields = ['url', 'title']
        content_fields = ['body_markdown', 'body_html']
        
        # Check required fields
        for field in required_fields:
            if not article_data.get(field):
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Check content fields (at least one must exist)
        if not any(article_data.get(field) for field in content_fields):
            logger.warning("No content found in article")
            return False
        
        # Minimum content length
        content = article_data.get('body_markdown', '') or article_data.get('body_html', '')
        if len(content.strip()) < 100:
            logger.warning(f"Content too short: {len(content)} characters")
            return False
        
        return True
    
    async def _enrich_market_data(self, article_data: Dict) -> Dict:
        """Enrich article data with current market prices and context."""
        try:
            published_at = article_data.get('published_at_utc')
            if not published_at:
                return article_data
            
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            
            # Only enrich for dates within 2022-01-01 .. now
            from datetime import timezone, date
            lower = date(2022, 1, 1)
            now = datetime.now(timezone.utc)
            if published_at.date() < lower or published_at > now:
                return article_data

            # Get current prices for context
            current_prices = await coingecko_client.get_current_prices(['BTC', 'ETH'])
            
            # Get historical prices at publication time
            market_data = {}
            for asset in ['BTC', 'ETH']:
                try:
                    # Get price at publication time
                    base_price = await coingecko_client.get_price_at_timestamp(asset, published_at)
                    
                    if base_price:
                        # Calculate forward returns if enough time has passed
                        forward_returns = await coingecko_client.get_forward_returns(
                            asset, published_at, [4, 24, 72, 168]
                        )
                        
                        market_data[asset] = {
                            'base_price': base_price,
                            'forward_returns': forward_returns,
                            'current_price': current_prices.get(asset, {}).get('price')
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get market data for {asset}: {e}")
                    continue
            
            if market_data:
                article_data['market_data'] = market_data
                logger.debug(f"Enriched article with market data for {len(market_data)} assets")
            
            return article_data
            
        except Exception as e:
            logger.error(f"Market data enrichment failed: {e}")
            return article_data
    
    async def _check_alert_conditions(self, 
                                     article_data: Dict,
                                     asset_scores: Dict[str, FlowScoreComponents]):
        """Check if any scores meet alert conditions."""
        try:
            alerts_to_send = []
            
            for asset, score_components in asset_scores.items():
                score = score_components.final_score
                confidence = score_components.overall_confidence
                
                # Check if score meets alert criteria
                if confidence >= self.alert_thresholds['confidence_minimum']:
                    if abs(score) >= self.alert_thresholds['extreme']:
                        alert_level = 'extreme'
                    elif abs(score) >= self.alert_thresholds['significant']:
                        alert_level = 'significant'
                    else:
                        continue  # Below threshold
                    
                    alert_data = {
                        'level': alert_level,
                        'asset': asset,
                        'score': score,
                        'confidence': confidence,
                        'article_url': article_data.get('url'),
                        'article_title': article_data.get('title'),
                        'published_at': article_data.get('published_at_utc'),
                        'signals': score_components.signals,
                        'component_breakdown': {
                            'xgboost': score_components.xgboost_score,
                            'finbert': score_components.finbert_score,
                            'vision': score_components.vision_score,
                            'market_context': score_components.market_context_score
                        }
                    }
                    
                    alerts_to_send.append(alert_data)
            
            # Send alerts (implement notification logic)
            if alerts_to_send:
                await self._send_alerts(alerts_to_send)
                
        except Exception as e:
            logger.error(f"Alert checking failed: {e}")
    
    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts via configured channels (placeholder for now)."""
        for alert in alerts:
            logger.info(
                f"ALERT [{alert['level'].upper()}] {alert['asset']}: "
                f"Score {alert['score']:.3f} (confidence {alert['confidence']:.3f})"
            )
            # TODO: Implement Telegram alerts
    
    def _create_neutral_scores(self, assets: List[str], reason: str) -> Dict[str, FlowScoreComponents]:
        """Create neutral scores for all assets."""
        return {
            asset: ensemble_scorer.create_neutral_score(reason)
            for asset in assets
        }
    
    def get_recent_scores(self, limit: int = 20) -> List[Dict]:
        """Get recent article scores for monitoring."""
        recent_articles = sorted(
            self.processed_articles.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:limit]
        
        scores_summary = []
        for url, data in recent_articles:
            article_info = data['article_data']
            scores = data['scores']
            
            score_summary = {
                'url': url,
                'title': article_info['title'],
                'published_at': article_info['published_at'],
                'processed_at': data['timestamp'],
                'scores': {
                    asset: {
                        'score': components.final_score,
                        'confidence': components.overall_confidence
                    }
                    for asset, components in scores.items()
                }
            }
            scores_summary.append(score_summary)
        
        return scores_summary
    
    def get_scoring_statistics(self) -> Dict:
        """Get comprehensive scoring statistics."""
        if not self.processed_articles:
            return {'message': 'No articles scored yet'}
        
        # Collect all scores
        all_scores = {'BTC': [], 'ETH': []}
        all_confidences = {'BTC': [], 'ETH': []}
        
        for article_data in self.processed_articles.values():
            scores = article_data['scores']
            for asset, components in scores.items():
                if asset in all_scores:
                    all_scores[asset].append(components.final_score)
                    all_confidences[asset].append(components.overall_confidence)
        
        # Calculate statistics
        stats = {
            'total_articles': len(self.processed_articles),
            'assets': {}
        }
        
        for asset in ['BTC', 'ETH']:
            if all_scores[asset]:
                import numpy as np
                scores = all_scores[asset]
                confidences = all_confidences[asset]
                
                stats['assets'][asset] = {
                    'count': len(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'mean_confidence': np.mean(confidences),
                    'bullish_ratio': sum(1 for s in scores if s > 0.1) / len(scores) if scores else 0,
                    'bearish_ratio': sum(1 for s in scores if s < -0.1) / len(scores) if scores else 0,
                    'neutral_ratio': sum(1 for s in scores if -0.1 <= s <= 0.1) / len(scores) if scores else 0
                }
        
        # Add ensemble statistics
        stats['ensemble'] = ensemble_scorer.get_scoring_stats()
        
        return stats
    
    def clear_processed_articles(self, keep_recent: int = 100):
        """Clear old processed articles to manage memory."""
        if len(self.processed_articles) <= keep_recent:
            return
        
        # Keep most recent articles
        recent_articles = sorted(
            self.processed_articles.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:keep_recent]
        
        self.processed_articles = dict(recent_articles)
        logger.info(f"Cleared old articles, kept {keep_recent} most recent")


# Global flow scorer instance
flow_scorer = FlowScorer()
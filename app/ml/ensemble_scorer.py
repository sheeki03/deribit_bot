import numpy as np
import asyncio
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
import json

from app.core.config import settings
from app.core.logging import logger
from app.ml.xgboost_model import xgboost_model
from app.ml.finbert_model import finbert_model
from app.vision.vision_ai import vision_ai
from app.market_data.coingecko_client import coingecko_client


@dataclass
class FlowScoreComponents:
    """Components of a FlowScore calculation."""
    xgboost_score: float
    xgboost_confidence: float
    finbert_score: float
    finbert_confidence: float
    vision_score: float
    vision_confidence: float
    market_context_score: float
    final_score: float
    overall_confidence: float
    signals: List[str]
    component_weights: Dict[str, float]


class EnsembleFlowScorer:
    """
    Advanced ensemble scorer combining multiple AI/ML models for ultimate accuracy.
    
    Architecture:
    - XGBoost (60% weight): Fast, traditional ML with engineered features
    - FinBERT (25% weight): Deep financial language understanding  
    - Vision AI (10% weight): Chart and image analysis
    - Market Context (5% weight): Price momentum and timing
    
    Features:
    - Adaptive weighting based on confidence scores
    - Market regime awareness
    - Uncertainty quantification
    - Performance tracking and optimization
    """
    
    def __init__(self):
        # Default model weights (can be optimized based on performance)
        self.model_weights = {
            'xgboost': 0.60,      # Fast, reliable, good for real-time
            'finbert': 0.25,      # Deep language understanding
            'vision': 0.10,       # Chart/image analysis
            'market_context': 0.05 # Market timing factors
        }
        
        # Confidence thresholds for adaptive weighting
        self.confidence_thresholds = {
            'high': 0.80,
            'medium': 0.60,
            'low': 0.40
        }
        
        # Market timing configuration
        self.market_config = {
            'market_hours_start': 9,
            'market_hours_end': 16,
            'timezone': 'UTC',  # All times in UTC
            'market_hours_boost': 0.1,
            'weekday_boost': 0.05,
            'recent_boost_4h': 0.1,
            'recent_boost_24h': 0.05
        }
        
        # Model performance tracking (bounded to prevent memory issues)
        self.performance_history = []
        self.max_performance_history = 1000
        
        # Market regime detection (removed unused feature)
        # TODO: Implement market regime awareness in future versions
        
        # Scoring cache for performance
        self._score_cache = {}
        
    async def calculate_flowscore(self, 
                                 article_data: Dict,
                                 asset: str = 'BTC',
                                 use_cache: bool = True) -> FlowScoreComponents:
        """
        Calculate comprehensive FlowScore for an article.
        
        Args:
            article_data: Complete article data with content, images, etc.
            asset: Target asset ('BTC' or 'ETH')
            use_cache: Whether to use cached results
            
        Returns:
            FlowScoreComponents with detailed breakdown
        """
        try:
            # Cache key for performance
            cache_key = f"{article_data.get('url', 'unknown')}_{asset}"
            
            if use_cache and cache_key in self._score_cache:
                logger.debug(f"Using cached FlowScore for {cache_key}")
                return self._score_cache[cache_key]
            
            logger.info(f"Calculating FlowScore for {asset}: {article_data.get('title', 'Unknown')}")
            
            # Extract text content
            text_content = article_data.get('body_markdown', '') or article_data.get('body_html', '')
            title = article_data.get('title', '')
            
            if not text_content:
                logger.warning("No text content found for FlowScore calculation")
                return self._create_neutral_score("No content")
            
            # Run all models in parallel for speed
            xgboost_task = self._get_xgboost_score(article_data, asset)
            finbert_task = self._get_finbert_score(text_content, title, asset)
            vision_task = self._get_vision_score(article_data.get('images', []), asset)
            market_task = self._get_market_context_score(article_data, asset)
            
            # Execute all tasks concurrently
            xgboost_result, finbert_result, vision_result, market_result = await asyncio.gather(
                xgboost_task, finbert_task, vision_task, market_task,
                return_exceptions=True
            )
            
            # Handle any exceptions
            if isinstance(xgboost_result, Exception):
                logger.error(f"XGBoost scoring failed: {xgboost_result}")
                xgboost_result = {'score': 0.0, 'confidence': 0.0}
            
            if isinstance(finbert_result, Exception):
                logger.error(f"FinBERT scoring failed: {finbert_result}")
                finbert_result = {'score': 0.0, 'confidence': 0.0}
            
            if isinstance(vision_result, Exception):
                logger.error(f"Vision scoring failed: {vision_result}")
                vision_result = {'score': 0.0, 'confidence': 0.0}
            
            if isinstance(market_result, Exception):
                logger.error(f"Market context scoring failed: {market_result}")
                market_result = {'score': 0.0, 'confidence': 0.0}
            
            # Calculate adaptive weights based on confidence
            adaptive_weights = self._calculate_adaptive_weights({
                'xgboost': xgboost_result['confidence'],
                'finbert': finbert_result['confidence'],
                'vision': vision_result['confidence'],
                'market_context': market_result['confidence']
            })
            
            # Calculate weighted final score
            final_score = (
                adaptive_weights['xgboost'] * xgboost_result['score'] +
                adaptive_weights['finbert'] * finbert_result['score'] +
                adaptive_weights['vision'] * vision_result['score'] +
                adaptive_weights['market_context'] * market_result['score']
            )
            
            # Calculate overall confidence
            overall_confidence = (
                adaptive_weights['xgboost'] * xgboost_result['confidence'] +
                adaptive_weights['finbert'] * finbert_result['confidence'] +
                adaptive_weights['vision'] * vision_result['confidence'] +
                adaptive_weights['market_context'] * market_result['confidence']
            )
            
            # Collect signals from all models
            signals = []
            signals.extend(xgboost_result.get('signals', []))
            signals.extend(finbert_result.get('signals', []))
            signals.extend(vision_result.get('signals', []))
            signals.extend(market_result.get('signals', []))
            
            # Create FlowScore components
            flow_score = FlowScoreComponents(
                xgboost_score=xgboost_result['score'],
                xgboost_confidence=xgboost_result['confidence'],
                finbert_score=finbert_result['score'],
                finbert_confidence=finbert_result['confidence'],
                vision_score=vision_result['score'],
                vision_confidence=vision_result['confidence'],
                market_context_score=market_result['score'],
                final_score=np.clip(final_score, -1.0, 1.0),
                overall_confidence=np.clip(overall_confidence, 0.0, 1.0),
                signals=list(set(signals)),  # Remove duplicates
                component_weights=adaptive_weights
            )
            
            # Cache result
            if use_cache:
                self._score_cache[cache_key] = flow_score
            
            # Track performance (with size limit)
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'asset': asset,
                'final_score': flow_score.final_score,
                'confidence': flow_score.overall_confidence,
                'components': {
                    'xgboost': xgboost_result['score'],
                    'finbert': finbert_result['score'],
                    'vision': vision_result['score'],
                    'market_context': market_result['score']
                },
                'weights': adaptive_weights
            })
            
            # Maintain maximum size to prevent memory issues
            if len(self.performance_history) > self.max_performance_history:
                self.performance_history = self.performance_history[-self.max_performance_history:]
            
            logger.info(
                f"FlowScore calculated: {flow_score.final_score:.3f} "
                f"(confidence: {flow_score.overall_confidence:.3f})"
            )
            
            return flow_score
            
        except Exception as e:
            logger.error(f"FlowScore calculation failed: {str(e)}")
            return self._create_neutral_score(f"Error: {str(e)}")
    
    async def _get_xgboost_score(self, article_data: Dict, asset: str) -> Dict[str, Union[float, List[str]]]:
        """Get XGBoost model prediction."""
        try:
            if not xgboost_model.is_trained:
                logger.warning("XGBoost model not trained, returning neutral score")
                return {'score': 0.0, 'confidence': 0.0, 'signals': ['Model not trained']}
            
            # Get prediction with confidence for single article
            predictions, confidences = xgboost_model.predict(
                [article_data], return_probabilities=True
            )
            
            prediction = predictions[0] if len(predictions) > 0 else 0.0
            confidence = confidences[0] if len(confidences) > 0 else 0.0
            
            # Get feature importance for signals
            top_features = xgboost_model.get_feature_importance(top_n=5)
            signals = [f"Feature: {feature}" for feature in top_features.keys()]
            
            return {
                'score': float(prediction),
                'confidence': float(confidence),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"XGBoost scoring error: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': [f'XGBoost error: {str(e)}']}
    
    async def _get_finbert_score(self, text: str, title: str, asset: str) -> Dict[str, Union[float, List[str]]]:
        """Get FinBERT sentiment prediction."""
        try:
            # Load model if not already loaded
            if not finbert_model.is_loaded:
                success = finbert_model.load_model()
                if not success:
                    return {'score': 0.0, 'confidence': 0.0, 'signals': ['FinBERT not available']}
            
            # Combine title and content, prioritize asset-specific content
            combined_text = f"{title}. {text}"
            
            # Filter for asset-specific content
            asset_text = self._filter_asset_content(combined_text, asset)
            
            # Get sentiment prediction
            result = finbert_model.predict_sentiment(asset_text)
            
            # Get options context analysis
            options_context = finbert_model.analyze_options_context(asset_text)
            
            # Generate signals
            signals = []
            if result['sentiment'] == 'positive':
                signals.append("FinBERT: Bullish sentiment")
            elif result['sentiment'] == 'negative':
                signals.append("FinBERT: Bearish sentiment")
            
            if options_context['options_relevance'] > 0.5:
                signals.append(f"High options relevance: {options_context['options_relevance']:.2f}")
            
            if options_context['strategy_mentions']:
                signals.extend([f"Strategy: {strategy}" for strategy in options_context['strategy_mentions']])
            
            return {
                'score': result['sentiment_score'],
                'confidence': result['confidence'],
                'signals': signals,
                'options_context': options_context
            }
            
        except Exception as e:
            logger.error(f"FinBERT scoring error: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': [f'FinBERT error: {str(e)}']}
    
    async def _get_vision_score(self, images_data: List[Dict], asset: str) -> Dict[str, Union[float, List[str]]]:
        """Get Vision AI prediction from images."""
        try:
            if not images_data:
                return {'score': 0.0, 'confidence': 0.0, 'signals': ['No images available']}
            
            vision_scores = []
            vision_confidences = []
            signals = []
            
            # Analyze each image
            for image_data in images_data[:3]:  # Limit to top 3 images
                image_type = image_data.get('image_type', 'unknown')
                vision_analysis = image_data.get('vision_analysis', {})
                
                if vision_analysis:
                    # Parse vision AI sentiment
                    combined_sentiment = vision_analysis.get('combined_sentiment', 'neutral')
                    confidence = vision_analysis.get('confidence', 0.0)
                    
                    # Map sentiment to score
                    sentiment_mapping = {'bullish': 1.0, 'bearish': -1.0, 'neutral': 0.0}
                    score = sentiment_mapping.get(combined_sentiment.lower(), 0.0)
                    
                    vision_scores.append(score)
                    vision_confidences.append(confidence)
                    
                    # Add signals
                    if combined_sentiment != 'neutral':
                        signals.append(f"Vision: {image_type} shows {combined_sentiment} sentiment")
                    
                    # Add key insights
                    key_insights = vision_analysis.get('key_insights', [])
                    signals.extend([f"Chart insight: {insight}" for insight in key_insights[:2]])
            
            if vision_scores:
                # Weighted average by confidence
                weights = np.array(vision_confidences) + 1e-8  # Avoid division by zero
                weighted_score = np.average(vision_scores, weights=weights)
                avg_confidence = np.mean(vision_confidences)
                
                return {
                    'score': float(weighted_score),
                    'confidence': float(avg_confidence),
                    'signals': signals
                }
            else:
                return {'score': 0.0, 'confidence': 0.0, 'signals': ['No vision analysis available']}
                
        except Exception as e:
            logger.error(f"Vision scoring error: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': [f'Vision error: {str(e)}']}
    
    async def _get_market_context_score(self, article_data: Dict, asset: str) -> Dict[str, Union[float, List[str]]]:
        """Get market context and timing score."""
        try:
            published_at = article_data.get('published_at_utc')
            if not published_at:
                return {'score': 0.0, 'confidence': 0.0, 'signals': ['No publication timestamp']}
            
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            
            # Get current price data for context
            current_prices = await coingecko_client.get_current_prices([asset])
            
            signals = []
            market_score = 0.0
            confidence = 0.5  # Medium confidence for market context
            
            if asset in current_prices:
                asset_data = current_prices[asset]
                
                # Time-based factors (timezone-aware)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                
                current_time = datetime.now(timezone.utc)
                
                hour = published_at.hour
                is_market_hours = (self.market_config['market_hours_start'] <= hour <= 
                                 self.market_config['market_hours_end'])
                is_weekend = published_at.weekday() >= 5
                
                # Market timing score
                if is_market_hours:
                    market_score += self.market_config['market_hours_boost']
                    signals.append("Published during market hours")
                
                if not is_weekend:
                    market_score += self.market_config['weekday_boost']
                    signals.append("Published on weekday")
                
                # Recency factor (newer = higher impact)
                hours_since = (current_time - published_at).total_seconds() / 3600
                if hours_since < 4:
                    market_score += self.market_config['recent_boost_4h']
                    signals.append("Very recent publication")
                elif hours_since < 24:
                    market_score += self.market_config['recent_boost_24h']
                    signals.append("Recent publication")
                
                confidence = 0.7
            
            return {
                'score': float(market_score),
                'confidence': float(confidence),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Market context scoring error: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'signals': [f'Market context error: {str(e)}']}
    
    def _filter_asset_content(self, text: str, asset: str) -> str:
        """Filter text content for asset-specific mentions."""
        if not text:
            return text
        
        # Split into sentences
        sentences = text.split('.')
        
        # Asset keywords
        asset_keywords = {
            'BTC': ['btc', 'bitcoin'],
            'ETH': ['eth', 'ethereum']
        }
        
        keywords = asset_keywords.get(asset, [])
        if not keywords:
            return text  # Return full text if asset not recognized
        
        # Find sentences mentioning the asset
        asset_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                asset_sentences.append(sentence.strip())
        
        # If no specific mentions, return first part of original text
        if not asset_sentences:
            return text[:1000]  # First 1000 characters
        
        # Return asset-specific content
        return '. '.join(asset_sentences)
    
    def _calculate_adaptive_weights(self, confidences: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate adaptive weights based on model confidence scores.
        
        High-confidence models get higher weight, low-confidence models get lower weight.
        """
        adaptive_weights = {}
        total_adjusted_weight = 0
        
        for model, base_weight in self.model_weights.items():
            confidence = confidences.get(model, 0.0)
            
            # Confidence multiplier (0.5x to 1.5x based on confidence)
            if confidence >= self.confidence_thresholds['high']:
                multiplier = 1.3
            elif confidence >= self.confidence_thresholds['medium']:
                multiplier = 1.0
            elif confidence >= self.confidence_thresholds['low']:
                multiplier = 0.7
            else:
                multiplier = 0.5
            
            adjusted_weight = base_weight * multiplier
            adaptive_weights[model] = adjusted_weight
            total_adjusted_weight += adjusted_weight
        
        # Normalize weights to sum to 1.0
        if total_adjusted_weight > 0:
            for model in adaptive_weights:
                adaptive_weights[model] /= total_adjusted_weight
        else:
            # Fallback to equal weights
            adaptive_weights = {model: 0.25 for model in self.model_weights.keys()}
        
        return adaptive_weights
    
    def create_neutral_score(self, reason: str) -> FlowScoreComponents:
        """Create a neutral FlowScore for error cases (public method)."""
        return self._create_neutral_score(reason)
    
    def _create_neutral_score(self, reason: str) -> FlowScoreComponents:
        """Create a neutral FlowScore for error cases."""
        return FlowScoreComponents(
            xgboost_score=0.0,
            xgboost_confidence=0.0,
            finbert_score=0.0,
            finbert_confidence=0.0,
            vision_score=0.0,
            vision_confidence=0.0,
            market_context_score=0.0,
            final_score=0.0,
            overall_confidence=0.0,
            signals=[reason],
            component_weights=self.model_weights.copy()
        )
    
    def get_scoring_stats(self) -> Dict:
        """Get ensemble scoring statistics."""
        if not self.performance_history:
            return {'total_scores': 0, 'message': 'No scores calculated yet'}
        
        recent_scores = self.performance_history[-100:]  # Last 100 scores
        
        scores = [entry['final_score'] for entry in recent_scores]
        confidences = [entry['confidence'] for entry in recent_scores]
        
        # Component performance
        components_stats = {}
        for component in ['xgboost', 'finbert', 'vision', 'market_context']:
            component_scores = [entry['components'][component] for entry in recent_scores]
            components_stats[component] = {
                'mean': np.mean(component_scores),
                'std': np.std(component_scores),
                'min': np.min(component_scores),
                'max': np.max(component_scores)
            }
        
        return {
            'total_scores': len(self.performance_history),
            'recent_stats': {
                'mean_score': np.mean(scores),
                'mean_confidence': np.mean(confidences),
                'score_volatility': np.std(scores),
                'bullish_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
                'bearish_ratio': sum(1 for s in scores if s < -0.1) / len(scores),
                'neutral_ratio': sum(1 for s in scores if -0.1 <= s <= 0.1) / len(scores)
            },
            'component_stats': components_stats,
            'model_weights': self.model_weights
        }
    
    def clear_cache(self):
        """Clear scoring cache."""
        self._score_cache.clear()
        logger.info("Ensemble scoring cache cleared")
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights based on performance feedback."""
        # Validate weights
        if not np.isclose(sum(new_weights.values()), 1.0, rtol=1e-5):
            raise ValueError("Model weights must sum to 1.0")
        
        if set(new_weights.keys()) != set(self.model_weights.keys()):
            raise ValueError("New weights must include all model components")
        
        old_weights = self.model_weights.copy()
        self.model_weights = new_weights.copy()
        
        logger.info(f"Model weights updated: {old_weights} -> {new_weights}")


# Global ensemble scorer instance
ensemble_scorer = EnsembleFlowScorer()
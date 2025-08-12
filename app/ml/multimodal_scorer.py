#!/usr/bin/env python3
"""
Multimodal Fusion Scoring System

Combines text analysis, image analysis, and market data to generate comprehensive
FlowScores according to Enhanced PRD specifications:
- 30% Text Analysis
- 40% Image Analysis  
- 20% Market Context
- 10% Meta Signals

This implements Enhanced PRD Phase 2: Intelligence - Multimodal Fusion component.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from app.core.logging import logger
from app.ml.feature_extractors import OptionsFeatureExtractor
from app.vision.image_analyzer import image_analyzer
from app.market_data.coingecko_client import coingecko_client


@dataclass
class MultimodalScoreComponents:
    """Comprehensive breakdown of multimodal scoring components."""
    
    # Component scores (0.0 to 1.0, then normalized to -1.0 to +1.0)
    text_score: float
    image_score: float
    market_context_score: float
    meta_signals_score: float
    
    # Final weighted score
    final_flowscore: float
    
    # Confidence metrics
    text_confidence: float
    image_confidence: float
    market_confidence: float
    overall_confidence: float
    
    # Supporting data
    text_signals: Dict[str, Any]
    image_signals: Dict[str, Any]
    market_signals: Dict[str, Any]
    meta_signals: Dict[str, Any]
    
    # Processing metadata
    processing_timestamp: str
    asset: str
    article_url: str


class MultimodalScorer:
    """
    Advanced multimodal scoring system that combines multiple data sources
    to generate comprehensive FlowScore predictions.
    """
    
    # Enhanced PRD weights
    COMPONENT_WEIGHTS = {
        'text': 0.30,      # 30% - Enhanced text analysis with options terminology
        'image': 0.40,     # 40% - Image analysis (most valuable for options)
        'market': 0.20,    # 20% - Market context and timing
        'meta': 0.10       # 10% - Meta signals (author, publication timing, etc.)
    }
    
    def __init__(self):
        self.text_extractor = OptionsFeatureExtractor()
        self.image_analyzer = image_analyzer
        self.processing_stats = {
            'total_articles': 0,
            'successful_scores': 0,
            'component_failures': {'text': 0, 'image': 0, 'market': 0, 'meta': 0}
        }
    
    async def calculate_multimodal_score(self, 
                                       article_data: Dict,
                                       asset: str = 'BTC',
                                       images_data: Optional[List[Dict]] = None) -> MultimodalScoreComponents:
        """
        Calculate comprehensive multimodal FlowScore for an article.
        
        Args:
            article_data: Complete article data with text content
            asset: Target asset ('BTC' or 'ETH')
            images_data: Optional pre-analyzed image data
            
        Returns:
            MultimodalScoreComponents with complete breakdown
        """
        article_url = article_data.get('url', 'unknown')
        logger.info(f"Calculating multimodal score for {asset}: {article_url}")
        
        try:
            # Extract text content
            text_content = (
                article_data.get('body_text') or 
                article_data.get('body_markdown') or 
                article_data.get('body_html', '')
            )
            title = article_data.get('title', '')
            
            if not text_content:
                logger.warning(f"No text content found for article: {article_url}")
                return self._create_neutral_score(asset, article_url, "No text content")
            
            # Process all components in parallel
            tasks = [
                self._analyze_text_component(text_content, title, asset),
                self._analyze_image_component(article_data, images_data),
                self._analyze_market_component(article_data, asset),
                self._analyze_meta_component(article_data, asset)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            text_result, image_result, market_result, meta_result = results
            
            # Handle exceptions
            if isinstance(text_result, Exception):
                logger.error(f"Text analysis failed: {text_result}")
                text_result = self._create_neutral_component("text", str(text_result))
                self.processing_stats['component_failures']['text'] += 1
                
            if isinstance(image_result, Exception):
                logger.error(f"Image analysis failed: {image_result}")
                image_result = self._create_neutral_component("image", str(image_result))
                self.processing_stats['component_failures']['image'] += 1
                
            if isinstance(market_result, Exception):
                logger.error(f"Market analysis failed: {market_result}")
                market_result = self._create_neutral_component("market", str(market_result))
                self.processing_stats['component_failures']['market'] += 1
                
            if isinstance(meta_result, Exception):
                logger.error(f"Meta analysis failed: {meta_result}")
                meta_result = self._create_neutral_component("meta", str(meta_result))
                self.processing_stats['component_failures']['meta'] += 1
            
            # Combine scores with Enhanced PRD weights
            final_score = self._calculate_weighted_score(
                text_result, image_result, market_result, meta_result
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                text_result, image_result, market_result, meta_result
            )
            
            # Create comprehensive result
            multimodal_score = MultimodalScoreComponents(
                text_score=text_result['score'],
                image_score=image_result['score'],
                market_context_score=market_result['score'],
                meta_signals_score=meta_result['score'],
                final_flowscore=final_score,
                text_confidence=text_result['confidence'],
                image_confidence=image_result['confidence'],
                market_confidence=market_result['confidence'],
                overall_confidence=overall_confidence,
                text_signals=text_result['signals'],
                image_signals=image_result['signals'],
                market_signals=market_result['signals'],
                meta_signals=meta_result['signals'],
                processing_timestamp=datetime.now().isoformat(),
                asset=asset,
                article_url=article_url
            )
            
            self.processing_stats['successful_scores'] += 1
            
            logger.info(
                f"Multimodal score calculated",
                asset=asset,
                final_score=final_score,
                confidence=overall_confidence,
                text_score=text_result['score'],
                image_score=image_result['score']
            )
            
            return multimodal_score
            
        except Exception as e:
            logger.error(f"Multimodal scoring failed: {e}", article_url=article_url)
            return self._create_neutral_score(asset, article_url, f"Scoring error: {str(e)}")
        
        finally:
            self.processing_stats['total_articles'] += 1
    
    async def _analyze_text_component(self, text_content: str, title: str, asset: str) -> Dict:
        """Analyze text content using enhanced options terminology."""
        try:
            # Extract structured options data
            extraction = self.text_extractor.extract_structured_options_data(text_content, title)
            
            # Convert flow direction to numeric score
            flow_direction = extraction.get('flow_direction', 'neutral')
            if flow_direction == 'bullish':
                base_score = 0.6
            elif flow_direction == 'bearish':
                base_score = -0.6
            else:
                base_score = 0.0
            
            # Adjust based on strikes and notionals found
            strikes_count = len(extraction.get('strikes', []))
            notionals_count = len(extraction.get('notionals', []))
            
            # More data = higher confidence and stronger signal
            data_multiplier = min(1.0, (strikes_count + notionals_count) * 0.1 + 0.7)
            adjusted_score = base_score * data_multiplier
            
            # Asset-specific adjustment (if mentioned)
            text_lower = text_content.lower()
            title_lower = title.lower()
            asset_mentioned = asset.lower() in text_lower or asset.lower() in title_lower
            
            if not asset_mentioned:
                adjusted_score *= 0.8  # Reduce score if asset not directly mentioned
            
            # Clamp to [-1, 1]
            final_score = np.clip(adjusted_score, -1.0, 1.0)
            
            return {
                'score': final_score,
                'confidence': extraction.get('confidence', 0.5),
                'signals': {
                    'flow_direction': flow_direction,
                    'strikes_count': strikes_count,
                    'notionals_count': notionals_count,
                    'asset_mentioned': asset_mentioned,
                    'sentiment_indicators': extraction.get('sentiment_indicators', []),
                    'greeks_found': len(extraction.get('greeks', {})),
                    'expiries_found': len(extraction.get('expiries', [])),
                    'text_length': len(text_content)
                }
            }
            
        except Exception as e:
            logger.error(f"Text component analysis failed: {e}")
            raise
    
    async def _analyze_image_component(self, article_data: Dict, images_data: Optional[List[Dict]] = None) -> Dict:
        """Analyze images for options-specific visual content."""
        try:
            if images_data:
                # Use pre-analyzed image data
                image_signals = images_data
            else:
                # Need to analyze images from article
                image_signals = await self._extract_images_from_article(article_data)
            
            if not image_signals:
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'signals': {'message': 'No images found'}
                }
            
            # Analyze image types and content
            total_score = 0.0
            total_confidence = 0.0
            classified_images = 0
            
            type_weights = {
                'greeks_chart': 1.0,      # Highest value
                'flow_heatmap': 0.9,      # High value
                'skew_chart': 0.8,        # Medium-high value
                'price_chart': 0.6,       # Medium value
                'position_diagram': 0.7,  # Medium-high value
                'unknown': 0.2            # Low value
            }
            
            for img_data in image_signals:
                img_type = img_data.get('classification', 'unknown')
                img_confidence = img_data.get('confidence', 0.0)
                
                if img_confidence > 0.3:  # Only count confident classifications
                    weight = type_weights.get(img_type, 0.2)
                    
                    # Determine sentiment from image content
                    sentiment = img_data.get('sentiment', {})
                    bullish_signals = len(sentiment.get('bullish', []))
                    bearish_signals = len(sentiment.get('bearish', []))
                    
                    # Score based on sentiment indicators
                    if bullish_signals > bearish_signals:
                        img_score = weight * 0.7
                    elif bearish_signals > bullish_signals:
                        img_score = -weight * 0.7
                    else:
                        img_score = 0.0
                    
                    total_score += img_score
                    total_confidence += img_confidence
                    classified_images += 1
            
            if classified_images == 0:
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'signals': {'message': 'No confidently classified images'}
                }
            
            # Average and normalize
            avg_score = total_score / classified_images
            avg_confidence = total_confidence / classified_images
            
            # Clamp score to [-1, 1]
            final_score = np.clip(avg_score, -1.0, 1.0)
            
            return {
                'score': final_score,
                'confidence': avg_confidence,
                'signals': {
                    'total_images': len(image_signals),
                    'classified_images': classified_images,
                    'image_types': [img.get('classification') for img in image_signals],
                    'avg_image_confidence': avg_confidence,
                    'dominant_type': max(type_weights.items(), key=lambda x: x[1])[0] if classified_images > 0 else 'none'
                }
            }
            
        except Exception as e:
            logger.error(f"Image component analysis failed: {e}")
            raise
    
    async def _extract_images_from_article(self, article_data: Dict) -> List[Dict]:
        """Extract and analyze images from article data."""
        # This would typically pull from the images database or process images directly
        # For now, return empty list - this would be enhanced with actual image processing
        return []
    
    async def _analyze_market_component(self, article_data: Dict, asset: str) -> Dict:
        """Analyze market context at time of publication."""
        try:
            published_at = article_data.get('published_at_utc')
            if not published_at:
                return {
                    'score': 0.0,
                    'confidence': 0.0,
                    'signals': {'message': 'No publication timestamp'}
                }
            
            # Parse timestamp
            if isinstance(published_at, str):
                pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_time = published_at
            
            # Get market data around publication time
            market_signals = {}
            
            try:
                # Get price at publication
                base_price_data = await coingecko_client.get_price_at_timestamp(asset, pub_time)
                if base_price_data:
                    base_price = base_price_data.get('price')
                    market_signals['price_at_publication'] = base_price
                    
                    # Get forward returns for performance context
                    forward_returns = await coingecko_client.get_forward_returns(
                        asset, pub_time, [4, 24, 72, 168]  # 4h, 1d, 3d, 1w
                    )
                    market_signals['forward_returns'] = forward_returns
                    
                    # Score based on subsequent performance
                    # If we have forward returns, use them to validate prediction
                    if forward_returns:
                        # Use 24h return as primary signal
                        return_24h = forward_returns.get('ret_24h')
                        if return_24h is not None:
                            # Strong positive returns suggest bullish context
                            # Strong negative returns suggest bearish context
                            if return_24h > 0.05:  # >5% gain
                                market_score = 0.6
                            elif return_24h < -0.05:  # >5% loss  
                                market_score = -0.6
                            elif return_24h > 0.02:  # >2% gain
                                market_score = 0.3
                            elif return_24h < -0.02:  # >2% loss
                                market_score = -0.3
                            else:
                                market_score = 0.0
                            
                            confidence = 0.8
                        else:
                            market_score = 0.0
                            confidence = 0.2
                    else:
                        market_score = 0.0
                        confidence = 0.3
                else:
                    market_score = 0.0
                    confidence = 0.1
                    
            except Exception as e:
                logger.warning(f"Failed to get market data: {e}")
                market_score = 0.0
                confidence = 0.1
                market_signals['error'] = str(e)
            
            return {
                'score': market_score,
                'confidence': confidence,
                'signals': market_signals
            }
            
        except Exception as e:
            logger.error(f"Market component analysis failed: {e}")
            raise
    
    async def _analyze_meta_component(self, article_data: Dict, asset: str) -> Dict:
        """Analyze meta signals like timing, author, publication patterns."""
        try:
            meta_signals = {}
            meta_score = 0.0
            confidence = 0.5
            
            # Publication timing analysis
            published_at = article_data.get('published_at_utc')
            if published_at:
                if isinstance(published_at, str):
                    pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    pub_time = published_at
                
                # Weekend vs weekday effect
                weekday = pub_time.weekday()
                if weekday >= 5:  # Weekend
                    meta_signals['is_weekend'] = True
                    meta_score += 0.1  # Weekend articles might be more significant
                else:
                    meta_signals['is_weekend'] = False
                
                # Hour of day effect
                hour = pub_time.hour
                if 9 <= hour <= 16:  # Market hours (approximate)
                    meta_signals['market_hours'] = True
                    meta_score += 0.1
                else:
                    meta_signals['market_hours'] = False
                
                meta_signals['publication_hour'] = hour
                meta_signals['publication_weekday'] = weekday
            
            # Author analysis (if available)
            author = article_data.get('author')
            if author:
                meta_signals['has_author'] = True
                meta_signals['author'] = author
                confidence += 0.1
            else:
                meta_signals['has_author'] = False
            
            # Title analysis
            title = article_data.get('title', '')
            if title:
                title_lower = title.lower()
                
                # Look for urgency/importance indicators
                urgent_words = ['breaking', 'urgent', 'alert', 'massive', 'huge', 'unprecedented']
                urgent_count = sum(1 for word in urgent_words if word in title_lower)
                if urgent_count > 0:
                    meta_score += 0.2
                    meta_signals['urgency_indicators'] = urgent_count
                
                # Asset mention in title
                if asset.lower() in title_lower:
                    meta_score += 0.1
                    meta_signals['asset_in_title'] = True
                else:
                    meta_signals['asset_in_title'] = False
            
            # Content length (longer articles might be more analytical)
            text_content = (
                article_data.get('body_text') or 
                article_data.get('body_markdown') or 
                article_data.get('body_html', '')
            )
            
            if text_content:
                content_length = len(text_content)
                meta_signals['content_length'] = content_length
                
                # Longer content = higher confidence
                if content_length > 2000:
                    confidence += 0.1
                    meta_score += 0.05
                elif content_length < 500:
                    confidence -= 0.1
            
            # Clamp score and confidence
            final_score = np.clip(meta_score, -1.0, 1.0)
            final_confidence = np.clip(confidence, 0.0, 1.0)
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'signals': meta_signals
            }
            
        except Exception as e:
            logger.error(f"Meta component analysis failed: {e}")
            raise
    
    def _calculate_weighted_score(self, text_result: Dict, image_result: Dict, 
                                market_result: Dict, meta_result: Dict) -> float:
        """Calculate final weighted score using Enhanced PRD weights."""
        
        # Apply component weights
        weighted_score = (
            self.COMPONENT_WEIGHTS['text'] * text_result['score'] +
            self.COMPONENT_WEIGHTS['image'] * image_result['score'] +
            self.COMPONENT_WEIGHTS['market'] * market_result['score'] +
            self.COMPONENT_WEIGHTS['meta'] * meta_result['score']
        )
        
        # Clamp to [-1, 1]
        return np.clip(weighted_score, -1.0, 1.0)
    
    def _calculate_overall_confidence(self, text_result: Dict, image_result: Dict,
                                    market_result: Dict, meta_result: Dict) -> float:
        """Calculate overall confidence based on component confidences and weights."""
        
        # Weight confidences by component importance
        weighted_confidence = (
            self.COMPONENT_WEIGHTS['text'] * text_result['confidence'] +
            self.COMPONENT_WEIGHTS['image'] * image_result['confidence'] +
            self.COMPONENT_WEIGHTS['market'] * market_result['confidence'] +
            self.COMPONENT_WEIGHTS['meta'] * meta_result['confidence']
        )
        
        # Penalize if key components are missing
        penalties = 0
        if text_result['confidence'] < 0.3:
            penalties += 0.2
        if image_result['confidence'] < 0.3:
            penalties += 0.3  # Images are weighted highest
        
        final_confidence = max(0.0, weighted_confidence - penalties)
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _create_neutral_component(self, component_type: str, reason: str) -> Dict:
        """Create neutral component result."""
        return {
            'score': 0.0,
            'confidence': 0.1,
            'signals': {'error': reason, 'component': component_type}
        }
    
    def _create_neutral_score(self, asset: str, article_url: str, reason: str) -> MultimodalScoreComponents:
        """Create neutral multimodal score."""
        return MultimodalScoreComponents(
            text_score=0.0,
            image_score=0.0,
            market_context_score=0.0,
            meta_signals_score=0.0,
            final_flowscore=0.0,
            text_confidence=0.1,
            image_confidence=0.1,
            market_confidence=0.1,
            overall_confidence=0.1,
            text_signals={'error': reason},
            image_signals={'error': reason},
            market_signals={'error': reason},
            meta_signals={'error': reason},
            processing_timestamp=datetime.now().isoformat(),
            asset=asset,
            article_url=article_url
        )
    
    def get_processing_stats(self) -> Dict:
        """Get comprehensive processing statistics."""
        if self.processing_stats['total_articles'] == 0:
            return {'message': 'No articles processed yet'}
        
        success_rate = (
            self.processing_stats['successful_scores'] / 
            self.processing_stats['total_articles'] * 100
        )
        
        return {
            'total_articles_processed': self.processing_stats['total_articles'],
            'successful_scores': self.processing_stats['successful_scores'],
            'success_rate_percent': round(success_rate, 2),
            'component_failure_rates': {
                component: (
                    failures / self.processing_stats['total_articles'] * 100
                    if self.processing_stats['total_articles'] > 0 else 0
                )
                for component, failures in self.processing_stats['component_failures'].items()
            },
            'component_weights_used': self.COMPONENT_WEIGHTS
        }


# Global multimodal scorer instance
multimodal_scorer = MultimodalScorer()
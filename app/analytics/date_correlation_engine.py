"""
Date Correlation Engine
Extracts and correlates dates from scraped articles and images with price data.
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
from collections import defaultdict

from app.analytics.unified_options_analyzer import unified_analyzer
from app.market_data.price_data_loader import price_loader

logger = logging.getLogger(__name__)

@dataclass
class DateExtraction:
    """Represents a date extracted from an article or image."""
    source_type: str  # 'article', 'image'
    source_id: str    # URL or image hash
    extracted_date: str  # YYYY-MM-DD format
    confidence: float    # 0.0 to 1.0
    extraction_method: str  # 'published_at', 'title', 'body_text', 'ocr', 'vision'
    raw_text: str       # Original text containing the date
    content_summary: str  # Brief summary of the content

@dataclass
class CorrelatedAnalysis:
    """Analysis result correlating article/image content with price data."""
    date: str
    asset: str
    article_data: Optional[Dict[str, Any]] = None
    image_data: Optional[Dict[str, Any]] = None
    price_context: Optional[Dict[str, Any]] = None
    correlation_strength: float = 0.0
    sentiment_alignment: str = 'unknown'  # 'aligned', 'contrarian', 'neutral'
    market_impact_score: float = 0.0

class DateCorrelationEngine:
    """Engine for extracting dates and correlating content with price data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize date correlation engine."""
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / 'data'
        self.scraped_data_dir = Path(__file__).parent.parent.parent / 'scraped_data'
        self.test_results_dir = Path(__file__).parent.parent.parent / 'test_results'
        
        # Cache for extracted dates
        self._date_cache: Dict[str, List[DateExtraction]] = {}
        
        # Load all data
        self._load_articles()
        self._load_image_results()
        
    def _load_articles(self):
        """Load scraped articles data."""
        self.articles = []
        
        # Load cleaned articles
        cleaned_file = self.scraped_data_dir / 'cleaned' / 'articles_cleaned.json'
        if cleaned_file.exists():
            with open(cleaned_file, 'r') as f:
                self.articles = json.load(f)
            logger.info(f"Loaded {len(self.articles)} cleaned articles")
        else:
            logger.warning("No cleaned articles found")
    
    def _load_image_results(self):
        """Load image analysis results."""
        self.image_results = []
        
        image_file = self.test_results_dir / 'image_analysis.json'
        if image_file.exists():
            with open(image_file, 'r') as f:
                data = json.load(f)
                self.image_results = data.get('results', [])
            logger.info(f"Loaded {len(self.image_results)} image analysis results")
        else:
            logger.warning("No image analysis results found")
    
    def extract_dates_from_articles(self) -> List[DateExtraction]:
        """Extract dates from all articles."""
        extractions = []
        
        for article in self.articles:
            article_extractions = self._extract_dates_from_single_article(article)
            extractions.extend(article_extractions)
        
        logger.info(f"Extracted {len(extractions)} dates from {len(self.articles)} articles")
        return extractions
    
    def _extract_dates_from_single_article(self, article: Dict[str, Any]) -> List[DateExtraction]:
        """Extract dates from a single article."""
        extractions = []
        article_id = article.get('url', 'unknown')
        title = article.get('title', '')
        
        # Method 1: Extract from body text timestamps
        body_text = article.get('body_text', '')
        if body_text:
            # Look for patterns like "By Tony Stewart|2025-08-11T09:53:02+00:00August 11, 2025|"
            timestamp_patterns = [
                r'By\s+[^|]*\|(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}[^|]*\|',
                r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}',
                r'(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}:\d{2}'
            ]
            
            for pattern in timestamp_patterns:
                matches = re.findall(pattern, body_text)
                for match in matches:
                    if isinstance(match, tuple):
                        date_str = '-'.join(match[:3])  # Take first 3 elements for YYYY-MM-DD
                    else:
                        date_str = match
                    
                    try:
                        # Validate date
                        datetime.strptime(date_str, '%Y-%m-%d')
                        
                        # Check if date is reasonable (between 2021 and 2026)
                        year = int(date_str.split('-')[0])
                        if 2021 <= year <= 2026:
                            extractions.append(DateExtraction(
                                source_type='article',
                                source_id=article_id,
                                extracted_date=date_str,
                                confidence=0.9,
                                extraction_method='body_timestamp',
                                raw_text=pattern,
                                content_summary=title[:100]
                            ))
                    except ValueError:
                        continue
        
        # Method 2: Extract from title
        if title:
            # Look for date patterns in title
            title_patterns = [
                r'(\d{4})-(\d{2})-(\d{2})',
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})',
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})'
            ]
            
            for pattern in title_patterns:
                matches = re.findall(pattern, title, re.IGNORECASE)
                for match in matches:
                    try:
                        if len(match) == 3 and match[0].isdigit():
                            # YYYY-MM-DD format
                            date_str = f"{match[0]}-{match[1].zfill(2)}-{match[2].zfill(2)}"
                        elif len(match) == 3:
                            # Month name format
                            month_name, day, year = match
                            month_num = self._month_name_to_number(month_name)
                            if month_num:
                                date_str = f"{year}-{month_num:02d}-{int(day):02d}"
                            else:
                                continue
                        else:
                            continue
                        
                        # Validate date
                        datetime.strptime(date_str, '%Y-%m-%d')
                        year = int(date_str.split('-')[0])
                        if 2021 <= year <= 2026:
                            extractions.append(DateExtraction(
                                source_type='article',
                                source_id=article_id,
                                extracted_date=date_str,
                                confidence=0.7,
                                extraction_method='title',
                                raw_text=title,
                                content_summary=title
                            ))
                    except (ValueError, IndexError):
                        continue
        
        # Method 3: Parse published_at_utc if it looks valid
        published_at = article.get('published_at_utc', '')
        if published_at:
            try:
                dt = pd.to_datetime(published_at)
                year = dt.year
                # Only trust published dates that are reasonable
                if 2021 <= year <= 2026:
                    date_str = dt.strftime('%Y-%m-%d')
                    extractions.append(DateExtraction(
                        source_type='article',
                        source_id=article_id,
                        extracted_date=date_str,
                        confidence=0.8,
                        extraction_method='published_at',
                        raw_text=published_at,
                        content_summary=title
                    ))
            except:
                pass
        
        return extractions
    
    def _month_name_to_number(self, month_name: str) -> Optional[int]:
        """Convert month name to number."""
        months = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5, 'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        return months.get(month_name.lower())
    
    def extract_dates_from_images(self) -> List[DateExtraction]:
        """Extract dates from image analysis results."""
        extractions = []
        
        for result in self.image_results:
            image_extractions = self._extract_dates_from_single_image(result)
            extractions.extend(image_extractions)
        
        logger.info(f"Extracted {len(extractions)} dates from {len(self.image_results)} images")
        return extractions
    
    def _extract_dates_from_single_image(self, result: Dict[str, Any]) -> List[DateExtraction]:
        """Extract dates from a single image analysis result."""
        extractions = []
        
        # Get image identifier
        image_id = result.get('image_hash', result.get('image_id', 'unknown'))
        
        # Method 1: Extract from OCR text
        ocr_text = result.get('ocr_text', '')
        if ocr_text:
            date_patterns = [
                r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
                r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
                r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, ocr_text, re.IGNORECASE)
                for match in matches:
                    try:
                        if len(match) == 3:
                            if match[0].isdigit() and len(match[0]) == 4:
                                # YYYY-MM-DD or YYYY/MM/DD
                                date_str = f"{match[0]}-{int(match[1]):02d}-{int(match[2]):02d}"
                            elif match[2].isdigit() and len(match[2]) == 4:
                                # MM/DD/YYYY or DD/MM/YYYY
                                date_str = f"{match[2]}-{int(match[0]):02d}-{int(match[1]):02d}"
                            else:
                                # Month name format
                                month_num = self._month_name_to_number(match[0])
                                if month_num:
                                    date_str = f"{match[2]}-{month_num:02d}-{int(match[1]):02d}"
                                else:
                                    continue
                            
                            # Validate date
                            datetime.strptime(date_str, '%Y-%m-%d')
                            year = int(date_str.split('-')[0])
                            if 2021 <= year <= 2026:
                                extractions.append(DateExtraction(
                                    source_type='image',
                                    source_id=image_id,
                                    extracted_date=date_str,
                                    confidence=0.6,
                                    extraction_method='ocr',
                                    raw_text=ocr_text[:200],
                                    content_summary=f"Image analysis: {result.get('classification', {}).get('type', 'unknown')}"
                                ))
                    except (ValueError, IndexError):
                        continue
        
        # Method 2: Extract from vision analysis content
        vision_content = result.get('vision_analysis', {}).get('content', '')
        if vision_content:
            # Similar pattern matching as OCR but with lower confidence
            date_patterns = [r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})']
            
            for pattern in date_patterns:
                matches = re.findall(pattern, vision_content)
                for match in matches:
                    try:
                        date_str = f"{match[0]}-{int(match[1]):02d}-{int(match[2]):02d}"
                        datetime.strptime(date_str, '%Y-%m-%d')
                        year = int(date_str.split('-')[0])
                        if 2021 <= year <= 2026:
                            extractions.append(DateExtraction(
                                source_type='image',
                                source_id=image_id,
                                extracted_date=date_str,
                                confidence=0.4,
                                extraction_method='vision',
                                raw_text=vision_content[:200],
                                content_summary=f"Vision analysis content"
                            ))
                    except (ValueError, IndexError):
                        continue
        
        return extractions
    
    def correlate_content_with_prices(
        self,
        date_extractions: List[DateExtraction],
        assets: List[str] = ['BTC', 'ETH']
    ) -> List[CorrelatedAnalysis]:
        """Correlate extracted dates with price data and analysis."""
        correlations = []
        
        # Group extractions by date
        date_groups = defaultdict(list)
        for extraction in date_extractions:
            date_groups[extraction.extracted_date].append(extraction)
        
        for date_str, extractions in date_groups.items():
            for asset in assets:
                try:
                    # Get price context for this date and asset
                    price_context = unified_analyzer.analyze_options_context(
                        asset, date_str, include_image_analysis=False
                    )
                    
                    # Separate article and image data
                    article_extractions = [e for e in extractions if e.source_type == 'article']
                    image_extractions = [e for e in extractions if e.source_type == 'image']
                    
                    # Create correlation
                    correlation = CorrelatedAnalysis(
                        date=date_str,
                        asset=asset,
                        article_data={
                            'count': len(article_extractions),
                            'extractions': [asdict(e) for e in article_extractions],
                            'confidence_avg': np.mean([e.confidence for e in article_extractions]) if article_extractions else 0
                        } if article_extractions else None,
                        image_data={
                            'count': len(image_extractions),
                            'extractions': [asdict(e) for e in image_extractions],
                            'confidence_avg': np.mean([e.confidence for e in image_extractions]) if image_extractions else 0
                        } if image_extractions else None,
                        price_context=asdict(price_context),
                        correlation_strength=self._calculate_correlation_strength(extractions, price_context),
                        sentiment_alignment=self._assess_sentiment_alignment(extractions, price_context),
                        market_impact_score=self._calculate_market_impact(extractions, price_context)
                    )
                    
                    correlations.append(correlation)
                    
                except Exception as e:
                    logger.warning(f"Failed to correlate {asset} on {date_str}: {e}")
                    continue
        
        logger.info(f"Created {len(correlations)} price-content correlations")
        return correlations
    
    def _calculate_correlation_strength(
        self,
        extractions: List[DateExtraction],
        price_context: Any
    ) -> float:
        """Calculate correlation strength between content and price data."""
        # Base strength on number of sources and confidence
        base_strength = min(len(extractions) * 0.2, 1.0)
        
        # Boost for high confidence extractions
        high_confidence_count = sum(1 for e in extractions if e.confidence >= 0.7)
        confidence_boost = min(high_confidence_count * 0.3, 0.5)
        
        return min(base_strength + confidence_boost, 1.0)
    
    def _assess_sentiment_alignment(
        self,
        extractions: List[DateExtraction],
        price_context: Any
    ) -> str:
        """Assess sentiment alignment between content and price movement."""
        # This is simplified - in practice you'd analyze content sentiment
        if hasattr(price_context, 'price_change_1d'):
            if price_context.price_change_1d > 2:
                return 'bullish_price'
            elif price_context.price_change_1d < -2:
                return 'bearish_price'
        
        return 'neutral'
    
    def _calculate_market_impact(
        self,
        extractions: List[DateExtraction],
        price_context: Any
    ) -> float:
        """Calculate estimated market impact score."""
        # Base impact on content volume and price volatility
        content_factor = min(len(extractions) / 5.0, 1.0)
        
        volatility_factor = 0.5
        if hasattr(price_context, 'realized_vol_30d'):
            volatility_factor = min(price_context.realized_vol_30d / 100.0, 1.0)
        
        return content_factor * volatility_factor
    
    def generate_correlation_report(
        self,
        correlations: List[CorrelatedAnalysis],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """Generate comprehensive correlation analysis report."""
        if start_date or end_date:
            filtered_correlations = []
            for corr in correlations:
                if start_date and corr.date < start_date:
                    continue
                if end_date and corr.date > end_date:
                    continue
                filtered_correlations.append(corr)
            correlations = filtered_correlations
        
        if not correlations:
            return {'error': 'No correlations found for the specified period'}
        
        # Basic statistics
        total_correlations = len(correlations)
        dates_covered = len(set(c.date for c in correlations))
        
        # Content statistics
        articles_correlations = sum(1 for c in correlations if c.article_data)
        images_correlations = sum(1 for c in correlations if c.image_data)
        
        # Correlation strength analysis
        correlation_strengths = [c.correlation_strength for c in correlations]
        avg_correlation_strength = np.mean(correlation_strengths)
        
        # Market impact analysis
        impact_scores = [c.market_impact_score for c in correlations]
        avg_impact_score = np.mean(impact_scores)
        
        # Date range analysis
        dates = [c.date for c in correlations]
        date_range = {
            'start': min(dates),
            'end': max(dates),
            'span_days': (pd.to_datetime(max(dates)) - pd.to_datetime(min(dates))).days
        }
        
        # Top correlation examples
        top_correlations = sorted(correlations, key=lambda x: x.correlation_strength, reverse=True)[:5]
        
        report = {
            'summary': {
                'total_correlations': total_correlations,
                'unique_dates': dates_covered,
                'articles_with_correlations': articles_correlations,
                'images_with_correlations': images_correlations,
                'average_correlation_strength': avg_correlation_strength,
                'average_market_impact': avg_impact_score
            },
            'date_coverage': date_range,
            'top_correlations': [
                {
                    'date': corr.date,
                    'asset': corr.asset,
                    'correlation_strength': corr.correlation_strength,
                    'sentiment_alignment': corr.sentiment_alignment,
                    'has_articles': corr.article_data is not None,
                    'has_images': corr.image_data is not None
                }
                for corr in top_correlations
            ],
            'analysis_quality': {
                'high_confidence_correlations': sum(1 for c in correlations if c.correlation_strength >= 0.7),
                'medium_confidence_correlations': sum(1 for c in correlations if 0.4 <= c.correlation_strength < 0.7),
                'low_confidence_correlations': sum(1 for c in correlations if c.correlation_strength < 0.4)
            }
        }
        
        return report


# Global instance
date_correlator = DateCorrelationEngine()


def analyze_content_dates() -> Dict[str, Any]:
    """Analyze all content dates and create correlations."""
    # Extract dates from all sources
    article_dates = date_correlator.extract_dates_from_articles()
    image_dates = date_correlator.extract_dates_from_images()
    
    all_dates = article_dates + image_dates
    
    # Create correlations
    correlations = date_correlator.correlate_content_with_prices(all_dates)
    
    # Generate report
    report = date_correlator.generate_correlation_report(correlations)
    
    return {
        'date_extractions': {
            'articles': len(article_dates),
            'images': len(image_dates),
            'total': len(all_dates)
        },
        'correlations': len(correlations),
        'report': report
    }
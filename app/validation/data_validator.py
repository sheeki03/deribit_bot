import re
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from app.core.config import settings
from app.core.logging import logger


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float  # 0-100
    metadata: Dict[str, Any]


class DataValidator:
    """
    Comprehensive data validation for option flows content and market data.
    
    Features:
    - Content quality assessment
    - Market data integrity checks  
    - Image validation and analysis
    - Duplicate detection
    - Anomaly detection
    - Data freshness validation
    """
    
    def __init__(self):
        # Validation thresholds
        self.thresholds = {
            'min_content_length': 100,
            'max_content_length': 50000,
            'min_title_length': 10,
            'max_title_length': 200,
            'max_image_size_mb': 10,
            'min_image_width': 100,
            'min_image_height': 100,
            'max_duplicate_similarity': 0.85,
            'max_article_age_days': 30,
            'min_options_relevance': 0.3,
            'max_price_deviation': 0.5,  # 50% price deviation threshold
            'minimum_overall_quality': 40.0  # Configurable minimum quality
        }
        
        # Configurable validation parameters
        self.allowed_domains = ['deribit.com', 'insights.deribit.com']
        self.expected_authors = ['tony stewart', 'deribit', 'tony', 'stewart']
        
        # Content patterns for validation
        self.quality_patterns = {
            'options_keywords': [
                r'\b(?:call|put|option|strike|expiry|delta|gamma|theta|vega)\b',
                r'\b(?:volatility|premium|exercise|assignment|otm|itm|atm)\b',
                r'\b(?:spread|straddle|strangle|collar|butterfly|condor)\b'
            ],
            'spam_indicators': [
                r'click here', r'subscribe now', r'limited time',
                r'act now', r'free trial', r'guaranteed profit'
            ],
            'low_quality_patterns': [
                r'^.{1,50}$',  # Very short content
                r'lorem ipsum',  # Placeholder text
                r'test\s+test',  # Test content
                r'^\s*$'  # Empty content
            ]
        }
        
        # Duplicate detection cache
        self.content_hashes = {}
        
    def validate_article(self, article_data: Dict) -> ValidationResult:
        """
        Comprehensive validation of article data.
        
        Args:
            article_data: Article dictionary with content and metadata
            
        Returns:
            ValidationResult with detailed validation outcome
        """
        errors = []
        warnings = []
        quality_metrics = {}
        
        try:
            # 1. Basic structure validation
            structure_valid, structure_errors = self._validate_structure(article_data)
            errors.extend(structure_errors)
            
            if not structure_valid:
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    quality_score=0.0,
                    metadata={'validation_stage': 'structure'}
                )
            
            # 2. Content quality validation
            content_score, content_warnings = self._validate_content_quality(article_data)
            warnings.extend(content_warnings)
            quality_metrics['content_quality'] = content_score
            
            # 3. URL and metadata validation
            url_score, url_warnings = self._validate_url_metadata(article_data)
            warnings.extend(url_warnings)
            quality_metrics['url_metadata'] = url_score
            
            # 4. Image validation (if images present)
            if article_data.get('images'):
                image_score, image_warnings = self._validate_images(article_data['images'])
                warnings.extend(image_warnings)
                quality_metrics['image_quality'] = image_score
            else:
                quality_metrics['image_quality'] = 50.0  # Neutral score for no images
            
            # 5. Market data validation (if present)
            if article_data.get('market_data'):
                market_score, market_warnings = self._validate_market_data(article_data['market_data'])
                warnings.extend(market_warnings)
                quality_metrics['market_data'] = market_score
            else:
                quality_metrics['market_data'] = 0.0  # No market data
            
            # 6. Duplicate detection
            is_duplicate, dup_warnings = self._check_duplicates(article_data)
            if is_duplicate:
                warnings.append("Potential duplicate content detected")
            
            # 7. Options relevance validation
            relevance_score = self._calculate_options_relevance(article_data)
            quality_metrics['options_relevance'] = relevance_score
            
            if relevance_score < self.thresholds['min_options_relevance']:
                warnings.append(f"Low options relevance: {relevance_score:.2f}")
            
            # 8. Freshness validation
            freshness_valid, freshness_warnings = self._validate_freshness(article_data)
            warnings.extend(freshness_warnings)
            
            # Calculate overall quality score
            overall_quality = self._calculate_quality_score(quality_metrics)
            
            # Determine if article is valid
            is_valid = (
                len(errors) == 0 and
                overall_quality >= self.thresholds['minimum_overall_quality'] and
                relevance_score >= self.thresholds['min_options_relevance']
            )
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                quality_score=overall_quality,
                metadata={
                    'quality_breakdown': quality_metrics,
                    'options_relevance': relevance_score,
                    'is_duplicate': is_duplicate,
                    'freshness_valid': freshness_valid
                }
            )
            
        except Exception as e:
            logger.error(f"Article validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[],
                quality_score=0.0,
                metadata={'exception': str(e)}
            )
    
    def _validate_structure(self, article_data: Dict) -> Tuple[bool, List[str]]:
        """Validate basic article structure."""
        errors = []
        
        # Required fields
        required_fields = ['url', 'title']
        for field in required_fields:
            if not article_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Content fields (at least one required)
        content_fields = ['body_markdown', 'body_html']
        if not any(article_data.get(field) for field in content_fields):
            errors.append("No content found (missing body_markdown and body_html)")
        
        # URL validation
        url = article_data.get('url')
        if url:
            try:
                parsed = urlparse(url)
                if not all([parsed.scheme, parsed.netloc]):
                    errors.append("Invalid URL format")
                
                domain_match = any(domain.lower() in parsed.netloc.lower() 
                                 for domain in self.allowed_domains)
                if not domain_match:
                    errors.append("URL not from Deribit domain")
                    
            except Exception:
                errors.append("URL parsing failed")
        
        return len(errors) == 0, errors
    
    def _validate_content_quality(self, article_data: Dict) -> Tuple[float, List[str]]:
        """Validate content quality and calculate quality score."""
        warnings = []
        quality_score = 100.0
        
        # Get content
        content = article_data.get('body_markdown', '') or article_data.get('body_html', '')
        title = article_data.get('title', '')
        
        # Length checks
        content_length = len(content.strip())
        if content_length < self.thresholds['min_content_length']:
            warnings.append(f"Content too short: {content_length} characters")
            quality_score -= 30
        elif content_length > self.thresholds['max_content_length']:
            warnings.append(f"Content very long: {content_length} characters")
            quality_score -= 10
        
        title_length = len(title.strip())
        if title_length < self.thresholds['min_title_length']:
            warnings.append(f"Title too short: {title_length} characters")
            quality_score -= 15
        elif title_length > self.thresholds['max_title_length']:
            warnings.append(f"Title too long: {title_length} characters")
            quality_score -= 5
        
        # Spam detection
        combined_text = f"{title} {content}".lower()
        spam_count = 0
        for pattern in self.quality_patterns['spam_indicators']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                spam_count += 1
        
        if spam_count > 0:
            warnings.append(f"Potential spam indicators found: {spam_count}")
            quality_score -= min(spam_count * 15, 45)
        
        # Low quality pattern detection
        for pattern in self.quality_patterns['low_quality_patterns']:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append("Low quality content pattern detected")
                quality_score -= 25
                break
        
        # Language and readability
        if not self._is_english_content(combined_text):
            warnings.append("Content may not be in English")
            quality_score -= 20
        
        # Structure quality (paragraphs, sentences)
        structure_score = self._assess_content_structure(content)
        quality_score = quality_score * 0.8 + structure_score * 0.2
        
        return max(0.0, min(100.0, quality_score)), warnings
    
    def _validate_url_metadata(self, article_data: Dict) -> Tuple[float, List[str]]:
        """Validate URL and metadata quality."""
        warnings = []
        quality_score = 100.0
        
        url = article_data.get('url', '')
        title = article_data.get('title', '')
        author = article_data.get('author', '')
        published_at = article_data.get('published_at_utc')
        
        # URL quality
        if 'option-flow' not in url.lower():
            warnings.append("URL doesn't contain 'option-flow'")
            quality_score -= 15
        
        # Title-URL consistency
        if title and url:
            title_words = set(re.findall(r'\w+', title.lower()))
            url_words = set(re.findall(r'\w+', url.lower()))
            
            # Calculate overlap
            if title_words:
                overlap = len(title_words.intersection(url_words)) / len(title_words)
                if overlap < 0.3:
                    warnings.append("Title and URL have low similarity")
                    quality_score -= 10
        
        # Author validation
        if not author:
            warnings.append("No author specified")
            quality_score -= 10
        elif not any(expected.lower() in author.lower() 
                    for expected in self.expected_authors):
            warnings.append(f"Unexpected author: {author}")
            quality_score -= 5
        
        # Publication date validation
        if not published_at:
            warnings.append("No publication date")
            quality_score -= 15
        else:
            try:
                if isinstance(published_at, str):
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    pub_date = published_at
                
                # Check if date is reasonable
                now = datetime.now(timezone.utc)
                if pub_date > now:
                    warnings.append("Publication date in future")
                    quality_score -= 20
                elif (now - pub_date).days > 365:
                    warnings.append("Publication date very old")
                    quality_score -= 5
                    
            except Exception:
                warnings.append("Invalid publication date format")
                quality_score -= 15
        
        return max(0.0, min(100.0, quality_score)), warnings
    
    def _validate_images(self, images_data: List[Dict]) -> Tuple[float, List[str]]:
        """Validate image data quality."""
        warnings = []
        quality_score = 100.0
        
        if not images_data:
            return 50.0, ["No images provided"]
        
        valid_images = 0
        total_images = len(images_data)
        
        for i, image_data in enumerate(images_data):
            image_quality = 100.0
            
            # Required fields
            required_fields = ['image_url', 'image_type']
            for field in required_fields:
                if not image_data.get(field):
                    warnings.append(f"Image {i}: Missing {field}")
                    image_quality -= 30
            
            # Image size validation
            file_size = image_data.get('file_size_bytes', 0)
            if file_size > self.thresholds['max_image_size_mb'] * 1024 * 1024:
                warnings.append(f"Image {i}: File too large ({file_size / (1024*1024):.1f}MB)")
                image_quality -= 20
            elif file_size == 0:
                warnings.append(f"Image {i}: No file size information")
                image_quality -= 10
            
            # Dimensions validation
            width = image_data.get('width', 0)
            height = image_data.get('height', 0)
            
            if width < self.thresholds['min_image_width'] or height < self.thresholds['min_image_height']:
                warnings.append(f"Image {i}: Dimensions too small ({width}x{height})")
                image_quality -= 25
            
            # Image type validation
            image_type = image_data.get('image_type', 'unknown')
            valid_types = ['greeks_chart', 'flow_heatmap', 'skew_chart', 'price_chart', 'position_diagram']
            
            if image_type == 'unknown':
                warnings.append(f"Image {i}: Unknown image type")
                image_quality -= 15
            elif image_type not in valid_types:
                warnings.append(f"Image {i}: Unexpected image type: {image_type}")
                image_quality -= 10
            
            # OCR confidence validation
            confidence = image_data.get('confidence_score', 0)
            if confidence > 0:
                if confidence < 0.3:
                    warnings.append(f"Image {i}: Low OCR confidence ({confidence:.2f})")
                    image_quality -= 20
            else:
                warnings.append(f"Image {i}: No OCR confidence score")
                image_quality -= 10
            
            # Processing status validation
            status = image_data.get('processing_status', 'unknown')
            if status != 'completed':
                warnings.append(f"Image {i}: Processing not completed ({status})")
                image_quality -= 25
            
            if image_quality >= 50:
                valid_images += 1
        
        # Calculate overall image quality score
        if total_images > 0:
            valid_ratio = valid_images / total_images
            quality_score = valid_ratio * 100
        
        return max(0.0, min(100.0, quality_score)), warnings
    
    def _validate_market_data(self, market_data: Dict) -> Tuple[float, List[str]]:
        """Validate market data integrity."""
        warnings = []
        quality_score = 100.0
        
        expected_assets = ['BTC', 'ETH']
        valid_assets = 0
        
        for asset in expected_assets:
            if asset not in market_data:
                warnings.append(f"Missing market data for {asset}")
                quality_score -= 25
                continue
            
            asset_data = market_data[asset]
            asset_quality = 100.0
            
            # Base price validation
            base_price = asset_data.get('base_price')
            if not base_price:
                warnings.append(f"{asset}: Missing base price data")
                asset_quality -= 40
            else:
                price = base_price.get('price', 0)
                if price <= 0:
                    warnings.append(f"{asset}: Invalid price: {price}")
                    asset_quality -= 30
                
                # Price sanity checks
                expected_ranges = {
                    'BTC': (10000, 200000),
                    'ETH': (500, 10000)
                }
                
                if asset in expected_ranges:
                    min_price, max_price = expected_ranges[asset]
                    if not (min_price <= price <= max_price):
                        warnings.append(f"{asset}: Price outside expected range: ${price:,.2f}")
                        asset_quality -= 25
                
                # Time difference validation
                time_diff = base_price.get('time_difference_seconds', 0)
                if time_diff > 3600:  # More than 1 hour difference
                    warnings.append(f"{asset}: Price timestamp too far from publication ({time_diff/3600:.1f}h)")
                    asset_quality -= 15
            
            # Forward returns validation
            forward_returns = asset_data.get('forward_returns', {})
            expected_periods = ['ret_4h', 'ret_24h', 'ret_72h', 'ret_168h']
            
            valid_returns = 0
            for period in expected_periods:
                ret_value = forward_returns.get(period)
                if ret_value is not None:
                    # Check for reasonable return values (-50% to +50%)
                    if abs(ret_value) > 0.5:
                        warnings.append(f"{asset}: Extreme {period} return: {ret_value:.3f}")
                        asset_quality -= 10
                    valid_returns += 1
            
            if valid_returns == 0:
                warnings.append(f"{asset}: No valid forward returns")
                asset_quality -= 20
            
            if asset_quality >= 50:
                valid_assets += 1
        
        # Calculate overall market data quality
        if expected_assets:
            quality_score = (valid_assets / len(expected_assets)) * 100
        
        return max(0.0, min(100.0, quality_score)), warnings
    
    def _check_duplicates(self, article_data: Dict) -> Tuple[bool, List[str]]:
        """Check for duplicate content using content hashing."""
        warnings = []
        
        content = article_data.get('body_markdown', '') or article_data.get('body_html', '')
        
        if not content:
            return False, warnings
        
        # Create content hash
        content_normalized = re.sub(r'\s+', ' ', content.lower().strip())
        content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
        
        # Check against existing hashes
        is_duplicate = content_hash in self.content_hashes
        
        if is_duplicate:
            original_url = self.content_hashes[content_hash]
            warnings.append(f"Duplicate content detected (original: {original_url})")
        else:
            # Store hash for future comparisons
            self.content_hashes[content_hash] = article_data.get('url', 'unknown')
        
        return is_duplicate, warnings
    
    def _calculate_options_relevance(self, article_data: Dict) -> float:
        """Calculate how relevant the content is to options trading."""
        content = article_data.get('body_markdown', '') or article_data.get('body_html', '')
        title = article_data.get('title', '')
        
        combined_text = f"{title} {content}".lower()
        
        if not combined_text.strip():
            return 0.0
        
        # Count options-related keywords
        total_matches = 0
        for pattern in self.quality_patterns['options_keywords']:
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            total_matches += matches
        
        # Normalize by content length
        word_count = len(combined_text.split())
        if word_count == 0:
            return 0.0
        
        # Calculate relevance score (0-1)
        relevance_score = min(total_matches / max(word_count / 100, 1), 1.0)
        
        return relevance_score
    
    def _validate_freshness(self, article_data: Dict) -> Tuple[bool, List[str]]:
        """Validate article freshness."""
        warnings = []
        
        published_at = article_data.get('published_at_utc')
        if not published_at:
            return True, warnings  # Can't validate without date
        
        try:
            if isinstance(published_at, str):
                pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                pub_date = published_at
            
            age_days = (datetime.now(timezone.utc) - pub_date).days
            
            if age_days > self.thresholds['max_article_age_days']:
                warnings.append(f"Article is {age_days} days old")
                return False, warnings
            
            return True, warnings
            
        except Exception as e:
            warnings.append(f"Date validation failed: {e}")
            return False, warnings
    
    def _is_english_content(self, text: str) -> bool:
        """Simple English language detection."""
        if not text:
            return False
        
        # Common English words
        common_words = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return False
        
        common_word_count = sum(1 for word in words if word in common_words)
        return common_word_count / len(words) > 0.1  # At least 10% common English words
    
    def _assess_content_structure(self, content: str) -> float:
        """Assess content structure quality."""
        if not content.strip():
            return 0.0
        
        score = 100.0
        
        # Paragraph structure
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 2:
            score -= 20  # Very little structure
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 3:
            score -= 30  # Too few sentences
        
        # Average sentence length
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            if avg_sentence_length < 5:
                score -= 15  # Very short sentences
            elif avg_sentence_length > 50:
                score -= 10  # Very long sentences
        
        return max(0.0, min(100.0, score))
    
    def _calculate_quality_score(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from component metrics."""
        # Weighted combination of quality metrics
        weights = {
            'content_quality': 0.35,
            'url_metadata': 0.15,
            'image_quality': 0.20,
            'market_data': 0.15,
            'options_relevance': 0.15
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                total_score += quality_metrics[metric] * weight
                total_weight += weight
        
        # Normalize by actual weights used
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def validate_batch(self, articles_data: List[Dict]) -> List[ValidationResult]:
        """Validate multiple articles efficiently."""
        logger.info(f"Validating batch of {len(articles_data)} articles")
        
        results = []
        valid_count = 0
        
        for i, article_data in enumerate(articles_data):
            try:
                result = self.validate_article(article_data)
                results.append(result)
                
                if result.is_valid:
                    valid_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Validated {i + 1}/{len(articles_data)} articles")
                    
            except Exception as e:
                logger.error(f"Batch validation failed for article {i}: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"Validation exception: {str(e)}"],
                    warnings=[],
                    quality_score=0.0,
                    metadata={'batch_index': i}
                ))
        
        logger.info(f"Batch validation completed: {valid_count}/{len(articles_data)} valid")
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Union[int, float]]:
        """Get validation statistics."""
        return {
            'cached_content_hashes': len(self.content_hashes),
            'validation_thresholds': self.thresholds.copy()
        }
    
    def clear_cache(self):
        """Clear validation cache."""
        self.content_hashes.clear()
        logger.info("Validation cache cleared")


# Global data validator instance
data_validator = DataValidator()
import re
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import Counter
import textstat

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from app.core.config import settings
from app.core.logging import logger


class OptionsFeatureExtractor:
    """
    Comprehensive feature extraction for options trading sentiment analysis.
    
    Extracts multiple types of features:
    - Text features (TF-IDF, sentiment lexicons, readability)
    - Numerical features (strikes, notionals, Greeks)
    - Temporal features (timing, market conditions)
    - Multimodal features (from images and charts)
    """
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize NLTK data
        self._download_nltk_data()
        
        # Options-specific vocabulary
        self.options_vocabulary = self._build_options_vocabulary()
        
        # Financial sentiment lexicons
        self.bullish_terms = self._build_bullish_lexicon()
        self.bearish_terms = self._build_bearish_lexicon()
        self.greek_terms = self._build_greek_lexicon()
        self.volatility_terms = self._build_volatility_lexicon()
        
        # Compiled regex patterns for efficiency
        self._compile_regex_patterns()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk_downloads = [
                'punkt', 'stopwords', 'wordnet', 
                'averaged_perceptron_tagger', 'omw-1.4'
            ]
            for dataset in nltk_downloads:
                try:
                    nltk.download(dataset, quiet=True)
                except:
                    pass
        except Exception as e:
            logger.warning("NLTK data download failed", error=str(e))
    
    def _build_options_vocabulary(self) -> Dict[str, List[str]]:
        """Build options trading specific vocabulary."""
        return {
            'option_types': [
                'call', 'calls', 'put', 'puts', 'option', 'options',
                'strike', 'strikes', 'expiry', 'expiries', 'expiration'
            ],
            'strategies': [
                'spread', 'spreads', 'straddle', 'strangle', 'butterfly',
                'condor', 'collar', 'protective', 'covered', 'naked'
            ],
            'actions': [
                'buying', 'selling', 'bought', 'sold', 'writing', 'wrote',
                'exercising', 'assigned', 'expired', 'closing', 'rolling'
            ],
            'size_terms': [
                'whale', 'large', 'massive', 'huge', 'big', 'small',
                'retail', 'institutional', 'block', 'sweep'
            ],
            'urgency': [
                'aggressive', 'passive', 'urgent', 'rushed', 'calm',
                'steady', 'panic', 'frantic', 'systematic'
            ]
        }
    
    def _build_bullish_lexicon(self) -> List[str]:
        """Build bullish sentiment terms specific to options."""
        return [
            'bullish', 'bull', 'upside', 'rally', 'surge', 'spike',
            'call buying', 'call flow', 'upside betting', 'squeeze',
            'gamma squeeze', 'short squeeze', 'breakout', 'momentum',
            'accumulation', 'positioning', 'optimism', 'confident',
            'aggressive calls', 'call spreads', 'protective puts selling'
        ]
    
    def _build_bearish_lexicon(self) -> List[str]:
        """Build bearish sentiment terms specific to options."""
        return [
            'bearish', 'bear', 'downside', 'decline', 'drop', 'crash',
            'put buying', 'put flow', 'downside protection', 'hedging',
            'hedge', 'covering', 'fear', 'panic', 'defensive',
            'protective puts', 'put spreads', 'call selling', 'overwriting'
        ]
    
    def _build_greek_lexicon(self) -> List[str]:
        """Build Greek terms and their sentiment implications."""
        return [
            'delta', 'gamma', 'theta', 'vega', 'rho',
            'delta neutral', 'gamma hedging', 'theta decay',
            'vega risk', 'pin risk', 'assignment risk'
        ]
    
    def _build_volatility_lexicon(self) -> List[str]:
        """Build volatility-related terms."""
        return [
            'volatility', 'vol', 'iv', 'implied vol', 'realized vol',
            'vol surface', 'vol smile', 'vol skew', 'vol crush',
            'vix', 'fear index', 'complacency'
        ]
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.patterns = {
            'strikes': re.compile(r'(\d+[kK]?)\s*(?:strike|level|price)', re.IGNORECASE),
            'notionals': re.compile(r'\$(\d+(?:\.\d+)?)\s*([mMbB]|million|billion)', re.IGNORECASE),
            'percentages': re.compile(r'(\d+(?:\.\d+)?)%'),
            'dates': re.compile(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*\d*)', re.IGNORECASE),
            'contracts': re.compile(r'(\d+(?:\.\d+)?[kK]?)\s*(?:contract|contracts)', re.IGNORECASE),
            'multipliers': re.compile(r'(\d+)x', re.IGNORECASE)
        }
    
    def extract_text_features(self, text: str, title: str = "") -> Dict[str, Union[float, int, List]]:
        """
        Extract comprehensive text features for ML models.
        
        Args:
            text: Article content
            title: Article title (optional)
            
        Returns:
            Dictionary of text features
        """
        if not text:
            return self._empty_text_features()
        
        # Clean text
        clean_text = self._clean_text(text)
        combined_text = f"{title} {clean_text}" if title else clean_text
        
        features = {}
        
        # Basic text statistics
        features.update(self._extract_basic_text_stats(clean_text, title))
        
        # Sentiment lexicon features
        features.update(self._extract_sentiment_features(combined_text))
        
        # Options-specific features
        features.update(self._extract_options_features(combined_text))
        
        # Numerical extraction features
        features.update(self._extract_numerical_features(text))
        
        # Readability and complexity features
        features.update(self._extract_readability_features(clean_text))
        
        # N-gram and keyword features
        features.update(self._extract_keyword_features(clean_text))
        
        return features
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-ASCII characters but keep financial symbols
        text = re.sub(r'[^\x00-\x7F$€£¥%]+', '', text)
        
        return text.strip()
    
    def _extract_basic_text_stats(self, text: str, title: str = "") -> Dict[str, float]:
        """Extract basic text statistics."""
        if not text:
            return {
                'text_length': 0, 'word_count': 0, 'sentence_count': 0,
                'avg_word_length': 0, 'avg_sentence_length': 0,
                'title_length': len(title) if title else 0
            }
        
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'title_length': len(title) if title else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'capital_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def _extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """Extract sentiment features using custom lexicons."""
        if not text:
            return {'bullish_score': 0, 'bearish_score': 0, 'greek_score': 0, 'vol_score': 0}
        
        text_lower = text.lower()
        
        # Count sentiment terms with context weighting
        bullish_score = self._count_weighted_terms(text_lower, self.bullish_terms)
        bearish_score = self._count_weighted_terms(text_lower, self.bearish_terms)
        greek_score = self._count_weighted_terms(text_lower, self.greek_terms)
        vol_score = self._count_weighted_terms(text_lower, self.volatility_terms)
        
        # Normalize by text length
        text_len = len(text.split())
        norm_factor = max(text_len / 100, 1)  # Normalize per 100 words
        
        return {
            'bullish_score': bullish_score / norm_factor,
            'bearish_score': bearish_score / norm_factor,
            'greek_score': greek_score / norm_factor,
            'vol_score': vol_score / norm_factor,
            'sentiment_ratio': (bullish_score - bearish_score) / max(bullish_score + bearish_score, 1)
        }
    
    def _count_weighted_terms(self, text: str, terms: List[str]) -> float:
        """Count terms with positional and context weighting."""
        total_score = 0
        
        for term in terms:
            if ' ' in term:  # Multi-word terms
                count = len(re.findall(re.escape(term), text))
                total_score += count * 2  # Multi-word terms get higher weight
            else:  # Single word terms
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(term) + r'\b'
                count = len(re.findall(pattern, text))
                total_score += count
        
        return total_score
    
    def _extract_options_features(self, text: str) -> Dict[str, float]:
        """Extract options-specific trading features."""
        if not text:
            return {f'{cat}_count': 0 for cat in self.options_vocabulary.keys()}
        
        text_lower = text.lower()
        features = {}
        
        # Count vocabulary categories
        for category, terms in self.options_vocabulary.items():
            count = sum(len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower)) for term in terms)
            features[f'{category}_count'] = count
        
        # Specific options patterns
        features['call_put_ratio'] = self._calculate_call_put_ratio(text_lower)
        features['urgency_score'] = self._calculate_urgency_score(text_lower)
        features['size_emphasis'] = self._calculate_size_emphasis(text_lower)
        
        return features
    
    def _calculate_call_put_ratio(self, text: str) -> float:
        """Calculate call to put mention ratio."""
        call_mentions = len(re.findall(r'\bcall[s]?\b', text))
        put_mentions = len(re.findall(r'\bput[s]?\b', text))
        
        if put_mentions == 0:
            return call_mentions  # If no puts, return call count
        return call_mentions / put_mentions
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate urgency/intensity score."""
        urgency_terms = ['urgent', 'aggressive', 'massive', 'huge', 'panic', 'rush']
        return sum(len(re.findall(r'\b' + term + r'\b', text)) for term in urgency_terms)
    
    def _calculate_size_emphasis(self, text: str) -> float:
        """Calculate emphasis on position sizes."""
        size_terms = ['whale', 'large', 'big', 'massive', 'huge', 'block']
        return sum(len(re.findall(r'\b' + term + r'\b', text)) for term in size_terms)
    
    def _extract_numerical_features(self, text: str) -> Dict[str, Union[float, List]]:
        """Extract numerical data from text."""
        if not text:
            return {
                'strike_count': 0, 'notional_count': 0, 'contract_count': 0,
                'avg_strike': 0, 'total_notional': 0, 'max_multiplier': 0
            }
        
        features = {}
        
        # Extract strikes
        strikes = self.patterns['strikes'].findall(text)
        strike_values = [self._parse_k_notation(s) for s in strikes]
        strike_values = [v for v in strike_values if v > 0]
        
        features['strike_count'] = len(strike_values)
        features['avg_strike'] = np.mean(strike_values) if strike_values else 0
        features['strike_std'] = np.std(strike_values) if len(strike_values) > 1 else 0
        
        # Extract notionals
        notionals = self.patterns['notionals'].findall(text)
        notional_values = []
        for amount, unit in notionals:
            multiplier = {'m': 1e6, 'million': 1e6, 'b': 1e9, 'billion': 1e9}
            value = float(amount) * multiplier.get(unit.lower(), 1)
            notional_values.append(value)
        
        features['notional_count'] = len(notional_values)
        features['total_notional'] = sum(notional_values)
        features['avg_notional'] = np.mean(notional_values) if notional_values else 0
        
        # Extract contracts
        contracts = self.patterns['contracts'].findall(text)
        contract_values = [self._parse_k_notation(c) for c in contracts]
        contract_values = [v for v in contract_values if v > 0]
        
        features['contract_count'] = len(contract_values)
        features['total_contracts'] = sum(contract_values)
        
        # Extract multipliers (e.g., "10x", "50x")
        multipliers = self.patterns['multipliers'].findall(text)
        multiplier_values = [int(m) for m in multipliers]
        features['max_multiplier'] = max(multiplier_values) if multiplier_values else 0
        
        # Extract percentages
        percentages = self.patterns['percentages'].findall(text)
        pct_values = [float(p) for p in percentages]
        features['percentage_count'] = len(pct_values)
        features['max_percentage'] = max(pct_values) if pct_values else 0
        
        return features
    
    def _parse_k_notation(self, value_str: str) -> float:
        """Parse values with K notation (e.g., '50k' -> 50000)."""
        try:
            value_str = value_str.strip().lower()
            if value_str.endswith('k'):
                return float(value_str[:-1]) * 1000
            return float(value_str)
        except (ValueError, AttributeError):
            return 0
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract text readability and complexity features."""
        if not text or len(text) < 10:
            return {
                'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
                'gunning_fog': 0, 'automated_readability': 0, 'avg_syllables': 0
            }
        
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'automated_readability': textstat.automated_readability_index(text),
                'avg_syllables': textstat.avg_syllables_per_word(text)
            }
        except:
            return {
                'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
                'gunning_fog': 0, 'automated_readability': 0, 'avg_syllables': 0
            }
    
    def _extract_keyword_features(self, text: str) -> Dict[str, float]:
        """Extract keyword density and N-gram features."""
        if not text:
            return {'btc_mentions': 0, 'eth_mentions': 0, 'crypto_density': 0}
        
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        # Crypto mentions
        btc_mentions = len(re.findall(r'\b(?:btc|bitcoin)\b', text_lower))
        eth_mentions = len(re.findall(r'\b(?:eth|ethereum)\b', text_lower))
        
        # Crypto density (crypto terms / total words)
        crypto_terms = ['btc', 'bitcoin', 'eth', 'ethereum', 'crypto', 'cryptocurrency']
        crypto_count = sum(len(re.findall(r'\b' + term + r'\b', text_lower)) for term in crypto_terms)
        crypto_density = crypto_count / len(words) if words else 0
        
        return {
            'btc_mentions': btc_mentions,
            'eth_mentions': eth_mentions,
            'crypto_density': crypto_density,
            'asset_ratio': btc_mentions / max(eth_mentions, 1)
        }
    
    def extract_temporal_features(self, 
                                 published_at: datetime,
                                 current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Extract temporal features that affect sentiment impact.
        
        Args:
            published_at: When the article was published
            current_time: Current time (default: now)
            
        Returns:
            Dictionary of temporal features
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Time since publication
        time_diff = current_time - published_at
        hours_since = time_diff.total_seconds() / 3600
        
        # Time of day features (0-23)
        hour_of_day = published_at.hour
        
        # Day of week features (0=Monday, 6=Sunday)
        day_of_week = published_at.weekday()
        
        # Market timing features
        is_weekend = day_of_week >= 5
        is_market_hours = 9 <= hour_of_day <= 16  # Approximate market hours
        is_overnight = hour_of_day < 6 or hour_of_day > 22
        
        # Seasonal features
        month = published_at.month
        quarter = (month - 1) // 3 + 1
        
        return {
            'hours_since_published': hours_since,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': float(is_weekend),
            'is_market_hours': float(is_market_hours),
            'is_overnight': float(is_overnight),
            'month': month,
            'quarter': quarter,
            'hour_sin': np.sin(2 * np.pi * hour_of_day / 24),
            'hour_cos': np.cos(2 * np.pi * hour_of_day / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7)
        }
    
    def extract_multimodal_features(self, 
                                   images_data: List[Dict],
                                   vision_analysis: Dict = None) -> Dict[str, float]:
        """
        Extract features from image analysis and multimodal data.
        
        Args:
            images_data: List of image analysis results
            vision_analysis: Combined vision AI analysis
            
        Returns:
            Dictionary of multimodal features
        """
        features = {}
        
        # Image count and type features
        features['image_count'] = len(images_data)
        
        if not images_data:
            return {
                'image_count': 0, 'greeks_chart_count': 0, 'flow_heatmap_count': 0,
                'avg_ocr_confidence': 0, 'total_image_size': 0, 'vision_sentiment': 0
            }
        
        # Image type distribution
        image_types = [img.get('image_type', 'unknown') for img in images_data]
        type_counter = Counter(image_types)
        
        features.update({
            f'{img_type}_count': count for img_type, count in type_counter.items()
        })
        
        # OCR and confidence features
        ocr_confidences = [img.get('confidence_score', 0) for img in images_data]
        features['avg_ocr_confidence'] = np.mean(ocr_confidences) if ocr_confidences else 0
        features['min_ocr_confidence'] = np.min(ocr_confidences) if ocr_confidences else 0
        
        # Image size features
        image_sizes = [img.get('file_size_bytes', 0) for img in images_data]
        features['total_image_size'] = sum(image_sizes)
        features['avg_image_size'] = np.mean(image_sizes) if image_sizes else 0
        
        # Vision AI sentiment features
        if vision_analysis:
            features['vision_sentiment'] = self._parse_vision_sentiment(vision_analysis)
            features['vision_confidence'] = vision_analysis.get('confidence', 0)
        else:
            features['vision_sentiment'] = 0
            features['vision_confidence'] = 0
        
        return features
    
    def _parse_vision_sentiment(self, vision_analysis: Dict) -> float:
        """Parse sentiment from vision analysis results."""
        combined_sentiment = vision_analysis.get('combined_sentiment', 'neutral')
        
        sentiment_mapping = {
            'bullish': 1.0,
            'bearish': -1.0,
            'neutral': 0.0
        }
        
        return sentiment_mapping.get(combined_sentiment.lower(), 0.0)
    
    def fit_tfidf_vectorizer(self, texts: List[str], max_features: int = 1000):
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of training texts
            max_features: Maximum number of TF-IDF features
        """
        try:
            # Clean texts
            clean_texts = [self._clean_text(text) for text in texts if text]
            
            # Configure TF-IDF with options-specific settings
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.95,  # Maximum document frequency
                strip_accents='unicode',
                lowercase=True
            )
            
            self.tfidf_vectorizer.fit(clean_texts)
            logger.info(f"TF-IDF vectorizer fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")
            
        except Exception as e:
            logger.error("Failed to fit TF-IDF vectorizer", error=str(e))
    
    def get_tfidf_features(self, text: str) -> np.ndarray:
        """
        Get TF-IDF features for a text.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF feature vector
        """
        if not self.tfidf_vectorizer:
            logger.warning("TF-IDF vectorizer not fitted")
            return np.array([])
        
        try:
            clean_text = self._clean_text(text)
            return self.tfidf_vectorizer.transform([clean_text]).toarray()[0]
        except Exception as e:
            logger.error("Failed to extract TF-IDF features", error=str(e))
            return np.zeros(len(self.tfidf_vectorizer.vocabulary_))
    
    def _empty_text_features(self) -> Dict[str, Union[float, int, List]]:
        """Return empty feature dictionary for missing text."""
        return {
            'text_length': 0, 'word_count': 0, 'sentence_count': 0,
            'bullish_score': 0, 'bearish_score': 0, 'sentiment_ratio': 0,
            'strike_count': 0, 'notional_count': 0, 'call_put_ratio': 0
        }


# Global feature extractor instance
feature_extractor = OptionsFeatureExtractor()
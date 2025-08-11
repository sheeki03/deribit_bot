import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, PreTrainedTokenizer, PreTrainedModel
)
from torch.nn.functional import softmax

from app.core.config import settings
from app.core.logging import logger


class FinBERTSentimentModel:
    """
    FinBERT model for financial sentiment analysis optimized for options trading content.
    
    Features:
    - Pre-trained on financial data
    - Optimized for options terminology
    - Batch processing for efficiency
    - Fine-tuning capability
    - Confidence scoring
    - GPU acceleration when available
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = self._setup_device()
        self.max_length = 512
        self.is_loaded = False
        
        # Model paths for fine-tuned versions
        self.model_dir = Path(settings.data_dir) / "models" / "finbert"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Options-specific preprocessing
        self.options_replacements = self._build_options_replacements()
        
        # Performance tracking
        self.prediction_history = []
    
    def _setup_device(self) -> torch.device:
        """Set up the appropriate device (GPU if available, CPU otherwise)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():  # Apple M1/M2
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def _build_options_replacements(self) -> Dict[str, str]:
        """Build dictionary for options terminology normalization."""
        return {
            # Greek normalization
            'delta': 'option delta',
            'gamma': 'option gamma',
            'theta': 'option theta',
            'vega': 'option vega',
            
            # Position normalization
            'long calls': 'bullish call position',
            'short calls': 'bearish call position',
            'long puts': 'bearish put position',
            'short puts': 'bullish put position',
            
            # Flow terminology
            'call flow': 'call buying activity',
            'put flow': 'put buying activity',
            'whale activity': 'large institutional trading',
            'retail flow': 'small individual trading',
            
            # Volatility terminology
            'vol crush': 'volatility decline',
            'vol spike': 'volatility increase',
            'iv expansion': 'implied volatility increase',
            'iv compression': 'implied volatility decrease',
            
            # Market structure
            'gamma squeeze': 'options driven price acceleration',
            'pin risk': 'options expiration price targeting',
            'max pain': 'maximum options loss level'
        }
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load FinBERT model and tokenizer.
        
        Args:
            model_path: Optional path to fine-tuned model
            
        Returns:
            True if loading succeeded
        """
        try:
            logger.info(f"Loading FinBERT model: {model_path or self.model_name}")
            
            # Load tokenizer and model
            model_source = model_path or self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
            
            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                return_all_scores=True
            )
            
            self.is_loaded = True
            logger.info("FinBERT model loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error("Failed to load FinBERT model", error=str(e))
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better FinBERT performance with options content.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Replace options-specific terminology for better understanding
        for original, replacement in self.options_replacements.items():
            text = text.replace(original, replacement)
        
        # Normalize common abbreviations
        text = text.replace("BTC", "Bitcoin")
        text = text.replace("ETH", "Ethereum")
        text = text.replace("IV", "implied volatility")
        text = text.replace("OI", "open interest")
        
        # Handle strike price notation
        import re
        text = re.sub(r'(\d+)k strike', r'\1000 dollar strike price', text)
        text = re.sub(r'(\d+)K strike', r'\1000 dollar strike price', text)
        
        # Truncate if too long (FinBERT has 512 token limit)
        if len(text) > 2000:  # Conservative character limit
            text = text[:2000]
        
        return text
    
    def predict_sentiment(self, 
                         text: str,
                         return_probabilities: bool = True,
                         preprocess: bool = True) -> Dict[str, Union[str, float, Dict]]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text
            return_probabilities: Whether to return class probabilities
            preprocess: Whether to preprocess text
            
        Returns:
            Sentiment prediction with confidence scores
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        if not text or not text.strip():
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
        
        try:
            # Preprocess text
            if preprocess:
                processed_text = self.preprocess_text(text)
            else:
                processed_text = text
            
            # Make prediction
            results = self.pipeline(processed_text)
            
            # Process results (FinBERT returns all scores)
            sentiment_scores = {item['label'].lower(): item['score'] for item in results[0]}
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[primary_sentiment]
            
            # Map to our sentiment scale (-1 to +1)
            sentiment_score = self._map_sentiment_to_score(sentiment_scores)
            
            result = {
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'sentiment_score': sentiment_score,  # -1 to +1 scale
                'text_length': len(processed_text)
            }
            
            if return_probabilities:
                result['probabilities'] = sentiment_scores
            
            # Track prediction for monitoring
            self.prediction_history.append({
                'timestamp': datetime.now().isoformat(),
                'sentiment': primary_sentiment,
                'confidence': confidence,
                'text_length': len(processed_text)
            })
            
            return result
            
        except Exception as e:
            logger.error("FinBERT prediction failed", error=str(e))
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, 
                     texts: List[str],
                     batch_size: int = 16,
                     return_probabilities: bool = True,
                     preprocess: bool = True) -> List[Dict]:
        """
        Predict sentiment for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Processing batch size
            return_probabilities: Whether to return class probabilities
            preprocess: Whether to preprocess texts
            
        Returns:
            List of sentiment predictions
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before making predictions")
        
        if not texts:
            return []
        
        results = []
        
        try:
            # Process in batches for memory efficiency
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess batch
                if preprocess:
                    batch_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # Filter empty texts
                processed_batch = []
                original_indices = []
                
                for idx, text in enumerate(batch_texts):
                    if text and text.strip():
                        processed_batch.append(text)
                        original_indices.append(i + idx)
                    else:
                        # Add empty result for empty text
                        results.append({
                            'sentiment': 'neutral',
                            'confidence': 0.0,
                            'sentiment_score': 0.0,
                            'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                        })
                
                if processed_batch:
                    # Make batch prediction
                    batch_results = self.pipeline(processed_batch)
                    
                    # Process batch results
                    for j, result in enumerate(batch_results):
                        sentiment_scores = {item['label'].lower(): item['score'] for item in result}
                        
                        primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
                        confidence = sentiment_scores[primary_sentiment]
                        sentiment_score = self._map_sentiment_to_score(sentiment_scores)
                        
                        prediction_result = {
                            'sentiment': primary_sentiment,
                            'confidence': confidence,
                            'sentiment_score': sentiment_score,
                            'text_length': len(processed_batch[j])
                        }
                        
                        if return_probabilities:
                            prediction_result['probabilities'] = sentiment_scores
                        
                        results.append(prediction_result)
            
            logger.info(f"Processed batch of {len(texts)} texts")
            
            return results
            
        except Exception as e:
            logger.error("FinBERT batch prediction failed", error=str(e))
            # Return neutral results for all texts
            return [{
                'sentiment': 'neutral',
                'confidence': 0.0,
                'sentiment_score': 0.0,
                'error': str(e)
            }] * len(texts)
    
    def _map_sentiment_to_score(self, sentiment_scores: Dict[str, float]) -> float:
        """
        Map FinBERT sentiment probabilities to -1 to +1 score.
        
        Args:
            sentiment_scores: Dictionary with positive, negative, neutral probabilities
            
        Returns:
            Sentiment score from -1 (bearish) to +1 (bullish)
        """
        positive_prob = sentiment_scores.get('positive', 0)
        negative_prob = sentiment_scores.get('negative', 0)
        neutral_prob = sentiment_scores.get('neutral', 0)
        
        # Weighted score: positive contributes +1, negative contributes -1, neutral contributes 0
        sentiment_score = positive_prob * 1.0 + negative_prob * (-1.0) + neutral_prob * 0.0
        
        # Ensure score is in [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        return sentiment_score
    
    def analyze_options_context(self, text: str) -> Dict[str, Union[float, List[str]]]:
        """
        Analyze options-specific context in text.
        
        Args:
            text: Input text
            
        Returns:
            Options context analysis
        """
        if not text:
            return {'options_relevance': 0.0, 'key_terms': [], 'strategy_mentions': []}
        
        text_lower = text.lower()
        
        # Options relevance terms
        options_terms = [
            'call', 'put', 'option', 'strike', 'expiry', 'delta', 'gamma',
            'theta', 'vega', 'volatility', 'premium', 'exercise', 'assignment'
        ]
        
        # Strategy mentions
        strategy_terms = [
            'spread', 'straddle', 'strangle', 'butterfly', 'condor',
            'collar', 'covered call', 'protective put'
        ]
        
        # Market sentiment terms
        bullish_terms = ['bullish', 'call buying', 'upside', 'squeeze']
        bearish_terms = ['bearish', 'put buying', 'downside', 'hedge', 'protection']
        
        # Calculate relevance scores
        options_count = sum(1 for term in options_terms if term in text_lower)
        options_relevance = min(1.0, options_count / 5.0)  # Normalize to [0, 1]
        
        # Find mentioned terms
        found_terms = [term for term in options_terms if term in text_lower]
        found_strategies = [term for term in strategy_terms if term in text_lower]
        
        # Sentiment context
        bullish_count = sum(1 for term in bullish_terms if term in text_lower)
        bearish_count = sum(1 for term in bearish_terms if term in text_lower)
        
        return {
            'options_relevance': options_relevance,
            'key_terms': found_terms,
            'strategy_mentions': found_strategies,
            'bullish_context': bullish_count,
            'bearish_context': bearish_count,
            'sentiment_context_ratio': (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
        }
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        info = {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'device': str(self.device),
            'max_length': self.max_length
        }
        
        if self.prediction_history:
            recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
            
            sentiments = [p['sentiment'] for p in recent_predictions]
            confidences = [p['confidence'] for p in recent_predictions]
            
            info.update({
                'total_predictions': len(self.prediction_history),
                'recent_avg_confidence': np.mean(confidences) if confidences else 0,
                'recent_sentiment_distribution': {
                    sentiment: sentiments.count(sentiment) / len(sentiments) if sentiments else 0
                    for sentiment in set(sentiments)
                } if sentiments else {}
            })
        
        return info
    
    def benchmark_performance(self, 
                            test_texts: List[str],
                            n_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            test_texts: List of test texts
            n_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        if not self.is_loaded:
            raise ValueError("Model must be loaded before benchmarking")
        
        import time
        
        # Warm up
        if test_texts:
            for _ in range(3):
                self.predict_sentiment(test_texts[0])
        
        # Single text benchmark
        single_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            self.predict_sentiment(test_texts[0] if test_texts else "Test text")
            end_time = time.time()
            single_times.append(end_time - start_time)
        
        # Batch benchmark
        batch_times = []
        if len(test_texts) > 1:
            for _ in range(max(1, n_iterations // 5)):
                start_time = time.time()
                self.predict_batch(test_texts)
                end_time = time.time()
                batch_times.append(end_time - start_time)
        
        results = {
            'single_prediction_avg_ms': np.mean(single_times) * 1000,
            'single_prediction_std_ms': np.std(single_times) * 1000,
        }
        
        if batch_times:
            avg_batch_time = np.mean(batch_times)
            results.update({
                'batch_prediction_avg_ms': avg_batch_time * 1000,
                'batch_throughput_texts_per_sec': len(test_texts) / avg_batch_time if avg_batch_time > 0 else 0,
                'batch_size': len(test_texts)
            })
        
        return results
    
    def clear_history(self):
        """Clear prediction history to save memory."""
        self.prediction_history = []
        logger.info("FinBERT prediction history cleared")


# Global FinBERT model instance
finbert_model = FinBERTSentimentModel()
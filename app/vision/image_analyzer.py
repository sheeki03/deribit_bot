import asyncio
import hashlib
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import easyocr
import numpy as np
import requests
from PIL import Image, ImageEnhance
import base64

from app.core.config import settings
from app.core.logging import logger


class BulletproofImageAnalyzer:
    """
    Bulletproof image analysis system for option flows content.
    
    Handles:
    - Image classification (Greeks charts, flow heatmaps, etc.)
    - OCR text extraction with multiple fallbacks
    - Vision model analysis for chart interpretation
    - Robust error handling and retry logic
    """
    
    def __init__(self):
        self.ocr_reader = None
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': settings.user_agent})
        
        # Initialize OCR reader lazily
        self._init_ocr()
        
        # Image type classification patterns
        self.image_type_patterns = {
            'greeks_chart': [
                r'delta', r'gamma', r'theta', r'vega',
                r'strike', r'greeks', r'exposure'
            ],
            'flow_heatmap': [
                r'heatmap', r'flow', r'volume', r'interest',
                r'notional', r'size', r'concentration'
            ],
            'skew_chart': [
                r'skew', r'implied.*volatility', r'iv',
                r'term.*structure', r'smile'
            ],
            'price_chart': [
                r'price', r'btc', r'eth', r'usd',
                r'candlestick', r'chart', r'timeframe'
            ],
            'position_diagram': [
                r'position', r'strategy', r'payoff',
                r'profit.*loss', r'breakeven'
            ]
        }
    
    def _init_ocr(self):
        """Initialize OCR reader with GPU acceleration and error handling."""
        try:
            # Try GPU first, fallback to CPU if not available
            import torch
            gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
            
            self.ocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
            
            if gpu_available:
                device_type = "CUDA" if torch.cuda.is_available() else "MPS"
                logger.info(f"OCR reader initialized with {device_type} GPU acceleration")
            else:
                logger.info("OCR reader initialized with CPU (no GPU available)")
                
        except Exception as e:
            logger.error("Failed to initialize OCR reader", error=str(e))
            # Fallback to CPU if GPU initialization fails
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("OCR reader initialized with CPU fallback")
            except Exception as e2:
                logger.error("Failed to initialize OCR reader with CPU fallback", error=str(e2))
                self.ocr_reader = None
    
    async def download_image(self, image_url: str, max_size_mb: int = 10) -> Optional[bytes]:
        """
        Download image with robust error handling and size limits.
        
        Args:
            image_url: URL of the image to download
            max_size_mb: Maximum file size in MB
            
        Returns:
            Image bytes or None if failed
        """
        try:
            response = self.session.get(
                image_url,
                timeout=settings.image_timeout_seconds,
                stream=True
            )
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                logger.warning(f"Image too large: {content_length} bytes", url=image_url)
                return None
            
            # Download with size limit
            image_data = b""
            for chunk in response.iter_content(chunk_size=8192):
                image_data += chunk
                if len(image_data) > max_size_mb * 1024 * 1024:
                    logger.warning(f"Image size exceeded limit", url=image_url)
                    return None
            
            logger.info(f"Downloaded image: {len(image_data)} bytes", url=image_url)
            return image_data
            
        except requests.RequestException as e:
            logger.error("Failed to download image", url=image_url, error=str(e))
            return None
        except Exception as e:
            logger.error("Unexpected error downloading image", url=image_url, error=str(e))
            return None
    
    def compute_image_hash(self, image_data: bytes) -> str:
        """Compute SHA-256 hash of image data for deduplication."""
        return hashlib.sha256(image_data).hexdigest()
    
    def save_image(self, image_data: bytes, image_id: str, image_type: str = "unknown") -> str:
        """
        Save image to local storage with organized directory structure.
        
        Args:
            image_data: Raw image bytes
            image_id: Unique identifier for the image
            image_type: Type of image (greeks_chart, flow_heatmap, etc.)
            
        Returns:
            Local file path
        """
        try:
            # Create type-specific directory
            type_dir = Path(settings.images_dir) / image_type
            type_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            image = Image.open(io.BytesIO(image_data))
            ext = image.format.lower() if image.format else 'jpg'
            
            # Save file
            file_path = type_dir / f"{image_id}.{ext}"
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Saved image to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error("Failed to save image", image_id=image_id, error=str(e))
            return ""
    
    def preprocess_image(self, image_data: bytes) -> Tuple[np.ndarray, Image.Image]:
        """
        Preprocess image for better OCR and analysis.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Tuple of (opencv_image, pil_image)
        """
        try:
            # Load as PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return cv_image, pil_image
            
        except Exception as e:
            logger.error("Failed to preprocess image", error=str(e))
            raise
    
    def classify_image_type(self, ocr_text: str, image_url: str = "") -> str:
        """
        Classify image type based on OCR text and URL patterns.
        
        Args:
            ocr_text: Text extracted from image
            image_url: URL of the image (for additional context)
            
        Returns:
            Image type classification
        """
        text_lower = ocr_text.lower()
        url_lower = image_url.lower()
        
        # Score each image type
        type_scores = {}
        for img_type, patterns in self.image_type_patterns.items():
            score = 0
            for pattern in patterns:
                # Check OCR text
                if re.search(pattern, text_lower):
                    score += 2
                # Check URL
                if re.search(pattern, url_lower):
                    score += 1
            type_scores[img_type] = score
        
        # Return type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return 'unknown'
    
    async def extract_text_ocr(self, cv_image: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Extract text using OCR with multiple attempts and preprocessing.
        
        Args:
            cv_image: OpenCV image array
            
        Returns:
            Dictionary with extracted text and confidence
        """
        if not self.ocr_reader:
            return {'text': '', 'confidence': 0.0}
        
        results = []
        
        # Try different preprocessing approaches
        preprocessing_methods = [
            lambda img: img,  # Original
            lambda img: cv2.GaussianBlur(img, (3, 3), 0),  # Slight blur
            lambda img: cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Adaptive threshold
        ]
        
        for i, preprocess_func in enumerate(preprocessing_methods):
            try:
                processed_img = preprocess_func(cv_image.copy())
                
                # Run OCR
                result = self.ocr_reader.readtext(processed_img, paragraph=True)
                
                if result:
                    # Combine all text with confidence scoring
                    texts = []
                    confidences = []
                    
                    for detection in result:
                        if len(detection) >= 3:
                            text = detection[1]
                            confidence = detection[2]
                            if confidence > 0.3:  # Filter low confidence
                                texts.append(text)
                                confidences.append(confidence)
                    
                    if texts:
                        combined_text = ' '.join(texts)
                        avg_confidence = sum(confidences) / len(confidences)
                        results.append({
                            'text': combined_text,
                            'confidence': avg_confidence,
                            'method': f'preprocessing_{i}'
                        })
                
            except Exception as e:
                logger.warning(f"OCR attempt {i} failed", error=str(e))
                continue
        
        # Return best result
        if results:
            best_result = max(results, key=lambda x: x['confidence'])
            return {
                'text': best_result['text'],
                'confidence': best_result['confidence']
            }
        
        return {'text': '', 'confidence': 0.0}
    
    def extract_numerical_data(self, text: str) -> Dict[str, List[Union[str, float]]]:
        """
        Extract numerical data relevant to options trading from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with extracted numerical data
        """
        data = {
            'strikes': [],
            'notionals': [],
            'percentages': [],
            'dates': [],
            'prices': []
        }
        
        try:
            # Extract strike prices (e.g., "110k", "50000", "$65K")
            strike_patterns = [
                r'(\d+[kK])\s*(?:strike|level|price)',
                r'(\d{4,6})\s*(?:strike|level)',
                r'\$(\d+[kK]?)\s*(?:strike|call|put)'
            ]
            
            for pattern in strike_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    data['strikes'].append(match)
            
            # Extract notional amounts (e.g., "$25m", "$1.4M", "25 million")
            notional_patterns = [
                r'\$(\d+(?:\.\d+)?)\s*([mM]|million)',
                r'\$(\d+(?:\.\d+)?[kK])',
                r'(\d+(?:\.\d+)?)\s*million'
            ]
            
            for pattern in notional_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        data['notionals'].append(f"{match[0]}{match[1]}")
                    else:
                        data['notionals'].append(match)
            
            # Extract percentages (e.g., "25%", "1.5%")
            pct_matches = re.findall(r'(\d+(?:\.\d+)?)%', text)
            data['percentages'].extend(pct_matches)
            
            # Extract dates (e.g., "Dec 27", "December", "Jan8")
            date_patterns = [
                r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*\d*)',
                r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)'
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                data['dates'].extend(matches)
            
            # Extract prices (e.g., "65000", "$BTC 45k")
            price_patterns = [
                r'(\d{4,6})\s*(?:USD|usd|\$)',
                r'(\d+[kK])\s*(?:level|price|BTC|ETH)'
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                data['prices'].extend(matches)
            
        except Exception as e:
            logger.error("Failed to extract numerical data", error=str(e))
        
        return data
    
    async def analyze_image_comprehensive(self, image_data: bytes, image_url: str = "") -> Dict:
        """
        Comprehensive image analysis combining OCR, classification, and data extraction.
        
        Args:
            image_data: Raw image bytes
            image_url: URL of the image
            
        Returns:
            Complete analysis results
        """
        try:
            # Preprocess image
            cv_image, pil_image = self.preprocess_image(image_data)
            
            # Get image dimensions
            height, width = cv_image.shape[:2]
            
            # Extract text via OCR
            ocr_result = await self.extract_text_ocr(cv_image)
            
            # Classify image type
            image_type = self.classify_image_type(ocr_result['text'], image_url)
            
            # Extract numerical data
            numerical_data = self.extract_numerical_data(ocr_result['text'])
            
            # Calculate image hash
            image_hash = self.compute_image_hash(image_data)
            
            # Analyze sentiment indicators from text
            sentiment_indicators = self._extract_sentiment_indicators(ocr_result['text'])
            
            analysis_result = {
                'image_hash': image_hash,
                'image_type': image_type,
                'dimensions': {'width': width, 'height': height},
                'file_size_bytes': len(image_data),
                'ocr_text': ocr_result['text'],
                'ocr_confidence': ocr_result['confidence'],
                'extracted_data': numerical_data,
                'sentiment_indicators': sentiment_indicators,
                'processing_status': 'completed'
            }
            
            logger.info(
                "Image analysis completed",
                image_type=image_type,
                ocr_confidence=ocr_result['confidence'],
                text_length=len(ocr_result['text'])
            )
            
            return analysis_result
            
        except Exception as e:
            logger.error("Comprehensive image analysis failed", error=str(e))
            return {
                'processing_status': 'failed',
                'error': str(e),
                'image_hash': self.compute_image_hash(image_data) if image_data else None
            }
    
    def _extract_sentiment_indicators(self, text: str) -> Dict[str, List[str]]:
        """
        Extract sentiment indicators specific to options trading.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with bullish/bearish indicators
        """
        indicators = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        # Bullish patterns
        bullish_patterns = [
            r'call\s+buying',
            r'upside\s+interest',
            r'gamma\s+squeeze',
            r'skew\s+eased',
            r'bullish\s+flow'
        ]
        
        # Bearish patterns
        bearish_patterns = [
            r'put\s+buying',
            r'hedging\s+activity',
            r'downside\s+protection',
            r'skew\s+firmed',
            r'bearish\s+flow'
        ]
        
        text_lower = text.lower()
        
        for pattern in bullish_patterns:
            matches = re.findall(pattern, text_lower)
            indicators['bullish'].extend(matches)
        
        for pattern in bearish_patterns:
            matches = re.findall(pattern, text_lower)
            indicators['bearish'].extend(matches)
        
        return indicators


# Global analyzer instance
image_analyzer = BulletproofImageAnalyzer()
from __future__ import annotations

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    import pytesseract  # type: ignore
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore
    Image = None  # type: ignore
    ImageEnhance = None  # type: ignore
    ImageFilter = None  # type: ignore

@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    language: str = 'eng'
    dpi: int = 300
    psm: int = 6  # Page segmentation mode
    oem: int = 3  # OCR engine mode
    enhance_contrast: bool = True
    enhance_sharpness: bool = True
    apply_threshold: bool = True
    custom_config: str = ''
    
    def to_tesseract_config(self) -> str:
        """Convert to tesseract config string."""
        base_config = f'--oem {self.oem} --psm {self.psm} -l {self.language}'
        if self.dpi != 300:
            base_config += f' --dpi {self.dpi}'
        if self.custom_config:
            base_config += f' {self.custom_config}'
        return base_config

def _preprocess_image(img: 'Image.Image', config: OCRConfig) -> 'Image.Image':
    """Apply image preprocessing to improve OCR accuracy."""
    if not (ImageEnhance and ImageFilter):
        return img
        
    processed = img.copy()
    
    # Convert to RGB if necessary
    if processed.mode != 'RGB':
        processed = processed.convert('RGB')
    
    # Apply enhancements
    if config.enhance_contrast:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.3)  # Increase contrast
        
    if config.enhance_sharpness:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.5)  # Increase sharpness
        
    if config.apply_threshold:
        # Convert to grayscale and apply threshold
        processed = processed.convert('L')  # Grayscale
        # Apply adaptive threshold-like effect
        processed = processed.point(lambda x: 255 if x > 128 else 0, mode='1')
        
    return processed

def ocr_image(path: str, config: Optional[OCRConfig] = None) -> Optional[str]:
    """Run OCR on an image with configurable preprocessing and settings.
    
    Args:
        path: Path to the image file
        config: OCR configuration. If None, uses default settings.
        
    Returns:
        Extracted text or None if OCR is unavailable or fails
    """
    if pytesseract is None or Image is None:
        return None
        
    if config is None:
        config = OCRConfig()
        
    try:
        with Image.open(path) as img:
            # Preprocess the image
            processed_img = _preprocess_image(img, config)
            
            # Run OCR with configuration
            tesseract_config = config.to_tesseract_config()
            text = pytesseract.image_to_string(processed_img, config=tesseract_config)
            
            return text.strip() if text else None
            
    except Exception:
        return None

def ocr_image_detailed(path: str, config: Optional[OCRConfig] = None) -> Dict[str, Any]:
    """Run OCR with detailed results including confidence scores.
    
    Returns:
        Dictionary with 'text', 'confidence', 'word_count', and 'success' keys
    """
    if pytesseract is None or Image is None:
        return {
            'text': None,
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': 'pytesseract not available'
        }
        
    if config is None:
        config = OCRConfig()
        
    try:
        with Image.open(path) as img:
            processed_img = _preprocess_image(img, config)
            tesseract_config = config.to_tesseract_config()
            
            # Get text
            text = pytesseract.image_to_string(processed_img, config=tesseract_config)
            text = text.strip() if text else ''
            
            # Get detailed data including confidence
            try:
                data = pytesseract.image_to_data(processed_img, config=tesseract_config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                word_count = len([word for word in data['text'] if word.strip()])
            except Exception:
                avg_confidence = 0.0
                word_count = len(text.split()) if text else 0
            
            return {
                'text': text if text else None,
                'confidence': avg_confidence,
                'word_count': word_count,
                'success': True,
                'preprocessing_applied': {
                    'contrast_enhanced': config.enhance_contrast,
                    'sharpness_enhanced': config.enhance_sharpness,
                    'threshold_applied': config.apply_threshold
                }
            }
            
    except Exception as e:
        return {
            'text': None,
            'confidence': 0.0,
            'word_count': 0,
            'success': False,
            'error': str(e)
        }

def get_optimal_ocr_config_for_charts() -> OCRConfig:
    """Get OCR configuration optimized for financial charts and trading interfaces."""
    return OCRConfig(
        language='eng',
        dpi=300,
        psm=6,  # Uniform block of text
        oem=3,  # Default OCR engine
        enhance_contrast=True,
        enhance_sharpness=True,
        apply_threshold=True,
        custom_config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,%-+$€£¥₿:/()[]{}"\' \\ '
    )

# Backward compatibility
ocr_image_simple = ocr_image  # Alias for the basic function

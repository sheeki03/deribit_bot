from __future__ import annotations

from typing import Dict, Optional

from app.ml.finbert_model import finbert_model, FinBERTSentimentModel


def _map_finbert_to_direction(label: str) -> str:
    label_l = (label or "").lower()
    if label_l == "positive":
        return "bullish"
    if label_l == "negative":
        return "bearish"
    return "neutral"


def score_article_text(title: Optional[str], body_text: Optional[str], model: Optional[FinBERTSentimentModel] = None) -> Dict[str, float | str]:
    """
    Score article text with FinBERT and return sentiment label and confidence.

    Args:
        title: Article title
        body_text: Main article body text
        model: Optional injected FinBERTSentimentModel for testing; defaults to global `finbert_model`.

    Returns:
        dict with fields: {
            'sentiment': 'bullish'|'neutral'|'bearish',
            'confidence': float in [0,1],
            'probabilities': {'positive': p, 'neutral': p, 'negative': p},
            'text_length': int
        }
    """
    m = model or finbert_model

    combined = ((title or "").strip() + "\n" + (body_text or "").strip()).strip()
    if not combined:
        return {
            'sentiment': 'neutral',
            'confidence': 0.0,
            'probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
            'text_length': 0,
        }

    # Ensure model is loaded if using the global instance
    if m is finbert_model and not m.is_loaded:
        # Lazy load to avoid heavy startup elsewhere
        loaded = m.load_model()
        if not loaded:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {'positive': 0.33, 'neutral': 0.34, 'negative': 0.33},
                'text_length': len(combined),
                'error': 'model_load_failed'
            }

    pred = m.predict_sentiment(combined, return_probabilities=True, preprocess=True)
    probs: Dict[str, float] = pred.get('probabilities', {})  # type: ignore

    # Determine dominant class
    if probs:
        primary = max(probs.items(), key=lambda x: x[1])[0]
        confidence = float(probs[primary])
    else:
        primary = pred.get('sentiment', 'neutral')  # fallback
        confidence = float(pred.get('confidence', 0.0))

    return {
        'sentiment': _map_finbert_to_direction(primary),
        'confidence': confidence,
        'probabilities': {
            'positive': float(probs.get('positive', 0.0)),
            'neutral': float(probs.get('neutral', 0.0)),
            'negative': float(probs.get('negative', 0.0)),
        },
        'text_length': int(pred.get('text_length', len(combined)))
    }

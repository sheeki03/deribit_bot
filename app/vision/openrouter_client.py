from __future__ import annotations

import os
import base64
from typing import Any, Dict, List, Optional
import json
import time
import logging
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-vl-32b-instruct:free")
DEFAULT_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "120"))
DEFAULT_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))


class OpenRouterVisionClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, 
                 model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT,
                 max_retries: int = DEFAULT_MAX_RETRIES):
        self.api_key = api_key or OPENROUTER_KEY
        self.base_url = base_url or OPENROUTER_BASE
        self.model = model or MODEL_NAME
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests

    @staticmethod
    @lru_cache(maxsize=32)
    def encode_image_to_data_url(path: str) -> str:
        """Encode image to data URL with caching and memory optimization."""
        try:
            # Use more specific MIME type detection
            ext = os.path.splitext(path)[1].lower().strip('.')
            mime_types = {
                'png': 'image/png',
                'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'webp': 'image/webp',
                'bmp': 'image/bmp',
                'tiff': 'image/tiff', 'tif': 'image/tiff'
            }
            mime = mime_types.get(ext, 'image/jpeg')
            
            # Stream-based encoding to reduce memory usage
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            raise ValueError(f"Failed to encode image {path}: {e}") from e

    @staticmethod
    def _strip_code_fences(s: str) -> str:
        """Clean JSON response by removing code fences and common formatting issues."""
        text = s.strip()
        
        # Handle code fences more efficiently
        if text.startswith("```"):
            lines = text.splitlines()
            start_idx = 1
            
            # Skip language specifier if present
            if lines and lines[0].strip().lower().startswith(("```json", "```javascript")):
                start_idx = 1
            elif len(lines) > 1 and lines[1].strip().lower() in ("json", "javascript"):
                start_idx = 2
                
            # Find end fence
            end_idx = len(lines)
            for i in range(len(lines) - 1, start_idx - 1, -1):
                if lines[i].strip().startswith("```"):
                    end_idx = i
                    break
                    
            text = "\n".join(lines[start_idx:end_idx]).strip()
        
        return text

    def analyze_image(self, image_path: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        sys_msg = system_prompt or (
            """
You are a senior crypto derivatives and volatility analyst. You will be given a single image (chart/heatmap/options table/market dashboard). Return STRICT JSON ONLY (no markdown, no code fences). Include:
{
  "chart_meta": {
    "chart_type": "line|candlestick|heatmap|table|term_structure|other",
    "assets": ["BTC","ETH", "..."],
    "source": "if visible",
    "time_range": {"start":"YYYY-MM-DD","end":"YYYY-MM-DD","interval":"e.g., 1h, 15m, daily"},
    "axes": {"x":"label+units if known", "y":"label+units if known"}
  },
  "content": {
    "series": [
      {"name":"e.g., ATM 7", "type":"line|bar|...","approx_range":[low,high], "notes":"legend/color mapping if visible"}
    ],
    "levels": [
      {"label":"support/resistance/strike", "value": 42000, "units":"USD", "why":"reason/annotation"}
    ],
    "events": [
      {"date":"YYYY-MM-DDTHH:MM:SSZ", "desc":"spike/drop/event", "evidence":"what in the image shows this"}
    ],
    "key_points": [
      {"label":"short summary bullet", "value":"concise detail"}
    ]
  },
  "diagnostics": {
    "patterns": ["uptrend|downtrend","volatility_spike","compression","term_structure_contango|backwardation","flow_concentration_by_strike"],
    "options_insight": {
      "term_structure": "contango/backwardation/flat + explanation",
      "skew_or_flows": "e.g., call overwriting near 40k; put demand at 35k"
    }
  },
  "interpretation": {
    "what_it_means": "plain-English explanation of what the image implies",
    "implications": {"BTC":"bullish|bearish|neutral and why", "ETH":"bullish|bearish|neutral and why"},
    "confidence": 0.0,
    "assumptions": ["assumptions if labels unreadable"]
  },
  "ocr_merge": {"used": false, "extracted_text_snippets": []}
}
Return valid JSON only.
"""
        )
        content = [
            {"type": "text", "text": "Analyze this image and return structured JSON as specified."},
            {"type": "image_url", "image_url": {"url": self.encode_image_to_data_url(image_path)}},
        ]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
        
        # Retry on transient errors
        backoff = 1.0
        last_exc: Optional[Exception] = None
        data = None
        request_start = time.time()
        
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = client.post(url, headers=self.headers, json=payload)
                    if resp.status_code in (429, 503):
                        raise httpx.HTTPStatusError("Transient error", request=resp.request, response=resp)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except httpx.HTTPStatusError as e:
                    last_exc = e
                    # Enhanced error context
                    status_code = e.response.status_code if e.response else 'unknown'
                    
                    # Backoff and retry for transient server/rate issues
                    if e.response is not None and e.response.status_code in (429, 503, 502, 504):
                        if attempt < self.max_retries - 1:  # Don't sleep on last attempt
                            time.sleep(backoff)
                            backoff = min(backoff * 2, 60)  # Cap at 60 seconds
                        continue
                    # Non-retryable error
                    raise RuntimeError(f"HTTP {status_code} error on attempt {attempt + 1}: {str(e)}") from e
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError) as e:
                    last_exc = e
                    if attempt < self.max_retries - 1:  # Don't sleep on last attempt
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30)  # Cap at 30 seconds for network errors
            if data is None and last_exc is not None:
                # Enhanced error reporting with request context
                request_time = time.time() - request_start
                error_context = {
                    'model': self.model,
                    'attempts': self.max_retries,
                    'request_duration': f"{request_time:.2f}s",
                    'image_path': image_path
                }
                
                if isinstance(last_exc, httpx.HTTPStatusError) and last_exc.response is not None:
                    status = last_exc.response.status_code
                    try:
                        body_snippet = (last_exc.response.text or "")[:400]
                    except Exception:
                        body_snippet = "<response body unavailable>"
                    
                    raise RuntimeError(
                        f"OpenRouter request failed after {self.max_retries} attempts: "
                        f"HTTP {status} — {body_snippet}. Context: {error_context}"
                    ) from last_exc
                
                raise RuntimeError(
                    f"OpenRouter request failed after {self.max_retries} attempts: {str(last_exc)}. "
                    f"Context: {error_context}"
                ) from last_exc
        # OpenRouter returns OpenAI-compatible schema
        request_time = time.time() - request_start
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected API response structure: {e}") from e
        
        raw = text
        cleaned = self._strip_code_fences(raw)
        parsed: Optional[Dict[str, Any]] = None
        
        # Enhanced JSON parsing with multiple fallback strategies
        parsing_attempts = [
            lambda x: json.loads(x),
            lambda x: json.loads(x.replace(",\n}\n", "\n}\n").replace(",\n]", "\n]")),
            lambda x: json.loads(x.replace(",\n}", "\n}").replace(",\n]", "\n]")),
            lambda x: json.loads(x.replace(",}", "}").replace(",]", "]"))
        ]
        
        for i, parse_func in enumerate(parsing_attempts):
            try:
                parsed = parse_func(cleaned)
                break
            except json.JSONDecodeError:
                if i == len(parsing_attempts) - 1:
                    # Log parsing failure for debugging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"JSON parsing failed for image {image_path}. "
                        f"Raw response length: {len(raw)}. "
                        f"Cleaned response preview: {cleaned[:200]}..."
                    )
                continue
        
        return {
            "raw": raw, 
            "parsed": parsed, 
            "model_used": self.model,
            "request_duration": request_time,
            "parsing_success": parsed is not None
        }

import base64
import io
import json
from typing import Dict, List, Optional, Union

from openai import OpenAI
from PIL import Image

from app.core.config import settings
from app.core.logging import logger


class VisionAIAnalyzer:
    """
    Advanced vision AI analysis using OpenRouter for GPT-4V and Claude-3.5-Sonnet
    for sophisticated chart interpretation and options flow analysis.
    """
    
    def __init__(self):
        # Initialize OpenRouter client
        self.openai_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Chart analysis prompts
        self.chart_analysis_prompts = {
            'greeks_chart': """
            Analyze this Greeks chart for options trading sentiment. Look for:
            1. Delta exposure levels and concentrations
            2. Gamma risk areas and potential squeeze points
            3. Strike levels with high activity
            4. Bullish vs bearish positioning
            5. Unusual or extreme readings
            
            Return a JSON object with:
            - sentiment: "bullish", "bearish", or "neutral"
            - confidence: 0-100
            - key_levels: list of important strike prices
            - gamma_areas: areas of high gamma concentration
            - delta_exposure: net delta exposure if visible
            - trading_signals: actionable insights
            """,
            
            'flow_heatmap': """
            Analyze this options flow heatmap. Identify:
            1. Areas of high volume/interest concentration
            2. Call vs put flow patterns
            3. Unusual activity or whale movements
            4. Strike and expiry concentrations
            5. Flow directional bias
            
            Return JSON with:
            - dominant_flow: "call_heavy", "put_heavy", or "balanced"
            - confidence: 0-100
            - concentration_areas: strike/expiry areas with high activity
            - unusual_activity: any anomalous patterns
            - sentiment_indication: market sentiment from flows
            - whale_activity: evidence of large player activity
            """,
            
            'skew_chart': """
            Analyze this implied volatility skew chart. Look for:
            1. Skew steepness and direction
            2. Put vs call IV levels
            3. Term structure patterns
            4. Unusual skew behavior
            5. Fear/greed indicators
            
            Return JSON with:
            - skew_direction: "put_skew", "call_skew", or "flat"
            - steepness: "steep", "moderate", or "flat"
            - confidence: 0-100
            - fear_greed_indicator: sentiment from skew
            - unusual_patterns: any abnormal skew behavior
            - term_structure: description of time-based patterns
            """,
            
            'price_chart': """
            Analyze this price chart with options context. Identify:
            1. Key support/resistance levels
            2. Technical patterns and trends
            3. Volume patterns if visible
            4. Annotated levels or zones
            5. Correlation with options activity
            
            Return JSON with:
            - trend_direction: "bullish", "bearish", or "sideways"
            - confidence: 0-100
            - key_levels: important price levels
            - technical_patterns: chart patterns identified
            - volume_analysis: volume insights if available
            - options_correlation: how this relates to options flow
            """,
            
            'position_diagram': """
            Analyze this options position/strategy diagram. Look for:
            1. Strategy type (spreads, straddles, etc.)
            2. Profit/loss zones
            3. Breakeven points
            4. Risk/reward profile
            5. Market outlook implied
            
            Return JSON with:
            - strategy_type: identified options strategy
            - market_outlook: "bullish", "bearish", or "neutral"
            - confidence: 0-100
            - breakeven_points: break-even price levels
            - max_profit_loss: risk/reward metrics
            - implied_volatility_view: vol expectations
            """
        }
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64 string."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def _prepare_image_for_analysis(self, image_data: bytes, max_size: int = 1024) -> bytes:
        """
        Prepare image for AI analysis by resizing if necessary.
        
        Args:
            image_data: Raw image bytes
            max_size: Maximum dimension in pixels
            
        Returns:
            Processed image bytes
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            
            # Resize if too large
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=90)
            return output.getvalue()
            
        except Exception as e:
            logger.error("Failed to prepare image", error=str(e))
            return image_data
    
    async def analyze_with_gpt4v(self, image_data: bytes, image_type: str, context: str = "") -> Optional[Dict]:
        """
        Analyze image using GPT-4 Vision via OpenRouter.
        
        Args:
            image_data: Raw image bytes
            image_type: Type of chart/image
            context: Additional context about the image
            
        Returns:
            Analysis results or None if failed
        """
        try:
            # Prepare image
            processed_image = self._prepare_image_for_analysis(image_data)
            base64_image = self._encode_image(processed_image)
            
            # Get appropriate prompt
            prompt = self.chart_analysis_prompts.get(image_type, self.chart_analysis_prompts['price_chart'])
            
            if context:
                prompt += f"\n\nAdditional context: {context}"
            
            # Make API call via OpenRouter
            response = self.openai_client.chat.completions.create(
                model="openai/gpt-4o",  # GPT-4O has vision capabilities
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Look for JSON in response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis_result = json.loads(json_str)
                else:
                    # Fallback: create structured result from text
                    analysis_result = {
                        "analysis": content,
                        "confidence": 70,
                        "model": "gpt-4v",
                        "raw_response": content
                    }
            except json.JSONDecodeError:
                analysis_result = {
                    "analysis": content,
                    "confidence": 50,
                    "model": "gpt-4v",
                    "raw_response": content
                }
            
            logger.info("GPT-4O analysis completed", image_type=image_type)
            return analysis_result
            
        except Exception as e:
            logger.error("GPT-4O analysis failed", error=str(e))
            return None
    
    async def analyze_with_claude(self, image_data: bytes, image_type: str, context: str = "") -> Optional[Dict]:
        """
        Analyze image using Claude-3.5-Sonnet via OpenRouter.
        
        Args:
            image_data: Raw image bytes
            image_type: Type of chart/image
            context: Additional context about the image
            
        Returns:
            Analysis results or None if failed
        """
        try:
            # Prepare image
            processed_image = self._prepare_image_for_analysis(image_data)
            base64_image = self._encode_image(processed_image)
            
            # Get appropriate prompt
            prompt = self.chart_analysis_prompts.get(image_type, self.chart_analysis_prompts['price_chart'])
            
            if context:
                prompt += f"\n\nAdditional context: {context}"
            
            # Make API call via OpenRouter
            response = self.openai_client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    analysis_result = json.loads(json_str)
                else:
                    analysis_result = {
                        "analysis": content,
                        "confidence": 70,
                        "model": "claude-3.5-sonnet",
                        "raw_response": content
                    }
            except json.JSONDecodeError:
                analysis_result = {
                    "analysis": content,
                    "confidence": 50,
                    "model": "claude-3.5-sonnet",
                    "raw_response": content
                }
            
            logger.info("Claude analysis completed", image_type=image_type)
            return analysis_result
            
        except Exception as e:
            logger.error("Claude analysis failed", error=str(e))
            return None
    
    async def analyze_image_multimodal(self, image_data: bytes, image_type: str, ocr_text: str = "", context: str = "") -> Dict:
        """
        Comprehensive image analysis using multiple AI models.
        
        Args:
            image_data: Raw image bytes
            image_type: Type of chart/image
            ocr_text: Text extracted via OCR
            context: Additional context
            
        Returns:
            Combined analysis results
        """
        results = {
            'gpt4v_analysis': None,
            'claude_analysis': None,
            'combined_sentiment': 'neutral',
            'confidence': 0.0,
            'key_insights': [],
            'numerical_data': {},
            'trading_signals': []
        }
        
        # Add OCR context to prompt
        full_context = f"OCR extracted text: {ocr_text}\n" + context if ocr_text else context
        
        # Run both models in parallel via OpenRouter
        import asyncio
        gpt4o_task = self.analyze_with_gpt4v(image_data, image_type, full_context)  # Actually GPT-4O
        claude_task = self.analyze_with_claude(image_data, image_type, full_context)
        
        gpt4o_result, claude_result = await asyncio.gather(
            gpt4o_task, claude_task, return_exceptions=True
        )
        
        # Process GPT-4O results
        if isinstance(gpt4o_result, dict):
            results['gpt4v_analysis'] = gpt4o_result
        
        # Process Claude results
        if isinstance(claude_result, dict):
            results['claude_analysis'] = claude_result
        
        # Combine results
        results = self._combine_vision_results(results, image_type)
        
        logger.info(
            "Multimodal vision analysis completed",
            image_type=image_type,
            combined_sentiment=results['combined_sentiment'],
            confidence=results['confidence']
        )
        
        return results
    
    def _combine_vision_results(self, results: Dict, image_type: str) -> Dict:
        """
        Combine results from multiple vision models into a single analysis.
        
        Args:
            results: Dictionary with individual model results
            image_type: Type of image being analyzed
            
        Returns:
            Combined analysis results
        """
        gpt4v = results.get('gpt4v_analysis')
        claude = results.get('claude_analysis')
        
        # Initialize combined results
        combined_sentiment = 'neutral'
        combined_confidence = 0.0
        key_insights = []
        trading_signals = []
        
        sentiments = []
        confidences = []
        
        # Extract sentiment and confidence from GPT-4V
        if gpt4v:
            if 'sentiment' in gpt4v:
                sentiments.append(gpt4v['sentiment'])
            elif 'market_outlook' in gpt4v:
                sentiments.append(gpt4v['market_outlook'])
            elif 'trend_direction' in gpt4v:
                sentiments.append(gpt4v['trend_direction'])
            
            if 'confidence' in gpt4v:
                confidences.append(gpt4v['confidence'] / 100.0 if gpt4v['confidence'] > 1 else gpt4v['confidence'])
            
            # Extract insights
            if 'trading_signals' in gpt4v:
                if isinstance(gpt4v['trading_signals'], list):
                    trading_signals.extend(gpt4v['trading_signals'])
                else:
                    trading_signals.append(gpt4v['trading_signals'])
        
        # Extract sentiment and confidence from Claude
        if claude:
            if 'sentiment' in claude:
                sentiments.append(claude['sentiment'])
            elif 'market_outlook' in claude:
                sentiments.append(claude['market_outlook'])
            elif 'trend_direction' in claude:
                sentiments.append(claude['trend_direction'])
            
            if 'confidence' in claude:
                confidences.append(claude['confidence'] / 100.0 if claude['confidence'] > 1 else claude['confidence'])
            
            # Extract insights
            if 'trading_signals' in claude:
                if isinstance(claude['trading_signals'], list):
                    trading_signals.extend(claude['trading_signals'])
                else:
                    trading_signals.append(claude['trading_signals'])
        
        # Combine sentiments (majority vote or average)
        if sentiments:
            # Map sentiments to numerical values
            sentiment_mapping = {'bullish': 1, 'bearish': -1, 'neutral': 0}
            sentiment_values = [sentiment_mapping.get(s.lower(), 0) for s in sentiments]
            
            avg_sentiment = sum(sentiment_values) / len(sentiment_values)
            
            if avg_sentiment > 0.3:
                combined_sentiment = 'bullish'
            elif avg_sentiment < -0.3:
                combined_sentiment = 'bearish'
            else:
                combined_sentiment = 'neutral'
        
        # Average confidence
        if confidences:
            combined_confidence = sum(confidences) / len(confidences)
        
        # Combine insights
        for result in [gpt4v, claude]:
            if result and isinstance(result, dict):
                # Extract various types of insights
                insight_keys = ['key_levels', 'concentration_areas', 'breakeven_points', 'unusual_patterns']
                for key in insight_keys:
                    if key in result and result[key]:
                        if isinstance(result[key], list):
                            key_insights.extend(result[key])
                        else:
                            key_insights.append(str(result[key]))
        
        # Update results
        results['combined_sentiment'] = combined_sentiment
        results['confidence'] = combined_confidence
        results['key_insights'] = list(set(key_insights))  # Remove duplicates
        results['trading_signals'] = list(set(trading_signals))  # Remove duplicates
        
        return results


# Global vision AI analyzer instance
vision_ai = VisionAIAnalyzer()
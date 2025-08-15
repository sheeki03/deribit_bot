import asyncio
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path

from telegram import Bot, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError
import requests
from PIL import Image

from app.core.config import settings
from app.core.logging import logger
from app.ml.multimodal_scorer import MultimodalScoreComponents


class TelegramNotifier:
    """
    Intelligent Telegram notification system for Deribit Option Flows.
    
    Features:
    - Smart alert filtering based on confidence and score thresholds
    - Rich message formatting with emojis and markdown
    - Image previews of charts and analysis
    - Interactive buttons for quick actions
    - Rate limiting and spam prevention
    - Historical context and statistics
    """
    
    def __init__(self):
        # Handle optional Telegram configuration
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self.bot = Bot(token=settings.telegram_bot_token)
            self.chat_id = settings.telegram_chat_id
            self.enabled = True
        else:
            self.bot = None
            self.chat_id = None
            self.enabled = False
            logger.info("Telegram notifications disabled (no credentials provided)")
        
        # Alert configuration
        self.alert_config = {
            'extreme_threshold': 0.5,      # ±0.5 for extreme alerts
            'significant_threshold': 0.3,   # ±0.3 for significant alerts
            'min_confidence': 0.7,          # Minimum confidence for alerts
            'max_alerts_per_hour': 10,      # Rate limiting
            'cooldown_minutes': 5           # Minimum time between similar alerts
        }
        
        # Message templates
        self.templates = {
            'extreme': {
                'emoji': '🚨',
                'title': 'EXTREME FLOW ALERT',
                'color': 'red'
            },
            'significant': {
                'emoji': '⚡',
                'title': 'Significant Flow Alert',
                'color': 'orange'
            },
            'info': {
                'emoji': 'ℹ️',
                'title': 'Info',
                'color': 'blue'
            }
        }
        
        # Rate limiting
        self.recent_alerts = []
        self.last_alert_times = {}
    
    async def send_flowscore_alert(self, 
                                  article_data: Dict,
                                  asset_scores: Dict[str, MultimodalScoreComponents],
                                  historical_context: Optional[Dict] = None) -> bool:
        """
        Send FlowScore alert with comprehensive analysis.
        
        Args:
            article_data: Article information
            asset_scores: FlowScore components for each asset
            historical_context: Historical performance context
            
        Returns:
            True if alert was sent successfully
        """
        try:
            # Check if Telegram is enabled
            if not self.enabled:
                logger.debug("Telegram notifications disabled, skipping alert")
                return False
                
            # Check if alert should be sent
            if not self._should_send_alert(asset_scores):
                return False
            
            # Determine alert level
            alert_level = self._determine_alert_level(asset_scores)
            
            # Format main message
            message_text = self._format_alert_message(
                article_data, asset_scores, alert_level, historical_context
            )
            
            # Prepare inline keyboard
            keyboard = self._create_inline_keyboard(article_data, asset_scores)
            
            # Get relevant images
            images = self._select_alert_images(article_data.get('images', []))
            
            # Send message with images if available
            if images and len(images) <= 3:  # Telegram limit for media groups
                await self._send_message_with_images(
                    message_text, images, keyboard
                )
            else:
                await self._send_text_message(message_text, keyboard)
            
            # Update rate limiting
            self._update_alert_tracking(asset_scores)
            
            logger.info(f"FlowScore alert sent successfully: {alert_level}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def _should_send_alert(self, asset_scores: Dict[str, MultimodalScoreComponents]) -> bool:
        """Determine if alert should be sent based on thresholds and rate limiting."""
        
        # Check if any score meets alert criteria
        alert_worthy = False
        
        for asset, score_components in asset_scores.items():
            score = score_components.final_score
            confidence = score_components.overall_confidence
            
            if (confidence >= self.alert_config['min_confidence'] and
                abs(score) >= self.alert_config['significant_threshold']):
                alert_worthy = True
                break
        
        if not alert_worthy:
            return False
        
        # Rate limiting check
        current_time = datetime.now()
        hour_ago = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Count recent alerts in the last hour
        recent_count = len([
            alert_time for alert_time in self.recent_alerts
            if alert_time >= hour_ago
        ])
        
        if recent_count >= self.alert_config['max_alerts_per_hour']:
            logger.warning("Alert rate limit exceeded")
            return False
        
        # Check cooldown for similar alerts
        alert_key = self._generate_alert_key(asset_scores)
        last_similar = self.last_alert_times.get(alert_key)
        
        if last_similar:
            minutes_since = (current_time - last_similar).total_seconds() / 60
            if minutes_since < self.alert_config['cooldown_minutes']:
                logger.info(f"Alert cooldown active: {minutes_since:.1f}min")
                return False
        
        return True
    
    def _determine_alert_level(self, asset_scores: Dict[str, MultimodalScoreComponents]) -> str:
        """Determine alert level based on scores."""
        max_abs_score = 0
        max_confidence = 0
        
        for score_components in asset_scores.values():
            max_abs_score = max(max_abs_score, abs(score_components.final_score))
            max_confidence = max(max_confidence, score_components.overall_confidence)
        
        if (max_abs_score >= self.alert_config['extreme_threshold'] and
            max_confidence >= 0.8):
            return 'extreme'
        elif max_abs_score >= self.alert_config['significant_threshold']:
            return 'significant'
        else:
            return 'info'
    
    def _format_alert_message(self, 
                            article_data: Dict,
                            asset_scores: Dict[str, MultimodalScoreComponents],
                            alert_level: str,
                            historical_context: Optional[Dict] = None) -> str:
        """Format comprehensive alert message."""
        
        template = self.templates[alert_level]
        
        # Header
        message = f"{template['emoji']} **{template['title']}**\n\n"
        
        # Article info
        title = article_data.get('title', 'Unknown Article')[:60]
        author = article_data.get('author', 'Unknown')
        pub_time = article_data.get('published_at_utc', datetime.now())
        
        if isinstance(pub_time, str):
            pub_time = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
        
        message += f"📰 **{title}**\n"
        message += f"👤 {author} | 🕒 {pub_time.strftime('%H:%M UTC')}\n\n"
        
        # FlowScores for each asset
        message += "📊 **FlowScores:**\n"
        
        for asset, score_components in asset_scores.items():
            score = score_components.final_score
            confidence = score_components.overall_confidence
            
            # Determine sentiment emoji
            if score > 0.3:
                sentiment_emoji = "🟢"
                sentiment_text = "BULLISH"
            elif score < -0.3:
                sentiment_emoji = "🔴"
                sentiment_text = "BEARISH"
            else:
                sentiment_emoji = "⚪"
                sentiment_text = "NEUTRAL"
            
            # Confidence indicator
            if confidence >= 0.8:
                conf_emoji = "🟢"
            elif confidence >= 0.6:
                conf_emoji = "🟡"
            else:
                conf_emoji = "🔴"
            
            message += f"{sentiment_emoji} **{asset}**: `{score:+.3f}` ({sentiment_text})\n"
            message += f"   {conf_emoji} Confidence: `{confidence:.1%}`\n"
        
        # Component breakdown (for extreme alerts)
        if alert_level == 'extreme':
            message += "\n🔧 **Component Breakdown:**\n"
            
            # Take the highest scoring asset for breakdown
            top_asset = max(asset_scores.items(), key=lambda x: abs(x[1].final_score))
            asset_name, components = top_asset
            
            message += f"• XGBoost: `{components.xgboost_score:+.3f}` ({components.xgboost_confidence:.1%})\n"
            message += f"• FinBERT: `{components.finbert_score:+.3f}` ({components.finbert_confidence:.1%})\n"
            message += f"• Vision AI: `{components.vision_score:+.3f}` ({components.vision_confidence:.1%})\n"
        
        # Key signals
        all_signals = []
        for components in asset_scores.values():
            all_signals.extend(components.signals[:3])  # Top 3 signals per asset
        
        if all_signals:
            message += f"\n🎯 **Key Signals:**\n"
            for signal in all_signals[:5]:  # Limit to 5 total signals
                message += f"• {signal}\n"
        
        # Historical context
        if historical_context:
            message += f"\n📈 **Historical Context:**\n"
            
            similar_score_returns = historical_context.get('similar_score_returns', {})
            if similar_score_returns:
                ret_24h = similar_score_returns.get('mean_24h_return', 0)
                hit_rate = similar_score_returns.get('hit_rate', 0)
                sample_size = similar_score_returns.get('sample_size', 0)
                
                message += f"• Similar scores (n={sample_size}):\n"
                message += f"  - Avg 24h return: `{ret_24h:+.2%}`\n"
                message += f"  - Hit rate: `{hit_rate:.1%}`\n"
        
        # Footer
        message += f"\n🔗 [Read Article]({article_data.get('url', '')})"
        message += f"\n⏰ Alert sent at {datetime.now().strftime('%H:%M:%S UTC')}"
        
        return message
    
    def _create_inline_keyboard(self, 
                              article_data: Dict,
                              asset_scores: Dict[str, MultimodalScoreComponents]) -> InlineKeyboardMarkup:
        """Create inline keyboard with quick action buttons."""
        
        buttons = []
        
        # Quick action buttons
        row1 = [
            InlineKeyboardButton("📊 Dashboard", url="http://localhost:8501"),
            InlineKeyboardButton("📰 Article", url=article_data.get('url', ''))
        ]
        buttons.append(row1)
        
        # Asset-specific buttons
        row2 = []
        for asset in asset_scores.keys():
            callback_data = f"asset_{asset}_{article_data.get('url', '')[:20]}"
            row2.append(
                InlineKeyboardButton(f"{asset} Details", callback_data=callback_data)
            )
        buttons.append(row2)
        
        # Action buttons
        row3 = [
            InlineKeyboardButton("🔕 Mute 1h", callback_data="mute_1h"),
            InlineKeyboardButton("⚙️ Settings", callback_data="settings")
        ]
        buttons.append(row3)
        
        return InlineKeyboardMarkup(buttons)
    
    def _select_alert_images(self, images: List[Dict]) -> List[Dict]:
        """Select most relevant images for the alert."""
        if not images:
            return []
        
        # Priority order for image types
        priority_types = [
            'greeks_chart',
            'flow_heatmap', 
            'skew_chart',
            'price_chart',
            'position_diagram'
        ]
        
        selected_images = []
        
        # Select highest priority images
        for img_type in priority_types:
            matching_images = [
                img for img in images 
                if img.get('image_type') == img_type
                and img.get('processing_status') == 'completed'
                and img.get('confidence_score', 0) > 0.3
            ]
            
            if matching_images:
                # Sort by confidence and take the best one
                best_image = max(matching_images, key=lambda x: x.get('confidence_score', 0))
                selected_images.append(best_image)
                
                if len(selected_images) >= 3:  # Telegram media group limit
                    break
        
        return selected_images
    
    async def _send_message_with_images(self, 
                                       message_text: str,
                                       images: List[Dict],
                                       keyboard: InlineKeyboardMarkup) -> None:
        """Send message with image media group."""
        
        try:
            media_group = []
            
            for i, image_data in enumerate(images):
                image_path = image_data.get('download_path')
                image_type = image_data.get('image_type', 'unknown')
                confidence = image_data.get('confidence_score', 0)
                
                if not image_path or not Path(image_path).exists():
                    continue
                
                # Prepare image for sending
                with open(image_path, 'rb') as img_file:
                    # Create caption for first image
                    caption = None
                    if i == 0:
                        caption = message_text[:1024]  # Telegram caption limit
                    
                    # Add image to media group
                    media_group.append(
                        InputMediaPhoto(
                            media=img_file.read(),
                            caption=caption,
                            parse_mode='Markdown'
                        )
                    )
            
            if media_group:
                # Send media group
                await self.bot.send_media_group(
                    chat_id=self.chat_id,
                    media=media_group
                )
                
                # Send keyboard separately (media groups don't support keyboards)
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text="Quick Actions:",
                    reply_markup=keyboard
                )
            else:
                # Fallback to text message
                await self._send_text_message(message_text, keyboard)
                
        except Exception as e:
            logger.error(f"Failed to send message with images: {e}")
            # Fallback to text message
            await self._send_text_message(message_text, keyboard)
    
    async def _send_text_message(self, 
                               message_text: str,
                               keyboard: InlineKeyboardMarkup) -> None:
        """Send text-only message."""
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message_text,
                parse_mode='Markdown',
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            
        except TelegramError as e:
            # Try without markdown if formatting fails
            logger.warning(f"Markdown formatting failed, sending plain text: {e}")
            
            plain_text = message_text.replace('*', '').replace('`', '').replace('_', '')
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=plain_text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
    
    async def send_system_notification(self, 
                                     message: str,
                                     level: str = 'info',
                                     include_stats: bool = False) -> bool:
        """Send system status notification."""
        
        # Check if Telegram is enabled
        if not self.enabled or self.bot is None:
            logger.info("Telegram notifications disabled, skipping system notification")
            return False
            
        try:
            template = self.templates.get(level, self.templates['info'])
            
            formatted_message = f"{template['emoji']} **System Notification**\n\n"
            formatted_message += message
            
            if include_stats:
                # Add basic system stats
                formatted_message += f"\n\n📊 **System Stats:**\n"
                formatted_message += f"• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                formatted_message += f"• Alerts sent today: {len(self.recent_alerts)}\n"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=formatted_message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send system notification: {e}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send daily summary of FlowScore activity."""
        
        # Check if Telegram is enabled
        if not self.enabled or self.bot is None:
            logger.info("Telegram notifications disabled, skipping daily summary")
            return False
            
        try:
            message = "📊 **Daily Option Flows Summary**\n\n"
            
            # Basic stats
            total_articles = summary_data.get('total_articles', 0)
            avg_btc_score = summary_data.get('avg_btc_score', 0)
            avg_eth_score = summary_data.get('avg_eth_score', 0)
            
            message += f"📰 Articles analyzed: {total_articles}\n"
            message += f"₿ BTC avg score: `{avg_btc_score:+.3f}`\n"
            message += f"Ξ ETH avg score: `{avg_eth_score:+.3f}`\n\n"
            
            # Top articles
            top_articles = summary_data.get('top_articles', [])
            if top_articles:
                message += "🏆 **Top FlowScores Today:**\n"
                for article in top_articles[:3]:
                    title = article['title'][:40]
                    score = article['max_score']
                    asset = article['asset']
                    
                    sentiment_emoji = "🟢" if score > 0 else "🔴"
                    message += f"{sentiment_emoji} {asset}: `{score:+.3f}` - {title}\n"
            
            # Performance stats
            performance = summary_data.get('performance', {})
            if performance:
                message += f"\n📈 **Performance:**\n"
                message += f"• Hit rate: {performance.get('hit_rate', 0):.1%}\n"
                message += f"• Avg confidence: {performance.get('avg_confidence', 0):.1%}\n"
            
            message += f"\n⏰ Summary for {datetime.now().strftime('%Y-%m-%d')}"
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    def _generate_alert_key(self, asset_scores: Dict[str, MultimodalScoreComponents]) -> str:
        """Generate key for alert cooldown tracking."""
        # Create a key based on score ranges to group similar alerts
        key_parts = []
        
        for asset, components in asset_scores.items():
            score = components.final_score
            score_range = round(score, 1)  # Round to nearest 0.1
            key_parts.append(f"{asset}_{score_range}")
        
        return "_".join(sorted(key_parts))
    
    def _update_alert_tracking(self, asset_scores: Dict[str, MultimodalScoreComponents]):
        """Update alert tracking for rate limiting."""
        current_time = datetime.now()
        
        # Add to recent alerts list
        self.recent_alerts.append(current_time)
        
        # Clean old alerts (keep only last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.recent_alerts = [
            alert_time for alert_time in self.recent_alerts
            if alert_time >= cutoff_time
        ]
        
        # Update cooldown tracking
        alert_key = self._generate_alert_key(asset_scores)
        self.last_alert_times[alert_key] = current_time
        
        # Clean old cooldown entries (keep only last hour)
        cutoff_cooldown = current_time - timedelta(hours=1)
        self.last_alert_times = {
            key: time for key, time in self.last_alert_times.items()
            if time >= cutoff_cooldown
        }
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        # Check if Telegram is enabled
        if not self.enabled or self.bot is None:
            logger.info("Telegram notifications disabled, skipping connection test")
            return False
            
        try:
            bot_info = await self.bot.get_me()
            logger.info(f"Telegram bot connected: @{bot_info.username}")
            
            # Send test message
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🤖 Deribit Option Flows bot is online and ready!"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)
        day_ago = current_time - timedelta(hours=24)
        
        alerts_last_hour = len([
            alert for alert in self.recent_alerts if alert >= hour_ago
        ])
        
        alerts_last_day = len([
            alert for alert in self.recent_alerts if alert >= day_ago
        ])
        
        return {
            'alerts_last_hour': alerts_last_hour,
            'alerts_last_day': alerts_last_day,
            'max_alerts_per_hour': self.alert_config['max_alerts_per_hour'],
            'active_cooldowns': len(self.last_alert_times),
            'alert_thresholds': {
                'extreme': self.alert_config['extreme_threshold'],
                'significant': self.alert_config['significant_threshold'],
                'min_confidence': self.alert_config['min_confidence']
            }
        }


# Global Telegram notifier instance
telegram_notifier = TelegramNotifier()
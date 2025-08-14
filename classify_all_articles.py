#!/usr/bin/env python3

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class ClassifiedArticle:
    """Classified article with enhanced metadata."""
    # Original data
    title: str
    url: str
    publication_date: str
    readable_date: str
    author: str
    content: str
    images: List[Dict[str, str]]
    extraction_confidence: float
    scrape_timestamp: str
    
    # New classification fields
    classification: Dict[str, Any]
    content_analysis: Dict[str, Any]
    market_context: Dict[str, Any]
    trading_signals: Dict[str, Any]

class ArticleClassifier:
    """Comprehensive article classification system."""
    
    def __init__(self):
        self.classification_rules = self._setup_classification_rules()
    
    def _setup_classification_rules(self):
        """Setup classification rules and patterns."""
        return {
            'themes': {
                'btc_focus': [r'btc', r'bitcoin', r'1\.2b btc', r'btc etf', r'btc halving', r'btc spot'],
                'eth_focus': [r'eth\b', r'ethereum', r'eth june', r'eth rotation', r'eth unlock', r'eth saw etf'],
                'volatility': [r'vol\b', r'volatility', r'iv', r'skew', r'gamma', r'vega'],
                'options_strategy': [r'call', r'put', r'spread', r'straddle', r'strangle', r'collar'],
                'market_structure': [r'flow', r'liquidity', r'order book', r'market maker', r'dealer'],
                'macro_events': [r'fomc', r'etf', r'halving', r'macro', r'fed', r'regulation'],
                'sentiment': [r'bullish', r'bearish', r'fear', r'greed', r'panic', r'euphoria'],
                'technical': [r'resistance', r'support', r'breakout', r'bounce', r'trend']
            },
            'trading_actions': {
                'buying': [r'buying', r'accumulation', r'long', r'bid', r'uptick'],
                'selling': [r'selling', r'distribution', r'short', r'ask', r'downtick', r'dump'],
                'rolling': [r'roll', r'rolling', r'extend', r'close', r'unwind'],
                'hedging': [r'hedge', r'protect', r'insurance', r'defensive'],
                'speculative': [r'bet', r'speculation', r'gamble', r'lottery', r'yolo']
            },
            'market_conditions': {
                'high_vol': [r'storm', r'volatile', r'chaotic', r'wild', r'explosive'],
                'low_vol': [r'quiet', r'calm', r'stable', r'range', r'sideways'],
                'trending': [r'momentum', r'trend', r'direction', r'move', r'breakout'],
                'uncertain': [r'mixed', r'unclear', r'waiting', r'observing', r'dilemma']
            }
        }
    
    def classify_article(self, article: Dict[str, Any]) -> ClassifiedArticle:
        """Classify a single article with comprehensive analysis."""
        
        # Extract basic info
        title = article['title']
        content = article['content']
        date = article['publication_date']
        
        # Combine title and content for analysis
        full_text = f"{title} {content}".lower()
        
        # Classification analysis
        classification = self._analyze_classification(full_text, title)
        
        # Content analysis
        content_analysis = self._analyze_content(full_text, content)
        
        # Market context
        market_context = self._analyze_market_context(date, title, full_text)
        
        # Trading signals
        trading_signals = self._analyze_trading_signals(full_text, title)
        
        return ClassifiedArticle(
            # Original fields
            title=article['title'],
            url=article['url'],
            publication_date=article['publication_date'],
            readable_date=article['readable_date'],
            author=article['author'],
            content=article['content'],
            images=article['images'],
            extraction_confidence=article['extraction_confidence'],
            scrape_timestamp=article['scrape_timestamp'],
            
            # New classification fields
            classification=classification,
            content_analysis=content_analysis,
            market_context=market_context,
            trading_signals=trading_signals
        )
    
    def _analyze_classification(self, full_text: str, title: str) -> Dict[str, Any]:
        """Analyze article classification themes."""
        
        themes = {}
        for theme, patterns in self.classification_rules['themes'].items():
            score = sum(1 for pattern in patterns if re.search(pattern, full_text))
            themes[theme] = score
        
        # Determine primary theme
        primary_theme = max(themes.items(), key=lambda x: x[1])[0] if themes else 'general'
        
        # Extract specific assets mentioned
        assets = []
        if re.search(r'btc|bitcoin', full_text):
            assets.append('BTC')
        if re.search(r'eth\b|ethereum', full_text):
            assets.append('ETH')
        if re.search(r'gold', full_text):
            assets.append('GOLD')
            
        return {
            'primary_theme': primary_theme,
            'theme_scores': themes,
            'assets_mentioned': assets,
            'article_type': self._determine_article_type(title),
            'complexity_score': min(len(full_text.split()) / 100, 10)  # 1-10 scale
        }
    
    def _determine_article_type(self, title: str) -> str:
        """Determine the type of article based on title pattern."""
        title_lower = title.lower()
        
        if re.search(r'week \d+', title_lower):
            return 'weekly_summary'
        elif '?' in title:
            return 'analysis_question'
        elif re.search(r'big|huge|massive|\$\d+[mb]', title_lower):
            return 'large_flow'
        elif re.search(r'etf|halving|macro|fomc', title_lower):
            return 'macro_event'
        else:
            return 'flow_analysis'
    
    def _analyze_content(self, full_text: str, content: str) -> Dict[str, Any]:
        """Analyze content characteristics."""
        
        words = content.split()
        sentences = content.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'readability_score': self._calculate_readability(words, sentences),
            'technical_density': self._calculate_technical_density(full_text),
            'sentiment_indicators': self._extract_sentiment_indicators(full_text)
        }
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """Simple readability score (1-10 scale)."""
        if not sentences:
            return 5.0
            
        avg_sentence_length = len(words) / len(sentences)
        # Normalize to 1-10 scale (higher = more complex)
        return min(avg_sentence_length / 3, 10)
    
    def _calculate_technical_density(self, full_text: str) -> float:
        """Calculate density of technical options terms."""
        technical_terms = [
            'gamma', 'delta', 'theta', 'vega', 'implied volatility', 'iv',
            'skew', 'term structure', 'open interest', 'volume',
            'strike', 'expiration', 'premium', 'intrinsic', 'time value'
        ]
        
        term_count = sum(1 for term in technical_terms if term in full_text)
        total_words = len(full_text.split())
        
        return (term_count / max(total_words / 100, 1)) * 10  # Normalize to 0-10
    
    def _extract_sentiment_indicators(self, full_text: str) -> Dict[str, int]:
        """Extract sentiment indicators from text."""
        
        sentiment_patterns = {
            'bullish': [r'bullish', r'optimistic', r'positive', r'upside', r'buying', r'accumulation'],
            'bearish': [r'bearish', r'pessimistic', r'negative', r'downside', r'selling', r'distribution'],
            'uncertain': [r'uncertain', r'mixed', r'unclear', r'waiting', r'observing', r'cautious'],
            'volatile': [r'volatile', r'chaotic', r'wild', r'explosive', r'storm', r'turbulent']
        }
        
        sentiments = {}
        for sentiment, patterns in sentiment_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, full_text))
            sentiments[sentiment] = score
            
        return sentiments
    
    def _analyze_market_context(self, date: str, title: str, full_text: str) -> Dict[str, Any]:
        """Analyze market context for the article date."""
        
        # Parse date
        article_date = datetime.strptime(date, '%Y-%m-%d')
        
        # Determine market period
        market_period = self._determine_market_period(article_date)
        
        # Extract price levels mentioned
        price_levels = self._extract_price_levels(full_text)
        
        # Analyze time sensitivity
        time_sensitivity = self._analyze_time_sensitivity(title, full_text)
        
        return {
            'market_period': market_period,
            'year': article_date.year,
            'quarter': f'Q{(article_date.month - 1) // 3 + 1}',
            'month': article_date.strftime('%B'),
            'price_levels_mentioned': price_levels,
            'time_sensitivity': time_sensitivity,
            'event_driven': self._is_event_driven(full_text)
        }
    
    def _determine_market_period(self, date: datetime) -> str:
        """Determine which market period the article belongs to."""
        
        if date.year == 2022:
            return 'bear_market_2022'
        elif date.year == 2023:
            if date.month <= 6:
                return 'recovery_early_2023'
            else:
                return 'bull_run_mid_2023'
        elif date.year == 2024:
            if date.month <= 3:
                return 'etf_approval_period'
            elif date.month <= 6:
                return 'halving_period_2024'
            else:
                return 'post_halving_2024'
        elif date.year == 2025:
            return 'bull_continuation_2025'
        else:
            return 'unknown_period'
    
    def _extract_price_levels(self, full_text: str) -> List[str]:
        """Extract specific price levels mentioned."""
        
        # Look for price patterns like $50k, 60000, etc.
        price_patterns = [
            r'\$\d+[km]?',  # $50k, $60000
            r'\d+k',        # 50k
            r'\d{4,5}'      # 50000, 60000
        ]
        
        prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, full_text)
            prices.extend(matches)
            
        return list(set(prices))[:10]  # Max 10 unique prices
    
    def _analyze_time_sensitivity(self, title: str, full_text: str) -> str:
        """Analyze time sensitivity of the article."""
        
        urgent_keywords = ['deadline', 'approaching', 'immediate', 'now', 'urgent', 'breaking']
        time_bound_keywords = ['week', 'expiry', 'expiration', 'fomc', 'event']
        
        if any(keyword in full_text.lower() for keyword in urgent_keywords):
            return 'high'
        elif any(keyword in full_text.lower() for keyword in time_bound_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _is_event_driven(self, full_text: str) -> bool:
        """Check if article is driven by specific events."""
        
        event_keywords = [
            'etf', 'halving', 'fomc', 'fed', 'regulation', 'earnings',
            'approval', 'launch', 'announcement', 'decision'
        ]
        
        return any(keyword in full_text.lower() for keyword in event_keywords)
    
    def _analyze_trading_signals(self, full_text: str, title: str) -> Dict[str, Any]:
        """Analyze trading signals and actionable insights."""
        
        # Analyze trading actions
        actions = {}
        for action, patterns in self.classification_rules['trading_actions'].items():
            score = sum(1 for pattern in patterns if re.search(pattern, full_text))
            actions[action] = score
        
        # Determine primary action
        primary_action = max(actions.items(), key=lambda x: x[1])[0] if any(actions.values()) else 'neutral'
        
        # Extract specific strikes/expirations mentioned
        strikes = self._extract_strikes(full_text)
        expirations = self._extract_expirations(full_text)
        
        # Analyze signal strength
        signal_strength = self._calculate_signal_strength(title, full_text, actions)
        
        return {
            'primary_action': primary_action,
            'action_scores': actions,
            'signal_strength': signal_strength,
            'strikes_mentioned': strikes,
            'expirations_mentioned': expirations,
            'directional_bias': self._determine_directional_bias(full_text),
            'risk_level': self._assess_risk_level(full_text)
        }
    
    def _extract_strikes(self, full_text: str) -> List[str]:
        """Extract option strikes mentioned."""
        
        strike_patterns = [
            r'\d+k strike',
            r'\$\d+k',
            r'\d+,\d+',
            r'strike \d+'
        ]
        
        strikes = []
        for pattern in strike_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            strikes.extend(matches)
            
        return list(set(strikes))[:5]  # Max 5 unique strikes
    
    def _extract_expirations(self, full_text: str) -> List[str]:
        """Extract option expirations mentioned."""
        
        expiration_patterns = [
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}',
            r'\d+d\s+(expiry|expiration)',
            r'week\s+\d+'
        ]
        
        expirations = []
        for pattern in expiration_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], str):
                    expirations.extend(matches)
                else:
                    expirations.extend([' '.join(m) if isinstance(m, tuple) else str(m) for m in matches])
            
        return list(set(expirations))[:5]  # Max 5 unique expirations
    
    def _calculate_signal_strength(self, title: str, full_text: str, actions: Dict[str, int]) -> float:
        """Calculate trading signal strength (0-1 scale)."""
        
        # Base strength on action intensity
        max_action_score = max(actions.values()) if actions.values() else 0
        base_strength = min(max_action_score / 5, 1.0)
        
        # Boost for strong language in title
        strong_words = ['big', 'massive', 'huge', 'explosive', 'breaking', 'urgent']
        title_boost = 0.2 if any(word in title.lower() for word in strong_words) else 0
        
        # Boost for specific numbers/amounts
        number_boost = 0.1 if re.search(r'\$\d+[mb]|\d+%', full_text) else 0
        
        return min(base_strength + title_boost + number_boost, 1.0)
    
    def _determine_directional_bias(self, full_text: str) -> str:
        """Determine overall directional bias."""
        
        bullish_score = len(re.findall(r'bullish|upside|long|buying|accumulation|calls', full_text, re.IGNORECASE))
        bearish_score = len(re.findall(r'bearish|downside|short|selling|distribution|puts', full_text, re.IGNORECASE))
        
        if bullish_score > bearish_score * 1.5:
            return 'bullish'
        elif bearish_score > bullish_score * 1.5:
            return 'bearish'
        else:
            return 'neutral'
    
    def _assess_risk_level(self, full_text: str) -> str:
        """Assess risk level of strategies discussed."""
        
        high_risk_terms = ['speculation', 'gambling', 'yolo', 'lottery', 'risky', 'dangerous']
        medium_risk_terms = ['aggressive', 'leveraged', 'momentum', 'breakout']
        low_risk_terms = ['hedge', 'defensive', 'conservative', 'safe', 'protection']
        
        high_score = sum(1 for term in high_risk_terms if term in full_text.lower())
        medium_score = sum(1 for term in medium_risk_terms if term in full_text.lower())
        low_score = sum(1 for term in low_risk_terms if term in full_text.lower())
        
        if high_score > 0:
            return 'high'
        elif medium_score > low_score:
            return 'medium'
        else:
            return 'low'

def classify_all_articles():
    """Classify all articles in the complete dataset."""
    
    print('🔄 CLASSIFYING ALL 126 ARTICLES')
    print('=' * 60)
    
    # Load complete dataset
    with open('scraped_data/playwright/complete_articles_dataset.json', 'r') as f:
        data = json.load(f)
    
    articles = data['articles']
    print(f'Processing {len(articles)} articles...')
    
    # Initialize classifier
    classifier = ArticleClassifier()
    
    # Classify all articles
    classified_articles = []
    for i, article in enumerate(articles, 1):
        if i % 10 == 0 or i == len(articles):
            print(f'  Progress: {i}/{len(articles)} articles classified')
        
        classified_article = classifier.classify_article(article)
        classified_articles.append(classified_article)
    
    # Convert to dictionaries for JSON serialization
    classified_data = [asdict(article) for article in classified_articles]
    
    # Create comprehensive classified dataset
    output_data = {
        'classification_metadata': {
            'classification_timestamp': datetime.now().isoformat(),
            'total_articles': len(classified_articles),
            'classification_version': '1.0',
            'date_range': data['scrape_metadata']['date_range']
        },
        'classified_articles': classified_data
    }
    
    # Save classified dataset
    output_path = Path('scraped_data/playwright/classified_articles_complete.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'\n✅ Classification complete!')
    print(f'📁 Saved to: {output_path}')
    
    # Generate classification summary
    summary = generate_classification_summary(classified_articles)
    print_classification_summary(summary)
    
    return classified_articles, summary

def generate_classification_summary(classified_articles: List[ClassifiedArticle]) -> Dict[str, Any]:
    """Generate comprehensive classification summary."""
    
    # Theme distribution
    theme_counts = defaultdict(int)
    asset_counts = defaultdict(int)
    article_type_counts = defaultdict(int)
    
    for article in classified_articles:
        theme_counts[article.classification['primary_theme']] += 1
        article_type_counts[article.classification['article_type']] += 1
        for asset in article.classification['assets_mentioned']:
            asset_counts[asset] += 1
    
    # Trading action distribution
    action_counts = defaultdict(int)
    directional_bias_counts = defaultdict(int)
    risk_level_counts = defaultdict(int)
    
    for article in classified_articles:
        action_counts[article.trading_signals['primary_action']] += 1
        directional_bias_counts[article.trading_signals['directional_bias']] += 1
        risk_level_counts[article.trading_signals['risk_level']] += 1
    
    # Market period distribution
    period_counts = defaultdict(int)
    for article in classified_articles:
        period_counts[article.market_context['market_period']] += 1
    
    return {
        'theme_distribution': dict(theme_counts),
        'asset_distribution': dict(asset_counts),
        'article_type_distribution': dict(article_type_counts),
        'trading_action_distribution': dict(action_counts),
        'directional_bias_distribution': dict(directional_bias_counts),
        'risk_level_distribution': dict(risk_level_counts),
        'market_period_distribution': dict(period_counts)
    }

def print_classification_summary(summary: Dict[str, Any]):
    """Print classification summary statistics."""
    
    print(f'\n📊 CLASSIFICATION SUMMARY')
    print('=' * 40)
    
    print(f'\n🎯 Theme Distribution:')
    for theme, count in sorted(summary['theme_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f'  {theme}: {count}')
    
    print(f'\n💰 Asset Coverage:')
    for asset, count in sorted(summary['asset_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f'  {asset}: {count} articles')
    
    print(f'\n📈 Trading Actions:')
    for action, count in sorted(summary['trading_action_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f'  {action}: {count}')
    
    print(f'\n📅 Market Periods:')
    for period, count in sorted(summary['market_period_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f'  {period}: {count}')

if __name__ == "__main__":
    classified_articles, summary = classify_all_articles()
    print(f'\n🎉 CLASSIFICATION COMPLETE!')
    print(f'📊 {len(classified_articles)} articles fully classified and organized by names and dates')
    print(f'🎯 Ready for comprehensive options analysis with price correlation!')
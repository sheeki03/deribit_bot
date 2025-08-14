"""
Data Processing Engine for Options Analysis Dashboard
Handles loading, caching, and preprocessing of unified dataset.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import streamlit as st

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data loading and preprocessing for the dashboard."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize data processor with base path."""
        self.base_path = base_path or Path(__file__).parent.parent.parent
        
        # Data containers
        self.articles: List[Dict[str, Any]] = []
        self.articles_df: pd.DataFrame = pd.DataFrame()
        self.price_data: pd.DataFrame = pd.DataFrame()
        self.unified_data: Dict[str, Any] = {}
        
        # Cached processed data
        self._weekly_data: Optional[pd.DataFrame] = None
        self._monthly_data: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        
        # Date range info
        self.date_range: Dict[str, str] = {}
        
    @st.cache_data
    def load_all_data(_self):
        """Load all required data sources with caching."""
        logger.info("Loading unified dataset...")
        
        try:
            # Load unified articles
            _self._load_unified_articles()
            
            # Load price data
            _self._load_price_data()
            
            # Set date range
            _self._set_date_range()
            
            # Create processed datasets
            _self._create_processed_datasets()
            
            logger.info(f"Data loaded successfully: {len(_self.articles)} articles, {len(_self.price_data)} price records")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _load_unified_articles(self):
        """Load the unified articles dataset."""
        unified_path = self.base_path / 'scraped_data' / 'playwright' / 'unified_articles_complete.json'
        
        if not unified_path.exists():
            raise FileNotFoundError(f"Unified articles file not found: {unified_path}")
        
        with open(unified_path, 'r', encoding='utf-8') as f:
            self.unified_data = json.load(f)
        
        self.articles = self.unified_data['unified_articles']
        
        # Convert articles to DataFrame for easier processing  
        self.articles_df = pd.DataFrame([
            {
                'title': article.get('title', ''),
                'date': pd.to_datetime(article.get('publication_date', '2022-01-01')),
                'readable_date': article.get('readable_date', ''),
                'slug': article.get('slug', ''),
                'primary_theme': article.get('classification', {}).get('primary_theme', ''),
                'article_type': article.get('classification', {}).get('article_type', ''),
                'assets_mentioned': article.get('classification', {}).get('assets_mentioned', []),
                'complexity_score': article.get('classification', {}).get('complexity_score', 0.0),
                'primary_action': article.get('trading_signals', {}).get('primary_action', ''),
                'signal_strength': article.get('trading_signals', {}).get('signal_strength', 0.0),
                'directional_bias': article.get('trading_signals', {}).get('directional_bias', ''),
                'risk_level': article.get('trading_signals', {}).get('risk_level', ''),
                'market_period': article.get('market_context', {}).get('market_period', ''),
                'time_sensitivity': article.get('market_context', {}).get('time_sensitivity', ''),
                'event_driven': article.get('market_context', {}).get('event_driven', False),
                'extraction_confidence': article.get('extraction_confidence', 0.0),
                'word_count': article.get('content_analysis', {}).get('word_count', 0),
                'technical_density': article.get('content_analysis', {}).get('technical_density', 0.0),
                'sentiment_bullish': article.get('content_analysis', {}).get('sentiment_indicators', {}).get('bullish', 0),
                'sentiment_bearish': article.get('content_analysis', {}).get('sentiment_indicators', {}).get('bearish', 0),
                'sentiment_uncertain': article.get('content_analysis', {}).get('sentiment_indicators', {}).get('uncertain', 0),
                'sentiment_volatile': article.get('content_analysis', {}).get('sentiment_indicators', {}).get('volatile', 0),
                'num_images': len(article.get('analyzed_images', [])),
                'has_body_text': bool(article.get('body_text')),
                'data_sources': article.get('data_sources', [])
            }
            for article in self.articles
        ])
        
        logger.info(f"Loaded {len(self.articles)} unified articles")
    
    def _load_price_data(self):
        """Load the combined price data."""
        price_path = self.base_path / 'data' / 'price_data' / 'combined_daily_prices.csv'
        
        if not price_path.exists():
            raise FileNotFoundError(f"Price data file not found: {price_path}")
        
        # Load price data
        self.price_data = pd.read_csv(price_path)
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        
        # Sort by date and asset
        self.price_data = self.price_data.sort_values(['asset', 'date'])
        
        logger.info(f"Loaded {len(self.price_data)} price records")
    
    def _set_date_range(self):
        """Set the overall date range for the dataset."""
        if not self.articles_df.empty and not self.price_data.empty:
            article_min = self.articles_df['date'].min()
            article_max = self.articles_df['date'].max()
            price_min = self.price_data['date'].min()
            price_max = self.price_data['date'].max()
            
            self.date_range = {
                'start': min(article_min, price_min).strftime('%Y-%m-%d'),
                'end': max(article_max, price_max).strftime('%Y-%m-%d'),
                'article_start': article_min.strftime('%Y-%m-%d'),
                'article_end': article_max.strftime('%Y-%m-%d'),
                'price_start': price_min.strftime('%Y-%m-%d'),
                'price_end': price_max.strftime('%Y-%m-%d')
            }
    
    def _create_processed_datasets(self):
        """Create pre-processed datasets for analysis."""
        # Create weekly aggregated data
        self._create_weekly_data()
        
        # Create monthly aggregated data
        self._create_monthly_data()
        
        # Create article-price correlations
        self._create_article_price_correlations()
    
    def _create_weekly_data(self):
        """Create weekly aggregated data."""
        if self.articles_df.empty or self.price_data.empty:
            return
        
        # Set date as index for resampling
        articles_weekly = self.articles_df.set_index('date').resample('W').agg({
            'title': 'count',  # Number of articles per week
            'signal_strength': 'mean',
            'extraction_confidence': 'mean',
            'technical_density': 'mean',
            'sentiment_bullish': 'sum',
            'sentiment_bearish': 'sum',
            'num_images': 'sum'
        }).rename(columns={'title': 'article_count'})
        
        # Create weekly price data
        price_weekly_btc = self.price_data[self.price_data['asset'] == 'BTC'].set_index('date').resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'return_1d': lambda x: (1 + x).prod() - 1,  # Compound weekly return
            'volatility_7d': 'mean'
        }).add_prefix('btc_')
        
        price_weekly_eth = self.price_data[self.price_data['asset'] == 'ETH'].set_index('date').resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'return_1d': lambda x: (1 + x).prod() - 1,  # Compound weekly return
            'volatility_7d': 'mean'
        }).add_prefix('eth_')
        
        # Combine weekly data
        self._weekly_data = articles_weekly.join([price_weekly_btc, price_weekly_eth], how='outer')
        self._weekly_data = self._weekly_data.fillna(0)
        
        logger.info(f"Created weekly dataset with {len(self._weekly_data)} weeks")
    
    def _create_monthly_data(self):
        """Create monthly aggregated data."""
        if self.articles_df.empty or self.price_data.empty:
            return
        
        # Monthly articles data
        articles_monthly = self.articles_df.set_index('date').resample('ME').agg({
            'title': 'count',
            'signal_strength': 'mean',
            'extraction_confidence': 'mean',
            'technical_density': 'mean',
            'sentiment_bullish': 'sum',
            'sentiment_bearish': 'sum',
            'num_images': 'sum'
        }).rename(columns={'title': 'article_count'})
        
        # Monthly price data
        price_monthly_btc = self.price_data[self.price_data['asset'] == 'BTC'].set_index('date').resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'return_1d': lambda x: (1 + x).prod() - 1,  # Compound monthly return
            'volatility_30d': 'mean'
        }).add_prefix('btc_')
        
        price_monthly_eth = self.price_data[self.price_data['asset'] == 'ETH'].set_index('date').resample('ME').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'return_1d': lambda x: (1 + x).prod() - 1,  # Compound monthly return
            'volatility_30d': 'mean'
        }).add_prefix('eth_')
        
        # Combine monthly data
        self._monthly_data = articles_monthly.join([price_monthly_btc, price_monthly_eth], how='outer')
        self._monthly_data = self._monthly_data.fillna(0)
        
        logger.info(f"Created monthly dataset with {len(self._monthly_data)} months")
    
    def _create_article_price_correlations(self):
        """Create article-price correlations for detailed analysis."""
        if self.articles_df.empty or self.price_data.empty:
            return
        
        correlations = []
        
        for _, article in self.articles_df.iterrows():
            article_date = article['date']
            
            # Get price data for different time windows after article
            for days in [1, 3, 7, 14, 30]:
                end_date = article_date + timedelta(days=days)
                
                for asset in ['BTC', 'ETH']:
                    if asset in article['assets_mentioned']:
                        # Get price data for this period
                        price_subset = self.price_data[
                            (self.price_data['asset'] == asset) &
                            (self.price_data['date'] >= article_date) &
                            (self.price_data['date'] <= end_date)
                        ].copy()
                        
                        if not price_subset.empty:
                            # Calculate period return
                            start_price = price_subset.iloc[0]['open']
                            end_price = price_subset.iloc[-1]['close']
                            period_return = (end_price - start_price) / start_price
                            
                            # Calculate period volatility
                            period_volatility = price_subset['return_1d'].std() * np.sqrt(days)
                            
                            correlations.append({
                                'article_date': article_date,
                                'article_title': article['title'],
                                'asset': asset,
                                'days_forward': days,
                                'signal_strength': article['signal_strength'],
                                'directional_bias': article['directional_bias'],
                                'primary_theme': article['primary_theme'],
                                'period_return': period_return,
                                'period_volatility': period_volatility,
                                'extraction_confidence': article['extraction_confidence'],
                                'market_period': article['market_period']
                            })
        
        self.correlations_df = pd.DataFrame(correlations)
        logger.info(f"Created {len(correlations)} article-price correlation records")
    
    def get_filtered_articles(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Get filtered articles based on dashboard filters."""
        df = self.articles_df.copy()
        
        # Apply date filter
        if filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        
        # Apply asset filter
        if filters.get('assets'):
            df = df[df['assets_mentioned'].apply(lambda x: any(asset in x for asset in filters['assets']) if isinstance(x, list) else False)]
        
        # Apply theme filter
        if filters.get('themes'):
            df = df[df['primary_theme'].isin(filters['themes'])]
        
        # Apply confidence filter
        if filters.get('confidence_threshold'):
            df = df[df['extraction_confidence'] >= filters['confidence_threshold']]
        
        return df
    
    def get_filtered_price_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Get filtered price data based on dashboard filters."""
        df = self.price_data.copy()
        
        # Apply date filter
        if filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        
        # Apply asset filter
        if filters.get('assets'):
            df = df[df['asset'].isin(filters['assets'])]
        
        return df
    
    def get_weekly_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get weekly aggregated data with optional filtering."""
        if self._weekly_data is None:
            return pd.DataFrame()
        
        df = self._weekly_data.copy()
        
        if filters and filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        
        return df
    
    def get_monthly_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get monthly aggregated data with optional filtering."""
        if self._monthly_data is None:
            return pd.DataFrame()
        
        df = self._monthly_data.copy()
        
        if filters and filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        
        return df
    
    def get_correlation_data(self, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """Get article-price correlation data with optional filtering."""
        if not hasattr(self, 'correlations_df'):
            return pd.DataFrame()
        
        df = self.correlations_df.copy()
        
        if filters:
            # Apply date filter
            if filters.get('date_range') and len(filters['date_range']) == 2:
                start_date, end_date = filters['date_range']
                df = df[(df['article_date'] >= pd.to_datetime(start_date)) & 
                       (df['article_date'] <= pd.to_datetime(end_date))]
            
            # Apply asset filter
            if filters.get('assets'):
                df = df[df['asset'].isin(filters['assets'])]
            
            # Apply confidence filter
            if filters.get('confidence_threshold'):
                df = df[df['extraction_confidence'] >= filters['confidence_threshold']]
        
        return df
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the dataset."""
        return {
            'total_articles': len(self.articles),
            'total_price_records': len(self.price_data),
            'total_images': sum(len(a.get('analyzed_images', [])) for a in self.articles),
            'date_range': self.date_range,
            'assets_covered': self.price_data['asset'].unique().tolist() if not self.price_data.empty else [],
            'themes_covered': self.articles_df['primary_theme'].unique().tolist() if not self.articles_df.empty else [],
            'market_periods': self.articles_df['market_period'].unique().tolist() if not self.articles_df.empty else []
        }
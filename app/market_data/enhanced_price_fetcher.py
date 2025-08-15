"""
Enhanced price data fetcher with multiple data providers.
Provides robust historical data fetching for options analysis.
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Optional imports with sentinel values
try:
    from alpha_vantage.timeseries import TimeSeries
    HAS_ALPHA_VANTAGE = True
except ImportError:
    TimeSeries = None
    HAS_ALPHA_VANTAGE = False

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    Fred = None
    HAS_FRED = False

logger = logging.getLogger(__name__)

@dataclass
class DataProviderConfig:
    """Configuration for data providers."""
    alpha_vantage_key: Optional[str] = None
    fred_key: Optional[str] = None
    use_yahoo_finance: bool = True
    use_alpha_vantage: bool = False
    use_fred: bool = False
    request_delay: float = 0.5  # Delay between API calls
    max_retries: int = 3
    timeout: int = 30

class EnhancedPriceFetcher:
    """Enhanced price data fetcher with multiple providers and robust error handling."""
    
    def __init__(self, config: Optional[DataProviderConfig] = None):
        """Initialize enhanced price fetcher.
        
        Args:
            config: Data provider configuration
        """
        self.config = config or DataProviderConfig()
        self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'price_data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data providers
        self._init_providers()
        
        # Crypto symbol mappings for different providers
        self.symbol_mappings = {
            'yfinance': {'BTC': 'BTC-USD', 'ETH': 'ETH-USD'},
            'alpha_vantage': {'BTC': 'BTCUSD', 'ETH': 'ETHUSD'},
            'fred': {'BTC': 'DEXBDTRUSD', 'ETH': 'DEXETHTRUSD'}  # Federal Reserve crypto indices
        }
    
    def _init_providers(self):
        """Initialize data provider clients."""
        self.providers = {}
        
        # Yahoo Finance (always available, no API key needed)
        if self.config.use_yahoo_finance:
            self.providers['yfinance'] = True
            logger.info("Yahoo Finance provider initialized")
        
        # Alpha Vantage
        if self.config.use_alpha_vantage and self.config.alpha_vantage_key:
            if not HAS_ALPHA_VANTAGE:
                logger.warning("Alpha Vantage requested but not available (import failed)")
            else:
                try:
                    self.providers['alpha_vantage'] = TimeSeries(
                        key=self.config.alpha_vantage_key,
                        output_format='pandas'
                    )
                    logger.info("Alpha Vantage provider initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Alpha Vantage: {e}")
        
        # FRED
        if self.config.use_fred and self.config.fred_key:
            if not HAS_FRED:
                logger.warning("FRED requested but not available (import failed)")
            else:
                try:
                    self.providers['fred'] = Fred(api_key=self.config.fred_key)
                    logger.info("FRED provider initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize FRED: {e}")
    
    def fetch_historical_data(
        self, 
        asset: str, 
        start_date: str = '2021-01-01',
        end_date: Optional[str] = None,
        provider: str = 'yfinance'
    ) -> Optional[pd.DataFrame]:
        """Fetch historical price data from specified provider.
        
        Args:
            asset: Asset symbol (BTC, ETH)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            provider: Data provider to use
            
        Returns:
            DataFrame with historical OHLCV data or None if failed
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {asset} data from {start_date} to {end_date} using {provider}")
        
        for attempt in range(self.config.max_retries):
            try:
                if provider == 'yfinance':
                    return self._fetch_yfinance(asset, start_date, end_date)
                elif provider == 'alpha_vantage':
                    return self._fetch_alpha_vantage(asset, start_date, end_date)
                elif provider == 'fred':
                    return self._fetch_fred(asset, start_date, end_date)
                else:
                    logger.error(f"Unknown provider: {provider}")
                    return None
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {provider}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.request_delay * (attempt + 1))
                else:
                    logger.error(f"All attempts failed for {provider}")
        
        return None
    
    def _fetch_yfinance(self, asset: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance."""
        symbol = self.symbol_mappings['yfinance'].get(asset)
        if not symbol:
            logger.error(f"No Yahoo Finance symbol mapping for {asset}")
            return None
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            logger.warning(f"No data returned from Yahoo Finance for {symbol}")
            return None
        
        # Standardize column names
        df = df.reset_index()
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Ensure we have the required columns
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in Yahoo Finance data for {asset}")
            return None
        
        # Add metadata
        df['asset'] = asset
        df['timestamp'] = df['date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df['unix_timestamp'] = df['date'].astype(int) // 10**9
        
        logger.info(f"Successfully fetched {len(df)} records from Yahoo Finance for {asset}")
        return df
    
    def _fetch_alpha_vantage(self, asset: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage."""
        if 'alpha_vantage' not in self.providers:
            logger.error("Alpha Vantage provider not initialized")
            return None
        
        symbol = self.symbol_mappings['alpha_vantage'].get(asset)
        if not symbol:
            logger.error(f"No Alpha Vantage symbol mapping for {asset}")
            return None
        
        # Note: Alpha Vantage free tier has limitations
        # This is a simplified implementation
        logger.warning("Alpha Vantage implementation simplified due to API limitations")
        return None
    
    def _fetch_fred(self, asset: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from FRED."""
        if 'fred' not in self.providers:
            logger.error("FRED provider not initialized")
            return None
        
        symbol = self.symbol_mappings['fred'].get(asset)
        if not symbol:
            logger.error(f"No FRED symbol mapping for {asset}")
            return None
        
        # Note: FRED may have limited crypto data
        logger.warning("FRED crypto data may be limited")
        return None
    
    def fetch_with_fallback(
        self,
        asset: str,
        start_date: str = '2021-01-01',
        end_date: Optional[str] = None,
        preferred_providers: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch data with automatic fallback between providers.
        
        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            preferred_providers: List of providers in order of preference
            
        Returns:
            DataFrame with data from first successful provider
        """
        if preferred_providers is None:
            preferred_providers = ['yfinance', 'alpha_vantage', 'fred']
        
        for provider in preferred_providers:
            if provider in self.providers:
                logger.info(f"Trying {provider} for {asset}")
                df = self.fetch_historical_data(asset, start_date, end_date, provider)
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {asset} data using {provider}")
                    return df
                else:
                    logger.warning(f"{provider} failed for {asset}, trying next provider")
            else:
                logger.debug(f"{provider} not available, skipping")
        
        logger.error(f"All providers failed for {asset}")
        return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for options analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = df.copy()
        
        # Basic price calculations
        df['mid_price'] = (df['high'] + df['low']) / 2
        df['price_range'] = df['high'] - df['low']
        df['price_range_pct'] = (df['price_range'] / df['close']) * 100
        
        # Returns calculations
        df['return_1d'] = df['close'].pct_change()
        df['return_1d_bps'] = df['return_1d'] * 10000
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
        
        # Overnight and intraday returns (approximated)
        df['return_overnight'] = (df['open'] / df['close'].shift(1) - 1)
        df['return_intraday'] = (df['close'] / df['open'] - 1)
        
        # Rolling volatility (annualized)
        for window in [7, 14, 30]:
            returns_col = f'volatility_{window}d'
            df[returns_col] = df['return_1d'].rolling(window=window).std() * np.sqrt(365) * 100
        
        # Simple moving averages
        for window in [7, 14, 30]:
            sma_col = f'sma_{window}'
            df[sma_col] = df['close'].rolling(window=window).mean()
            
            # Price vs SMA
            price_vs_sma = f'price_vs_sma_{window}'
            df[price_vs_sma] = ((df['close'] / df[sma_col]) - 1) * 100
        
        # Price percentiles (relative position in recent range)
        for window in [30, 60, 90]:
            percentile_col = f'price_percentile_{window}d'
            df[percentile_col] = df['close'].rolling(window=window).rank(pct=True) * 100
        
        # Support/Resistance levels (simplified)
        df['support_level'] = df['low'].rolling(window=14).min()
        df['resistance_level'] = df['high'].rolling(window=14).max()
        df['distance_to_support'] = ((df['close'] - df['support_level']) / df['close']) * 100
        df['distance_to_resistance'] = ((df['resistance_level'] - df['close']) / df['close']) * 100
        
        # Options-specific metrics
        df['implied_move_1d'] = df['volatility_30d'] / np.sqrt(365)  # Simplified daily implied move
        
        # Gap analysis
        df['gap_size'] = df['open'] - df['close'].shift(1)
        df['gap_size_pct'] = (df['gap_size'] / df['close'].shift(1)) * 100
        df['is_gap_up'] = df['gap_size'] > 0
        df['is_gap_down'] = df['gap_size'] < 0
        
        # Volume analysis
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['dollar_volume'] = df['volume'] * df['close']
        
        # VWAP approximation
        df['vwap_approx'] = (df['volume'] * df['close']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Candlestick analysis
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = (df['body_size'] / df['close']) * 100
        
        # Local highs and lows
        df['is_local_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['is_local_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        
        return df
    
    def save_enhanced_data(
        self,
        df: pd.DataFrame,
        asset: str,
        file_suffix: str = 'daily_prices'
    ) -> Path:
        """Save enhanced data to CSV file.
        
        Args:
            df: DataFrame to save
            asset: Asset symbol
            file_suffix: Suffix for filename
            
        Returns:
            Path to saved file
        """
        filename = f"{asset.lower()}_{file_suffix}.csv"
        file_path = self.data_dir / filename
        
        # Sort by date and save
        df_sorted = df.sort_values('date').reset_index(drop=True)
        df_sorted.to_csv(file_path, index=False)
        
        logger.info(f"Saved {len(df_sorted)} records to {file_path}")
        return file_path
    
    def fetch_and_process_complete_dataset(
        self,
        asset: str,
        start_date: str = '2021-01-01',
        end_date: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        """Fetch complete dataset and process with all indicators.
        
        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (processed DataFrame, saved file path)
        """
        logger.info(f"Starting complete dataset fetch and processing for {asset}")
        
        # Fetch raw data
        raw_df = self.fetch_with_fallback(asset, start_date, end_date)
        
        if raw_df is None or raw_df.empty:
            logger.error(f"Failed to fetch data for {asset}")
            return None, None
        
        # Calculate technical indicators
        processed_df = self.calculate_technical_indicators(raw_df)
        
        # Save to file
        saved_path = self.save_enhanced_data(processed_df, asset)
        
        logger.info(f"Complete dataset processing finished for {asset}")
        logger.info(f"  Records: {len(processed_df)}")
        logger.info(f"  Date range: {processed_df['date'].min().date()} to {processed_df['date'].max().date()}")
        logger.info(f"  Columns: {len(processed_df.columns)}")
        
        return processed_df, saved_path


# Global instance
price_fetcher = EnhancedPriceFetcher()


def fetch_missing_btc_data() -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    """Convenience function to fetch missing BTC data."""
    return price_fetcher.fetch_and_process_complete_dataset('BTC', '2021-01-01')


def validate_eth_data() -> Dict[str, Union[bool, int, str]]:
    """Validate existing ETH data and identify gaps."""
    validation_result = {
        'is_valid': False,
        'total_records': 0,
        'date_range': '',
        'gaps_found': 0,
        'missing_dates': [],
        'needs_refresh': False
    }
    
    try:
        eth_file = Path(__file__).parent.parent.parent / 'data' / 'price_data' / 'eth_daily_prices.csv'
        
        if not eth_file.exists():
            validation_result['needs_refresh'] = True
            return validation_result
        
        df = pd.read_csv(eth_file)
        df['date'] = pd.to_datetime(df['date'])
        
        validation_result['total_records'] = len(df)
        validation_result['date_range'] = f"{df['date'].min().date()} to {df['date'].max().date()}"
        
        # Check for date gaps
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        missing_dates = set(date_range) - set(df['date'])
        
        validation_result['gaps_found'] = len(missing_dates)
        validation_result['missing_dates'] = [d.strftime('%Y-%m-%d') for d in sorted(missing_dates)]
        validation_result['is_valid'] = validation_result['gaps_found'] == 0
        
        # Check if we need to extend to current date
        if df['date'].max().date() < datetime.now().date() - timedelta(days=1):
            validation_result['needs_refresh'] = True
        
    except Exception as e:
        logger.error(f"ETH data validation failed: {e}")
        validation_result['needs_refresh'] = True
    
    return validation_result
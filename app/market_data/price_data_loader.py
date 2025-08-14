"""
Price data loader for options analysis and market data operations.
Provides clean, standardized access to historical price data for BTC and ETH.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)

class PriceDataLoader:
    """Load and manage historical price data for crypto assets."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize price data loader.
        
        Args:
            data_dir: Directory containing price data CSVs
        """
        self.data_dir = data_dir or (Path(__file__).parent.parent.parent / 'data' / 'price_data')
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_available_assets(self) -> List[str]:
        """Get list of assets with available price data."""
        assets = []
        for asset in ['btc', 'eth']:
            if (self.data_dir / f'{asset}_daily_prices.csv').exists():
                assets.append(asset.upper())
        return assets
    
    def load_asset_data(self, asset: str, refresh_cache: bool = False) -> pd.DataFrame:
        """Load price data for a specific asset.
        
        Args:
            asset: Asset symbol (BTC, ETH)
            refresh_cache: Whether to reload from file
            
        Returns:
            DataFrame with comprehensive price and indicator data
        """
        asset = asset.upper()
        
        if not refresh_cache and asset in self._cache:
            return self._cache[asset].copy()
        
        file_path = self.data_dir / f'{asset.lower()}_daily_prices.csv'
        
        if not file_path.exists():
            raise FileNotFoundError(f"Price data not found for {asset} at {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            self._cache[asset] = df
            logger.info(f"Loaded {len(df)} price records for {asset}")
            return df.copy()
            
        except Exception as e:
            logger.error(f"Failed to load price data for {asset}: {e}")
            raise
    
    def get_price_at_date(self, asset: str, target_date: Union[str, date, datetime]) -> Optional[Dict]:
        """Get price data for a specific date.
        
        Args:
            asset: Asset symbol
            target_date: Date to look up
            
        Returns:
            Dictionary with price data or None if not found
        """
        df = self.load_asset_data(asset)
        
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()
        
        # Find exact match or closest date
        df['date_only'] = df['date'].dt.date
        exact_match = df[df['date_only'] == target_date]
        
        if not exact_match.empty:
            row = exact_match.iloc[0]
        else:
            # Find closest date (within 7 days)
            df['date_diff'] = abs(df['date'] - pd.to_datetime(target_date))
            closest = df[df['date_diff'] <= pd.Timedelta(days=7)].sort_values('date_diff')
            
            if closest.empty:
                return None
            row = closest.iloc[0]
        
        return {
            'date': row['date'].strftime('%Y-%m-%d'),
            'asset': asset,
            'close': row['close'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'volume': row.get('volume'),
            'volatility_30d': row.get('volatility_30d'),
            'implied_move_1d': row.get('implied_move_1d'),
            'support_level': row.get('support_level'),
            'resistance_level': row.get('resistance_level'),
            'price_percentile_30d': row.get('price_percentile_30d')
        }
    
    def get_returns_data(self, asset: str, start_date: Optional[str] = None, 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Get returns data for backtesting and analysis.
        
        Args:
            asset: Asset symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with returns and risk metrics
        """
        df = self.load_asset_data(asset)
        
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # Select relevant columns for returns analysis
        return_columns = [
            'date', 'asset', 'close',
            'return_1d', 'return_1d_bps', 'log_return_1d',
            'return_overnight', 'return_intraday',
            'volatility_7d', 'volatility_14d', 'volatility_30d',
            'implied_move_1d', 'price_percentile_30d'
        ]
        
        available_cols = [col for col in return_columns if col in df.columns]
        return df[available_cols].copy()
    
    def get_options_context(self, asset: str, analysis_date: Union[str, date]) -> Dict:
        """Get price context relevant for options analysis.
        
        Args:
            asset: Asset symbol
            analysis_date: Date of analysis
            
        Returns:
            Dictionary with options-relevant price context
        """
        price_data = self.get_price_at_date(asset, analysis_date)
        
        if not price_data:
            return {}
        
        df = self.load_asset_data(asset)
        analysis_date = pd.to_datetime(analysis_date).date()
        
        # Get historical context (last 30 days)
        end_idx = df[df['date'].dt.date <= analysis_date].index
        if end_idx.empty:
            return price_data
        
        last_idx = end_idx[-1]
        start_idx = max(0, last_idx - 30)
        context_df = df.iloc[start_idx:last_idx+1]
        
        current_price = price_data['close']
        
        # Calculate options-relevant metrics
        context = {
            **price_data,
            'spot_price': current_price,
            'recent_high_30d': context_df['high'].max(),
            'recent_low_30d': context_df['low'].min(),
            'price_change_30d': ((current_price - context_df.iloc[0]['close']) / context_df.iloc[0]['close']) * 100,
            'volatility_regime': 'high' if price_data.get('volatility_30d', 0) > 50 else 'normal' if price_data.get('volatility_30d', 0) > 25 else 'low',
            'trend_direction': 'up' if price_data.get('price_vs_sma_14', 0) > 2 else 'down' if price_data.get('price_vs_sma_14', 0) < -2 else 'sideways',
            'suggested_strikes': {
                'otm_calls': [current_price * mult for mult in [1.05, 1.10, 1.15, 1.20]],
                'atm_range': [current_price * mult for mult in [0.98, 1.00, 1.02]],
                'otm_puts': [current_price * mult for mult in [0.95, 0.90, 0.85, 0.80]]
            }
        }
        
        return context
    
    def get_correlation_data(self, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Get correlation data between BTC and ETH.
        
        Returns:
            DataFrame with both assets' returns for correlation analysis
        """
        assets_data = []
        
        for asset in ['BTC', 'ETH']:
            try:
                df = self.get_returns_data(asset, start_date, end_date)
                df = df[['date', 'return_1d', 'volatility_30d']].copy()
                df = df.rename(columns={
                    'return_1d': f'{asset}_return',
                    'volatility_30d': f'{asset}_vol'
                })
                assets_data.append(df)
            except FileNotFoundError:
                continue
        
        if len(assets_data) == 0:
            return pd.DataFrame()
        elif len(assets_data) == 1:
            return assets_data[0]
        else:
            # Merge on date
            result = assets_data[0]
            for df in assets_data[1:]:
                result = result.merge(df, on='date', how='outer')
            
            return result.sort_values('date').reset_index(drop=True)


# Global instance for easy access
price_loader = PriceDataLoader()


def get_price_at_date(asset: str, date: Union[str, date, datetime]) -> Optional[Dict]:
    """Convenience function to get price data for a specific date."""
    return price_loader.get_price_at_date(asset, date)


def get_options_context(asset: str, date: Union[str, date]) -> Dict:
    """Convenience function to get options context for analysis."""
    return price_loader.get_options_context(asset, date)


def get_available_assets() -> List[str]:
    """Convenience function to get available assets."""
    return price_loader.get_available_assets()
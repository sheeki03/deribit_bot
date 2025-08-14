#!/usr/bin/env python3
"""
Script to format Binance historical price CSV files for integration with options analysis.
Converts formatted numbers, standardizes dates, and creates clean price data.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import re
from pathlib import Path
import argparse


def parse_trading_number(value_str: str) -> float:
    """Parse numbers with K/M suffixes and comma formatting."""
    if pd.isna(value_str) or value_str == '':
        return np.nan
    
    # Remove quotes and whitespace
    value_str = str(value_str).strip().strip('"')
    
    # Handle percentage
    if value_str.endswith('%'):
        return float(value_str.rstrip('%'))
    
    # Remove commas
    value_str = value_str.replace(',', '')
    
    # Handle K/M suffixes
    multiplier = 1
    if value_str.endswith('K'):
        multiplier = 1000
        value_str = value_str.rstrip('K')
    elif value_str.endswith('M'):
        multiplier = 1000000
        value_str = value_str.rstrip('M')
    elif value_str.endswith('B'):
        multiplier = 1000000000
        value_str = value_str.rstrip('B')
    
    try:
        return float(value_str) * multiplier
    except ValueError:
        return np.nan


def parse_trading_date(date_str: str) -> pd.Timestamp:
    """Parse date in DD-MM-YYYY format."""
    try:
        # Remove quotes and BOM
        date_str = str(date_str).strip().strip('"').strip('\ufeff')
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    except:
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT


def format_price_csv(input_path: Path, asset: str) -> pd.DataFrame:
    """Format a single price CSV file with comprehensive financial metrics."""
    
    # Read raw CSV
    df = pd.read_csv(input_path)
    
    # Clean column names
    df.columns = df.columns.str.strip().str.strip('"')
    
    # Parse date
    df['date'] = df['Date'].apply(parse_trading_date)
    df = df.dropna(subset=['date']).sort_values('date')
    
    # Parse price columns
    price_columns = ['Price', 'Open', 'High', 'Low']
    for col in price_columns:
        if col in df.columns:
            df[col.lower()] = df[col].apply(parse_trading_number)
    
    # Parse volume and change
    if 'Vol.' in df.columns:
        df['volume'] = df['Vol.'].apply(parse_trading_number)
    if 'Change %' in df.columns:
        df['change_pct'] = df['Change %'].apply(parse_trading_number)
    
    # Standardize column names
    df = df.rename(columns={'price': 'close'})
    
    # Add comprehensive financial metrics
    df['asset'] = asset
    df['timestamp'] = df['date'].dt.strftime('%Y-%m-%dT00:00:00Z')
    df['unix_timestamp'] = df['date'].astype('int64') // 10**9
    
    # Price-based metrics
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['price_range'] = df['high'] - df['low']
    df['price_range_pct'] = (df['price_range'] / df['close']) * 100
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_size_pct'] = (df['body_size'] / df['close']) * 100
    
    # Returns calculations
    df['return_1d'] = df['close'].pct_change()
    df['return_1d_bps'] = df['return_1d'] * 10000
    df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))
    df['return_overnight'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['return_intraday'] = (df['close'] - df['open']) / df['open']
    
    # Rolling volatilities (annualized)
    for window in [7, 14, 30]:
        df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std() * np.sqrt(365) * 100
    
    # Price momentum indicators
    for window in [7, 14, 30]:
        df[f'sma_{window}'] = df['close'].rolling(window).mean()
        df[f'price_vs_sma_{window}'] = ((df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']) * 100
    
    # Support/resistance levels (local extremes)
    df['is_local_high'] = (df['high'] == df['high'].rolling(5, center=True).max())
    df['is_local_low'] = (df['low'] == df['low'].rolling(5, center=True).min())
    
    # Price percentiles (for options strike selection)
    for window in [30, 60, 90]:
        df[f'price_percentile_{window}d'] = df['close'].rolling(window).rank(pct=True) * 100
    
    # Liquidity metrics
    if 'volume' in df.columns:
        df['volume_sma_30'] = df['volume'].rolling(30).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_30']
        df['dollar_volume'] = df['volume'] * df['close']
        df['vwap_approx'] = (df['high'] + df['low'] + df['close']) / 3  # Approximation
    
    # Market structure
    df['gap_size'] = df['open'] - df['close'].shift(1)
    df['gap_size_pct'] = (df['gap_size'] / df['close'].shift(1)) * 100
    df['is_gap_up'] = df['gap_size_pct'] > 1.0
    df['is_gap_down'] = df['gap_size_pct'] < -1.0
    
    # Options-relevant metrics
    df['implied_move_1d'] = df['volatility_30d'] / np.sqrt(365) * df['close']  # Daily expected move
    df['support_level'] = df['low'].rolling(20).min()
    df['resistance_level'] = df['high'].rolling(20).max()
    df['distance_to_support'] = ((df['close'] - df['support_level']) / df['close']) * 100
    df['distance_to_resistance'] = ((df['resistance_level'] - df['close']) / df['close']) * 100
    
    # Select final columns in logical order
    final_columns = [
        'date', 'timestamp', 'unix_timestamp', 'asset',
        'open', 'high', 'low', 'close', 'volume',
        'mid_price', 'price_range', 'price_range_pct',
        'return_1d', 'return_1d_bps', 'log_return_1d',
        'return_overnight', 'return_intraday',
        'volatility_7d', 'volatility_14d', 'volatility_30d',
        'sma_7', 'sma_14', 'sma_30',
        'price_vs_sma_7', 'price_vs_sma_14', 'price_vs_sma_30',
        'price_percentile_30d', 'price_percentile_60d', 'price_percentile_90d',
        'implied_move_1d', 'support_level', 'resistance_level',
        'distance_to_support', 'distance_to_resistance',
        'gap_size', 'gap_size_pct', 'is_gap_up', 'is_gap_down',
        'volume_ratio', 'dollar_volume', 'vwap_approx',
        'upper_wick', 'lower_wick', 'body_size', 'body_size_pct',
        'is_local_high', 'is_local_low'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in final_columns if col in df.columns]
    result = df[available_columns].copy()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Format Binance price CSV files")
    parser.add_argument('--btc-file', default='Bitcoin Historical Data.csv', help='Bitcoin CSV file')
    parser.add_argument('--eth-file', default='ETH_USD Binance Historical Data.csv', help='ETH CSV file')
    parser.add_argument('--output-dir', default='data/price_data', help='Output directory')
    
    args = parser.parse_args()
    
    base_dir = Path.cwd()
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process BTC data
    btc_path = base_dir / args.btc_file
    if btc_path.exists():
        print(f"Processing Bitcoin data from {btc_path}")
        btc_df = format_price_csv(btc_path, 'BTC')
        btc_output = output_dir / 'btc_daily_prices.csv'
        btc_df.to_csv(btc_output, index=False)
        print(f"✅ Saved {len(btc_df)} BTC records to {btc_output}")
        print(f"   Date range: {btc_df['date'].min().date()} to {btc_df['date'].max().date()}")
        print(f"   Price range: ${btc_df['close'].min():,.2f} - ${btc_df['close'].max():,.2f}")
    else:
        print(f"❌ Bitcoin file not found: {btc_path}")
    
    # Process ETH data
    eth_path = base_dir / args.eth_file
    if eth_path.exists():
        print(f"\nProcessing Ethereum data from {eth_path}")
        eth_df = format_price_csv(eth_path, 'ETH')
        eth_output = output_dir / 'eth_daily_prices.csv'
        eth_df.to_csv(eth_output, index=False)
        print(f"✅ Saved {len(eth_df)} ETH records to {eth_output}")
        print(f"   Date range: {eth_df['date'].min().date()} to {eth_df['date'].max().date()}")
        print(f"   Price range: ${eth_df['close'].min():,.2f} - ${eth_df['close'].max():,.2f}")
    else:
        print(f"❌ Ethereum file not found: {eth_path}")
    
    # Create combined dataset
    dfs = []
    if btc_path.exists():
        dfs.append(btc_df)
    if eth_path.exists():
        dfs.append(eth_df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['date', 'asset'])
        
        combined_output = output_dir / 'combined_daily_prices.csv'
        combined_df.to_csv(combined_output, index=False)
        print(f"\n✅ Saved combined dataset to {combined_output}")
        print(f"   Total records: {len(combined_df)}")
        
        # Create summary stats
        summary_output = output_dir / 'price_data_summary.json'
        summary = {
            'files_processed': len(dfs),
            'total_records': len(combined_df),
            'assets': combined_df['asset'].unique().tolist(),
            'date_range': {
                'start': combined_df['date'].min().isoformat(),
                'end': combined_df['date'].max().isoformat()
            },
            'price_stats': {}
        }
        
        for asset in combined_df['asset'].unique():
            asset_data = combined_df[combined_df['asset'] == asset]
            summary['price_stats'][asset] = {
                'records': len(asset_data),
                'price_min': float(asset_data['close'].min()),
                'price_max': float(asset_data['close'].max()),
                'price_mean': float(asset_data['close'].mean()),
                'volatility_1d': float(asset_data['return_1d'].std() * 100)  # As percentage
            }
        
        import json
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved summary to {summary_output}")


if __name__ == '__main__':
    main()
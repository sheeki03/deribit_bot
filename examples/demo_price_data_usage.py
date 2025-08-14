#!/usr/bin/env python3
"""
Demonstration of formatted price data usage for options analysis.
Shows how to access and use the comprehensive price datasets.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.market_data.price_data_loader import price_loader, get_price_at_date, get_options_context
import json
from datetime import date, datetime


def demo_basic_usage():
    """Demonstrate basic price data access."""
    print("=== BASIC PRICE DATA USAGE ===\n")
    
    # Check available assets
    assets = price_loader.get_available_assets()
    print(f"Available assets: {assets}")
    
    # Load BTC data
    if 'BTC' in assets:
        btc_data = price_loader.load_asset_data('BTC')
        print(f"\nBTC Dataset:")
        print(f"  Records: {len(btc_data)}")
        print(f"  Date range: {btc_data['date'].min().date()} to {btc_data['date'].max().date()}")
        print(f"  Price range: ${btc_data['close'].min():,.2f} - ${btc_data['close'].max():,.2f}")
        print(f"  Columns: {list(btc_data.columns)}")
    
    # Load ETH data
    if 'ETH' in assets:
        eth_data = price_loader.load_asset_data('ETH')
        print(f"\nETH Dataset:")
        print(f"  Records: {len(eth_data)}")
        print(f"  Date range: {eth_data['date'].min().date()} to {eth_data['date'].max().date()}")
        print(f"  Price range: ${eth_data['close'].min():,.2f} - ${eth_data['close'].max():,.2f}")


def demo_specific_date_lookup():
    """Demonstrate getting price data for specific dates."""
    print("\n=== SPECIFIC DATE PRICE LOOKUP ===\n")
    
    # Get recent price data
    recent_date = "2025-08-10"
    
    for asset in ['BTC', 'ETH']:
        try:
            price_data = get_price_at_date(asset, recent_date)
            if price_data:
                print(f"{asset} on {recent_date}:")
                print(f"  Close: ${price_data['close']:,.2f}")
                print(f"  Daily Range: ${price_data['low']:,.2f} - ${price_data['high']:,.2f}")
                print(f"  Volume: {price_data['volume']:,.0f}" if price_data['volume'] else "  Volume: N/A")
                print(f"  30d Volatility: {price_data.get('volatility_30d', 0):.1f}%" if price_data.get('volatility_30d') else "  Volatility: N/A")
                print()
        except FileNotFoundError:
            print(f"{asset}: No data available")


def demo_options_context():
    """Demonstrate options analysis context."""
    print("=== OPTIONS ANALYSIS CONTEXT ===\n")
    
    analysis_date = "2025-08-10"
    
    for asset in ['BTC', 'ETH']:
        try:
            context = get_options_context(asset, analysis_date)
            if context:
                print(f"{asset} Options Context for {analysis_date}:")
                print(f"  Spot Price: ${context.get('spot_price', 0):,.2f}")
                print(f"  30d High/Low: ${context.get('recent_high_30d', 0):,.2f} / ${context.get('recent_low_30d', 0):,.2f}")
                print(f"  30d Price Change: {context.get('price_change_30d', 0):+.1f}%")
                print(f"  Volatility Regime: {context.get('volatility_regime', 'unknown')}")
                print(f"  Trend Direction: {context.get('trend_direction', 'unknown')}")
                
                strikes = context.get('suggested_strikes', {})
                if strikes:
                    print(f"  Suggested Option Strikes:")
                    if 'otm_calls' in strikes:
                        calls = [f"${s:,.0f}" for s in strikes['otm_calls'][:3]]
                        print(f"    OTM Calls: {', '.join(calls)}")
                    if 'atm_range' in strikes:
                        atm = [f"${s:,.0f}" for s in strikes['atm_range']]
                        print(f"    ATM Range: {', '.join(atm)}")
                    if 'otm_puts' in strikes:
                        puts = [f"${s:,.0f}" for s in strikes['otm_puts'][:3]]
                        print(f"    OTM Puts: {', '.join(puts)}")
                print()
        except FileNotFoundError:
            print(f"{asset}: No data available for options context")


def demo_returns_analysis():
    """Demonstrate returns data for backtesting."""
    print("=== RETURNS DATA FOR BACKTESTING ===\n")
    
    # Get recent returns data
    for asset in ['BTC', 'ETH']:
        try:
            returns_data = price_loader.get_returns_data(asset, start_date='2025-07-01')
            if not returns_data.empty:
                print(f"{asset} Returns Analysis (July-August 2025):")
                print(f"  Days: {len(returns_data)}")
                print(f"  Avg Daily Return: {returns_data['return_1d'].mean()*100:.2f}%")
                print(f"  Daily Volatility: {returns_data['return_1d'].std()*100:.2f}%")
                print(f"  Max Daily Move: {returns_data['return_1d'].abs().max()*100:.2f}%")
                
                if 'volatility_30d' in returns_data.columns:
                    current_vol = returns_data['volatility_30d'].iloc[-1]
                    print(f"  Current 30d Vol: {current_vol:.1f}%")
                
                print()
        except FileNotFoundError:
            print(f"{asset}: Limited returns data available")


def demo_correlation_analysis():
    """Demonstrate BTC-ETH correlation analysis."""
    print("=== BTC-ETH CORRELATION ===\n")
    
    try:
        corr_data = price_loader.get_correlation_data(start_date='2025-07-01')
        
        if not corr_data.empty and 'BTC_return' in corr_data.columns and 'ETH_return' in corr_data.columns:
            # Calculate correlation
            valid_data = corr_data[['BTC_return', 'ETH_return']].dropna()
            
            if len(valid_data) > 1:
                correlation = valid_data['BTC_return'].corr(valid_data['ETH_return'])
                print(f"BTC-ETH Return Correlation: {correlation:.3f}")
                
                # Recent volatility comparison
                if 'BTC_vol' in corr_data.columns and 'ETH_vol' in corr_data.columns:
                    latest_btc_vol = corr_data['BTC_vol'].dropna().iloc[-1] if not corr_data['BTC_vol'].dropna().empty else None
                    latest_eth_vol = corr_data['ETH_vol'].dropna().iloc[-1] if not corr_data['ETH_vol'].dropna().empty else None
                    
                    if latest_btc_vol and latest_eth_vol:
                        print(f"Recent Volatilities: BTC {latest_btc_vol:.1f}%, ETH {latest_eth_vol:.1f}%")
                        print(f"Volatility Ratio (ETH/BTC): {latest_eth_vol/latest_btc_vol:.2f}x")
                
                print(f"Data points used: {len(valid_data)}")
            else:
                print("Insufficient data for correlation calculation")
        else:
            print("Limited data available for correlation analysis")
            
    except Exception as e:
        print(f"Correlation analysis error: {e}")


def demo_data_structure():
    """Show the comprehensive data structure."""
    print("\n=== COMPREHENSIVE DATA STRUCTURE ===\n")
    
    try:
        # Load a small sample to show structure
        btc_data = price_loader.load_asset_data('BTC')
        if not btc_data.empty:
            latest = btc_data.iloc[-1]
            
            print("Sample BTC record (latest):")
            print(json.dumps({
                'date': latest['date'].strftime('%Y-%m-%d'),
                'basic_ohlcv': {
                    'open': float(latest['open']),
                    'high': float(latest['high']),
                    'low': float(latest['low']),
                    'close': float(latest['close']),
                    'volume': float(latest.get('volume', 0))
                },
                'returns': {
                    'return_1d': float(latest.get('return_1d', 0)) * 100,
                    'return_1d_bps': float(latest.get('return_1d_bps', 0)),
                    'return_overnight': float(latest.get('return_overnight', 0)) * 100,
                    'return_intraday': float(latest.get('return_intraday', 0)) * 100
                },
                'volatility': {
                    'volatility_7d': float(latest.get('volatility_7d', 0)),
                    'volatility_14d': float(latest.get('volatility_14d', 0)),
                    'volatility_30d': float(latest.get('volatility_30d', 0))
                },
                'options_metrics': {
                    'implied_move_1d': float(latest.get('implied_move_1d', 0)),
                    'support_level': float(latest.get('support_level', 0)),
                    'resistance_level': float(latest.get('resistance_level', 0)),
                    'distance_to_support': float(latest.get('distance_to_support', 0)),
                    'distance_to_resistance': float(latest.get('distance_to_resistance', 0))
                },
                'technical_indicators': {
                    'sma_7': float(latest.get('sma_7', 0)),
                    'sma_14': float(latest.get('sma_14', 0)),
                    'sma_30': float(latest.get('sma_30', 0)),
                    'price_percentile_30d': float(latest.get('price_percentile_30d', 0))
                }
            }, indent=2))
            
    except Exception as e:
        print(f"Error showing data structure: {e}")


def main():
    """Run all demonstrations."""
    print("🚀 COMPREHENSIVE PRICE DATA DEMO")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_specific_date_lookup()
        demo_options_context()
        demo_returns_analysis()
        demo_correlation_analysis()
        demo_data_structure()
        
        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("\n📊 Key Features:")
        print("• Comprehensive OHLCV data with 40+ calculated metrics")
        print("• Options-relevant context (strikes, volatility, support/resistance)")
        print("• Multiple volatility timeframes (7d, 14d, 30d)")
        print("• Technical indicators (SMA, percentiles, momentum)")
        print("• Returns analysis (overnight, intraday, log returns)")
        print("• Easy integration with existing options analysis systems")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
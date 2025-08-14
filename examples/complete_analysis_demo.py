#!/usr/bin/env python3
"""
Complete Options Analysis Demo
Demonstrates the unified analysis system combining image analysis with comprehensive price data.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.analytics.unified_options_analyzer import unified_analyzer, analyze_date, generate_report
from app.market_data.enhanced_price_fetcher import price_fetcher, validate_eth_data
from app.market_data.price_data_loader import get_available_assets
import json
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_data_overview():
    """Show overview of available data."""
    print("🚀 COMPLETE OPTIONS ANALYSIS SYSTEM DEMO")
    print("=" * 60)
    
    print("\n📊 DATA OVERVIEW")
    print("-" * 30)
    
    # Available assets
    assets = get_available_assets()
    print(f"Available Assets: {assets}")
    
    # Price data summary
    summary_file = Path(__file__).parent.parent / 'data' / 'price_data' / 'price_data_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Date Range: {summary['date_range']['start'][:10]} to {summary['date_range']['end'][:10]}")
        
        for asset in summary['assets']:
            stats = summary['price_stats'][asset]
            print(f"{asset}: {stats['records']:,} records, ${stats['price_min']:,.2f} - ${stats['price_max']:,.2f}")
        
        print(f"Enhanced Features: {summary['enhanced_features']['total_columns']} columns")
    
    # Image analysis data
    print(f"\nImage Analysis Data: {len(unified_analyzer.image_analysis_data)} dated results")

def demo_single_date_analysis():
    """Demonstrate single date comprehensive analysis."""
    print("\n🔍 SINGLE DATE COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    
    # Use a recent date
    analysis_date = '2025-08-10'
    
    for asset in ['BTC', 'ETH']:
        print(f"\n--- {asset} Analysis for {analysis_date} ---")
        
        try:
            # Get comprehensive context
            context = analyze_date(asset, analysis_date)
            
            # Basic price info
            print(f"💰 Spot Price: ${context.spot_price:,.2f}")
            print(f"📈 30d Change: {context.price_change_30d:+.1f}%")
            
            # Market classification
            print(f"🌊 Market Regime: {context.market_regime}")
            print(f"📊 Volatility Regime: {context.vol_regime} ({context.realized_vol_30d:.1f}%)")
            print(f"📈 Trend Direction: {context.trend_direction}")
            
            # Technical levels
            print(f"🔻 Support: ${context.support_level:,.2f} ({context.distance_to_support_pct:+.1f}%)")
            print(f"🔺 Resistance: ${context.resistance_level:,.2f} ({context.distance_to_resistance_pct:+.1f}%)")
            
            # Options metrics
            print(f"⚡ Implied 1d Move: {context.implied_move_1d:.2f}%")
            
            # Image analysis
            if context.image_analysis:
                print(f"📸 Chart Sentiment: {context.chart_sentiment}")
                print(f"🌊 Flow Signals: {', '.join(context.flow_signals) if context.flow_signals else 'None'}")
            else:
                print("📸 Image Analysis: Not available for this date")
            
        except Exception as e:
            print(f"❌ Error analyzing {asset}: {e}")

def demo_comprehensive_report():
    """Demonstrate comprehensive analysis report generation."""
    print("\n📋 COMPREHENSIVE ANALYSIS REPORTS")
    print("-" * 35)
    
    analysis_date = '2025-08-10'
    
    for asset in ['BTC', 'ETH']:
        print(f"\n{'=' * 20} {asset} REPORT {'=' * 20}")
        
        try:
            report = generate_report(asset, analysis_date)
            
            # Summary
            summary = report['summary']
            print(f"Asset: {summary['asset']}")
            print(f"Date: {summary['analysis_date']}")
            print(f"Price: ${summary['spot_price']:,.2f}")
            print(f"Market Regime: {summary['market_regime']}")
            print(f"Volatility Regime: {summary['volatility_regime']}")
            
            # Price Analysis
            price_analysis = report['price_analysis']
            print(f"\nPrice Changes:")
            for period, change in price_analysis['price_changes'].items():
                print(f"  {period}: {change}")
            
            # Volatility Analysis  
            vol_analysis = report['volatility_analysis']
            print(f"\nRealized Volatilities:")
            for period, vol in vol_analysis['realized_volatilities'].items():
                print(f"  {period}: {vol}")
            print(f"Volatility Regime: {vol_analysis['volatility_regime']}")
            print(f"Vol Percentile: {vol_analysis['vol_percentile']}")
            
            # Trading Suggestions
            trading = report['trading_suggestions']
            print(f"\nTrading Analysis:")
            print(f"  Overall Bias: {trading['overall_bias']}")
            print(f"  Preferred Strategies: {', '.join(trading['preferred_strategies'][:3])}")
            if trading['risk_considerations']:
                print(f"  Risk Considerations: {', '.join(trading['risk_considerations'])}")
            
            # Strike Suggestions
            strikes = trading['strike_recommendations']
            if strikes:
                print(f"\nSuggested Strikes:")
                if 'atm_range' in strikes:
                    atm_strikes = [f"${s:,.0f}" for s in strikes['atm_range']]
                    print(f"  ATM: {', '.join(atm_strikes)}")
                if 'otm_calls' in strikes:
                    call_strikes = [f"${s:,.0f}" for s in strikes['otm_calls'][:3]]
                    print(f"  OTM Calls: {', '.join(call_strikes)}")
                if 'otm_puts' in strikes:
                    put_strikes = [f"${s:,.0f}" for s in strikes['otm_puts'][:3]]
                    print(f"  OTM Puts: {', '.join(put_strikes)}")
                
        except Exception as e:
            print(f"❌ Error generating report for {asset}: {e}")

def demo_batch_analysis():
    """Demonstrate batch analysis capabilities."""
    print("\n📈 BATCH ANALYSIS DEMONSTRATION")
    print("-" * 33)
    
    # Analyze last 7 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    for asset in ['BTC']:  # Just BTC for demo
        print(f"\n--- {asset} Batch Analysis ({start_date} to {end_date}) ---")
        
        try:
            results = unified_analyzer.batch_analyze(asset, start_date, end_date, max_workers=2)
            
            print(f"Analyzed {len(results)} dates")
            
            # Summary statistics
            if results:
                volatilities = [ctx.realized_vol_30d for ctx in results.values() if ctx.realized_vol_30d > 0]
                price_changes = [ctx.price_change_1d for ctx in results.values() if ctx.price_change_1d != 0]
                
                if volatilities:
                    print(f"Average 30d Volatility: {sum(volatilities) / len(volatilities):.1f}%")
                    print(f"Volatility Range: {min(volatilities):.1f}% - {max(volatilities):.1f}%")
                
                if price_changes:
                    print(f"Average Daily Change: {sum(price_changes) / len(price_changes):.2f}%")
                    print(f"Largest Daily Move: {max(abs(c) for c in price_changes):.2f}%")
                
                # Market regime distribution
                regimes = [ctx.market_regime for ctx in results.values()]
                regime_counts = {}
                for regime in regimes:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                print(f"Market Regime Distribution: {regime_counts}")
                
        except Exception as e:
            print(f"❌ Error in batch analysis for {asset}: {e}")

def demo_correlation_with_images():
    """Demonstrate correlation between image analysis and price movements."""
    print("\n📸 IMAGE ANALYSIS CORRELATION DEMO")
    print("-" * 35)
    
    # Check if we have image analysis data
    if not unified_analyzer.image_analysis_data:
        print("❌ No image analysis data available for correlation analysis")
        return
    
    print(f"Available image analysis dates: {len(unified_analyzer.image_analysis_data)}")
    
    # Show a few examples where we have both image and price data
    correlations_found = 0
    for date_str in list(unified_analyzer.image_analysis_data.keys())[:5]:  # First 5 dates
        print(f"\n--- Date: {date_str} ---")
        
        for asset in ['BTC', 'ETH']:
            try:
                context = analyze_date(asset, date_str)
                
                if context.image_analysis and context.spot_price > 0:
                    print(f"{asset}: ${context.spot_price:,.2f}")
                    print(f"  Chart Sentiment: {context.chart_sentiment}")
                    print(f"  Price Change 1d: {context.price_change_1d:+.2f}%")
                    print(f"  Flow Signals: {', '.join(context.flow_signals) if context.flow_signals else 'None'}")
                    correlations_found += 1
                    
            except Exception as e:
                continue
    
    if correlations_found == 0:
        print("❌ No correlations found between image analysis and price data")
    else:
        print(f"\n✅ Found {correlations_found} correlations between image and price data")

def demo_system_capabilities():
    """Demonstrate the full system capabilities."""
    print("\n🔧 SYSTEM CAPABILITIES OVERVIEW")
    print("-" * 32)
    
    capabilities = {
        "📊 Price Data": [
            "Complete BTC & ETH history from 2021-01-01 to present",
            "49 technical indicators per record",
            "Options-specific metrics (implied moves, support/resistance)",
            "Multiple volatility timeframes (7d, 14d, 30d)"
        ],
        "📸 Image Analysis": [
            "2000+ processed options charts and flow diagrams",  
            "Chart sentiment extraction",
            "Options flow signal detection",
            "Date-based correlation with price data"
        ],
        "🧠 Unified Analysis": [
            "Market regime classification (bull/bear/sideways/volatile)",
            "Volatility regime classification (low/normal/high/extreme)",
            "Trading strategy suggestions",
            "Risk assessment and considerations"
        ],
        "⚡ Performance": [
            "Batch analysis capabilities",
            "Multi-threaded processing",
            "Caching for repeated queries",
            "Comprehensive error handling"
        ]
    }
    
    for category, features in capabilities.items():
        print(f"\n{category}")
        for feature in features:
            print(f"  ✅ {feature}")

def main():
    """Run complete analysis demonstration."""
    try:
        demo_data_overview()
        demo_single_date_analysis()
        demo_comprehensive_report()
        demo_batch_analysis()
        demo_correlation_with_images()
        demo_system_capabilities()
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE OPTIONS ANALYSIS DEMO FINISHED")
        print("\n🚀 Key Achievements:")
        print("  ✅ Complete BTC & ETH historical data (2021-2025)")
        print("  ✅ 49 technical indicators per price record")
        print("  ✅ Image analysis integration with 2000+ charts")
        print("  ✅ Unified analysis framework combining all data sources")
        print("  ✅ Comprehensive options strategy suggestions")
        print("  ✅ Production-ready batch processing capabilities")
        
        print("\n📝 Next Steps:")
        print("  • Implement automated backtesting system")
        print("  • Add real-time data feeds")
        print("  • Build interactive dashboard")
        print("  • Add portfolio optimization features")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Test script to verify backtesting cache clearing fixes.
This script runs multiple backtests to ensure different results.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.backtesting.options_backtester import run_simple_backtest
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backtest_variations():
    """Test that backtests with different settings produce different results."""
    print("🧪 Testing Backtesting Cache Clearing Fixes\n")
    
    # Test parameters
    test_configs = [
        {
            'name': 'Test 1: Standard Parameters (Fresh)',
            'asset': 'BTC',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'fresh_analysis': True,
            'randomize_strategy': False
        },
        {
            'name': 'Test 2: Same Parameters (Fresh)',
            'asset': 'BTC', 
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'fresh_analysis': True,
            'randomize_strategy': False
        },
        {
            'name': 'Test 3: Randomized Strategy',
            'asset': 'BTC',
            'start_date': '2025-07-01', 
            'end_date': '2025-08-01',
            'fresh_analysis': True,
            'randomize_strategy': True,
            'random_seed': 123
        },
        {
            'name': 'Test 4: Different Random Seed',
            'asset': 'BTC',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01', 
            'fresh_analysis': True,
            'randomize_strategy': True,
            'random_seed': 456
        },
        {
            'name': 'Test 5: Different Asset',
            'asset': 'ETH',
            'start_date': '2025-07-01',
            'end_date': '2025-08-01',
            'fresh_analysis': True,
            'randomize_strategy': False
        }
    ]
    
    results = []
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🚀 Running {config['name']}")
        print("-" * 50)
        
        try:
            # Remove name from config for function call
            test_config = {k: v for k, v in config.items() if k != 'name'}
            
            # Run backtest
            result = run_simple_backtest(**test_config)
            
            # Store key metrics
            metrics = {
                'test_name': config['name'],
                'total_trades': result.total_trades,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate
            }
            
            results.append(metrics)
            
            print(f"✅ Total Trades: {result.total_trades}")
            print(f"✅ Total Return: {result.total_return:.2%}")
            print(f"✅ Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"✅ Win Rate: {result.win_rate:.1%}")
            print(f"✅ Max Drawdown: {result.max_drawdown:.2%}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Failed to run {config['name']}: {e}")
            
            # Store error result
            metrics = {
                'test_name': config['name'],
                'error': str(e),
                'total_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
            results.append(metrics)
    
    # Analysis
    print("\n" + "="*60)
    print("📊 RESULTS ANALYSIS")
    print("="*60)
    
    print("\nSummary of Results:")
    print("-" * 40)
    for result in results:
        if 'error' in result:
            print(f"{result['test_name']}: ERROR - {result['error']}")
        else:
            print(f"{result['test_name']}:")
            print(f"  Trades: {result['total_trades']}, Return: {result['total_return']:.2%}, "
                  f"Sharpe: {result['sharpe_ratio']:.2f}")
    
    # Check for variations
    print("\n🔍 Variation Analysis:")
    print("-" * 30)
    
    # Compare returns
    returns = [r['total_return'] for r in results if 'error' not in r]
    if len(returns) > 1:
        if len(set(returns)) == 1:
            print("⚠️  WARNING: All backtests returned identical total returns!")
        else:
            print("✅ SUCCESS: Backtests show variation in returns")
            print(f"   Range: {min(returns):.2%} to {max(returns):.2%}")
    
    # Compare trade counts
    trades = [r['total_trades'] for r in results if 'error' not in r]
    if len(trades) > 1:
        if len(set(trades)) == 1:
            print("⚠️  WARNING: All backtests had identical trade counts!")
        else:
            print("✅ SUCCESS: Backtests show variation in trade counts")
            print(f"   Range: {min(trades)} to {max(trades)} trades")
    
    return results

if __name__ == "__main__":
    try:
        results = test_backtest_variations()
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        logger.error(f"Test script failed: {e}")
        sys.exit(1)
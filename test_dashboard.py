#!/usr/bin/env python3
"""
Test script to verify dashboard components work correctly.
"""

import sys
from pathlib import Path

# Add dashboard directory to path
dashboard_path = Path(__file__).parent / 'dashboard'
sys.path.insert(0, str(dashboard_path))
sys.path.insert(0, str(Path(__file__).parent))

def test_data_processor():
    """Test the DataProcessor class."""
    print("Testing DataProcessor...")
    
    try:
        from dashboard.utils.data_processor import DataProcessor
        
        processor = DataProcessor()
        print("✅ DataProcessor initialized successfully")
        
        # Check if required paths exist
        required_files = [
            processor.base_path / 'scraped_data' / 'playwright' / 'unified_articles_complete.json',
            processor.base_path / 'data' / 'price_data' / 'combined_daily_prices.csv'
        ]
        
        for file_path in required_files:
            if file_path.exists():
                print(f"✅ Found required file: {file_path.name}")
            else:
                print(f"❌ Missing required file: {file_path}")
                return False
        
        # Test data loading
        processor.load_all_data()
        print(f"✅ Loaded {len(processor.articles)} articles")
        print(f"✅ Loaded {len(processor.price_data)} price records")
        
        return True
        
    except Exception as e:
        print(f"❌ DataProcessor test failed: {e}")
        return False

def test_analysis_engine():
    """Test the AnalysisEngine class."""
    print("\nTesting AnalysisEngine...")
    
    try:
        from dashboard.utils.data_processor import DataProcessor
        from dashboard.utils.analysis_engine import AnalysisEngine
        
        processor = DataProcessor()
        processor.load_all_data()
        
        engine = AnalysisEngine(processor)
        print("✅ AnalysisEngine initialized successfully")
        
        # Test basic functionality
        filters = {'assets': ['BTC', 'ETH'], 'themes': ['volatility']}
        engine.apply_filters(filters)
        
        avg_confidence = engine.get_average_confidence()
        print(f"✅ Average confidence: {avg_confidence:.2%}")
        
        # Debug: Check articles_df structure
        print(f"✅ Articles DF columns: {list(processor.articles_df.columns)}")
        
        performance_summary = engine.get_performance_summary(filters)
        print("✅ Performance summary generated")
        
        return True
        
    except Exception as e:
        print(f"❌ AnalysisEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_utils():
    """Test the VisualizationUtils class."""
    print("\nTesting VisualizationUtils...")
    
    try:
        from dashboard.utils.visualization_utils import VisualizationUtils
        import pandas as pd
        import numpy as np
        
        # Test basic chart creation
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'value1': np.random.randn(100).cumsum(),
            'value2': np.random.randn(100).cumsum()
        })
        
        fig = VisualizationUtils.create_time_series_chart(
            test_data, 'date', ['value1', 'value2'], "Test Chart"
        )
        print("✅ Time series chart created successfully")
        
        # Test performance bar chart
        test_perf_data = {'Theme A': 0.05, 'Theme B': -0.02, 'Theme C': 0.03}
        perf_fig = VisualizationUtils.create_performance_bar_chart(test_perf_data, "Test Performance")
        print("✅ Performance bar chart created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ VisualizationUtils test failed: {e}")
        return False

def test_config():
    """Test the DashboardConfig class."""
    print("\nTesting DashboardConfig...")
    
    try:
        from dashboard.config.dashboard_config import DashboardConfig
        
        config = DashboardConfig()
        print("✅ DashboardConfig initialized successfully")
        
        # Test configuration values
        print(f"✅ Dashboard title: {config.DASHBOARD_TITLE}")
        print(f"✅ Available themes: {len(config.AVAILABLE_THEMES)}")
        print(f"✅ Time horizons: {config.TIME_HORIZONS}")
        
        # Test validation (may fail if paths don't exist)
        try:
            is_valid = config.validate_config()
            if is_valid:
                print("✅ Configuration validation passed")
            else:
                print("⚠️ Configuration validation failed (expected if data files missing)")
        except:
            print("⚠️ Configuration validation could not run")
        
        return True
        
    except Exception as e:
        print(f"❌ DashboardConfig test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Dashboard Components")
    print("=" * 50)
    
    tests = [
        test_config,
        test_data_processor,
        test_analysis_engine,
        test_visualization_utils
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should be ready to run.")
        print("\nTo start the dashboard, run:")
        print("streamlit run dashboard/streamlit_dashboard.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("The dashboard may not work correctly until issues are resolved.")

if __name__ == "__main__":
    main()
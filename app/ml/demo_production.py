#!/usr/bin/env python3
"""
Production FlowScore Pipeline Demo

Demonstrates the complete Enhanced PRD Phase 2: Intelligence system
with all components working together in a production-ready pipeline.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import logging
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.multimodal_scorer import multimodal_scorer
from app.core.logging import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo production FlowScore pipeline")
    parser.add_argument(
        "--cleaned-json",
        default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()),
        help="Path to cleaned articles JSON"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=15,
        help="Number of articles to process"
    )
    parser.add_argument(
        "--assets",
        default="BTC,ETH",
        help="Comma-separated assets to score"
    )
    parser.add_argument(
        "--concurrent", 
        type=int, 
        default=3,
        help="Maximum concurrent processing"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("test_results").resolve()),
        help="Output directory for results"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def load_articles(json_path: str, limit: int = None) -> List[Dict]:
    """Load cleaned articles from JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        if limit:
            articles = articles[:limit]
        
        logger.info(f"Loaded {len(articles)} articles for processing")
        return articles
        
    except Exception as e:
        logger.error(f"Failed to load articles: {e}")
        raise


def analyze_production_results(results: List[Dict], assets: List[str]) -> Dict:
    """Analyze production pipeline results."""
    analysis = {
        'summary': {
            'total_articles': len(results),
            'processing_timestamp': datetime.now().isoformat(),
            'assets_analyzed': assets
        },
        'flowscore_analysis': {},
        'quality_metrics': {},
        'component_performance': {}
    }
    
    # Collect FlowScores by asset
    asset_flowscores = {asset: [] for asset in assets}
    asset_confidences = {asset: [] for asset in assets}
    
    text_confidences = []
    image_confidences = []
    successful_articles = 0
    
    for result in results:
        article_data = result.get('article_data', {})
        processing_results = result.get('processing_results', {})
        
        article_successful = False
        
        for asset in assets:
            asset_result = processing_results.get(asset, {})
            multimodal_score = asset_result.get('multimodal_score', {})
            
            flowscore = multimodal_score.get('final_flowscore', 0.0)
            confidence = multimodal_score.get('overall_confidence', 0.0)
            
            if confidence > 0.2:  # Only count meaningful scores
                asset_flowscores[asset].append(flowscore)
                asset_confidences[asset].append(confidence)
                article_successful = True
            
            # Collect component confidences
            text_conf = multimodal_score.get('text_confidence', 0.0)
            image_conf = multimodal_score.get('image_confidence', 0.0)
            
            if text_conf > 0:
                text_confidences.append(text_conf)
            if image_conf > 0:
                image_confidences.append(image_conf)
        
        if article_successful:
            successful_articles += 1
    
    # Calculate asset statistics
    analysis['summary']['successful_articles'] = successful_articles
    analysis['summary']['success_rate'] = (
        successful_articles / len(results) * 100 if results else 0
    )
    
    
    for asset in assets:
        flowscores = asset_flowscores[asset]
        confidences = asset_confidences[asset]
        
        if flowscores:
            analysis['flowscore_analysis'][asset] = {
                'total_scores': len(flowscores),
                'mean_flowscore': float(np.mean(flowscores)),
                'std_flowscore': float(np.std(flowscores)),
                'min_flowscore': float(np.min(flowscores)),
                'max_flowscore': float(np.max(flowscores)),
                'mean_confidence': float(np.mean(confidences)),
                'distribution': {
                    'strong_bullish': sum(1 for s in flowscores if s > 0.4),
                    'moderate_bullish': sum(1 for s in flowscores if 0.1 < s <= 0.4),
                    'neutral': sum(1 for s in flowscores if -0.1 <= s <= 0.1),
                    'moderate_bearish': sum(1 for s in flowscores if -0.4 <= s < -0.1),
                    'strong_bearish': sum(1 for s in flowscores if s < -0.4)
                }
            }
    
    # Component performance
    if text_confidences:
        analysis['component_performance']['text'] = {
            'mean_confidence': float(np.mean(text_confidences)),
            'samples': len(text_confidences)
        }
    
    if image_confidences:
        analysis['component_performance']['image'] = {
            'mean_confidence': float(np.mean(image_confidences)),
            'samples': len(image_confidences)
        }
    
    return analysis


def print_demo_summary(analysis: Dict, performance_summary: Dict):
    """Print comprehensive demo summary."""
    print("\n" + "="*60)
    print("🚀 PRODUCTION FLOWSCORE PIPELINE DEMO RESULTS")
    print("="*60)
    
    # Summary
    summary = analysis['summary']
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Articles Processed: {summary['total_articles']}")
    print(f"   Successful Articles: {summary['successful_articles']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Assets Analyzed: {', '.join(summary['assets_analyzed'])}")
    
    # Performance Metrics
    perf = performance_summary.get('performance_metrics', {})
    print(f"\n⚡ PERFORMANCE METRICS:")
    print(f"   Processing Speed: {perf.get('articles_per_second', 0):.2f} articles/second")
    print(f"   Total Processing Time: {perf.get('processing_time_seconds', 0):.1f} seconds")
    print(f"   Text Extractions: {perf.get('text_extractions_completed', 0)}")
    print(f"   Score Generation Success: {perf.get('success_rate', 0):.1f}%")
    
    # Quality Metrics
    quality = performance_summary.get('quality_metrics', {})
    print(f"\n🎯 QUALITY METRICS:")
    print(f"   Average Text Confidence: {quality.get('average_text_confidence', 0):.3f}")
    print(f"   Average Image Confidence: {quality.get('average_image_confidence', 0):.3f}")
    print(f"   Average Multimodal Confidence: {quality.get('average_multimodal_confidence', 0):.3f}")
    
    # FlowScore Analysis
    flowscore_analysis = analysis.get('flowscore_analysis', {})
    print(f"\n💹 FLOWSCORE ANALYSIS:")
    for asset, stats in flowscore_analysis.items():
        print(f"   {asset}:")
        print(f"      Mean FlowScore: {stats.get('mean_flowscore', 0):.3f}")
        print(f"      Score Range: [{stats.get('min_flowscore', 0):.3f}, {stats.get('max_flowscore', 0):.3f}]")
        print(f"      Mean Confidence: {stats.get('mean_confidence', 0):.3f}")
        
        dist = stats.get('distribution', {})
        total_scores = stats.get('total_scores', 0)
        if total_scores > 0:
            print(f"      Distribution:")
            print(f"         Strong Bullish: {dist.get('strong_bullish', 0)} ({dist.get('strong_bullish', 0)/total_scores:.1%})")
            print(f"         Moderate Bullish: {dist.get('moderate_bullish', 0)} ({dist.get('moderate_bullish', 0)/total_scores:.1%})")
            print(f"         Neutral: {dist.get('neutral', 0)} ({dist.get('neutral', 0)/total_scores:.1%})")
            print(f"         Moderate Bearish: {dist.get('moderate_bearish', 0)} ({dist.get('moderate_bearish', 0)/total_scores:.1%})")
            print(f"         Strong Bearish: {dist.get('strong_bearish', 0)} ({dist.get('strong_bearish', 0)/total_scores:.1%})")
    
    # System Info
    system = performance_summary.get('system_info', {})
    print(f"\n🔧 SYSTEM INFO:")
    print(f"   Pipeline Version: {system.get('pipeline_version')}")
    print(f"   Enhanced PRD Phase: {system.get('enhanced_prd_phase')}")
    print(f"   Active Components: {', '.join(system.get('components_active', []))}")
    
    print("\n" + "="*60)
    print("✅ Enhanced PRD Phase 2: Intelligence - DEMONSTRATION COMPLETE")
    print("="*60)


def save_demo_results(results: List[Dict], analysis: Dict, performance: Dict, output_dir: str):
    """Save comprehensive demo results."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        complete_results = {
            'demo_metadata': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'ProductionFlowScorer_v1.0',
                'enhanced_prd_phase': 'Phase 2: Intelligence',
                'demo_description': 'Complete multimodal FlowScore pipeline demonstration'
            },
            'analysis': analysis,
            'performance_summary': performance,
            'detailed_results': results
        }
        
        results_file = output_path / 'production_demo_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Complete demo results saved to {results_file}")
        
        # Save summary report
        summary_file = output_path / 'production_demo_summary.json'
        summary_data = {
            'summary': analysis.get('summary'),
            'performance_metrics': performance.get('performance_metrics'),
            'quality_metrics': performance.get('quality_metrics'),
            'flowscore_analysis': analysis.get('flowscore_analysis'),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Demo summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to save demo results: {e}")
        raise


async def main_async(args: argparse.Namespace):
    """Main demo execution."""
    # Parse assets
    assets = [asset.strip() for asset in args.assets.split(',')]
    
    # Load articles
    articles = load_articles(args.cleaned_json, args.limit)
    if not articles:
        logger.error("No articles to process")
        return False
    
    print(f"\n🚀 Starting Production FlowScore Pipeline Demo")
    print(f"   Articles to process: {len(articles)}")
    print(f"   Assets to score: {', '.join(assets)}")
    print(f"   Max concurrent: {args.concurrent}")
    print(f"   Enhanced PRD Phase: 2 - Intelligence")
    
    # Reset scorer statistics for clean demo
    multimodal_scorer.reset_statistics()
    
    # Run production pipeline
    start_time = time.time()
    results = await multimodal_scorer.process_batch(
        articles, 
        assets, 
        max_concurrent=args.concurrent,
        include_images=True
    )
    processing_time = time.time() - start_time
    
    # Get performance summary
    performance_summary = multimodal_scorer.get_performance_summary()
    
    # Analyze results
    analysis = analyze_production_results(results, assets)
    
    # Print comprehensive summary
    print_demo_summary(analysis, performance_summary)
    
    # Save results
    save_demo_results(results, analysis, performance_summary, args.output_dir)
    
    return len(results) > 0


def main():
    """Main demo CLI."""
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        success = asyncio.run(main_async(args))
        return success
        
    except Exception as e:
        logger.error(f"Production demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
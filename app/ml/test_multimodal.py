#!/usr/bin/env python3
"""
Multimodal Scoring Test CLI

Test the multimodal fusion scoring system on cleaned articles data.
This validates the Enhanced PRD Phase 2: Intelligence - Multimodal Fusion component.
"""

import argparse
import asyncio
import json
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.multimodal_scorer import multimodal_scorer, MultimodalScoreComponents
from app.core.logging import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test multimodal scoring system")
    parser.add_argument(
        "--cleaned-json",
        default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()),
        help="Path to cleaned articles JSON"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10,
        help="Number of articles to test"
    )
    parser.add_argument(
        "--assets",
        default="BTC,ETH",
        help="Comma-separated list of assets to score"
    )
    parser.add_argument(
        "--output-file",
        default=str(Path("test_results/multimodal_scores.json").resolve()),
        help="Output file for detailed results"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=5,
        help="Batch size for concurrent processing"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


class MultimodalTester:
    """Test harness for multimodal scoring system."""
    
    def __init__(self, assets: List[str]):
        self.assets = assets
        self.scorer = multimodal_scorer
        self.results = []
        
    def load_articles(self, json_path: str, limit: Optional[int] = None) -> List[Dict]:
        """Load cleaned articles from JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            if limit:
                articles = articles[:limit]
            
            logger.info(f"Loaded {len(articles)} articles from {json_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to load articles: {e}")
            raise
    
    async def score_article(self, article: Dict) -> Dict[str, MultimodalScoreComponents]:
        """Score a single article for all assets."""
        article_url = article.get('url', 'unknown')
        article_scores = {}
        
        try:
            # Score for each asset
            scoring_tasks = [
                self.scorer.calculate_multimodal_score(article, asset)
                for asset in self.assets
            ]
            
            scores = await asyncio.gather(*scoring_tasks, return_exceptions=True)
            
            for i, asset in enumerate(self.assets):
                if isinstance(scores[i], Exception):
                    logger.error(f"Scoring failed for {asset} on {article_url}: {scores[i]}")
                    # Create neutral score for failed asset
                    article_scores[asset] = self.scorer._create_neutral_score(
                        asset, article_url, f"Scoring error: {scores[i]}"
                    )
                else:
                    article_scores[asset] = scores[i]
            
            return article_scores
            
        except Exception as e:
            logger.error(f"Article scoring failed: {article_url}: {e}")
            # Return neutral scores for all assets
            return {
                asset: self.scorer._create_neutral_score(asset, article_url, f"Article error: {e}")
                for asset in self.assets
            }
    
    async def run_batch_test(self, articles: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Run multimodal scoring test on batch of articles."""
        logger.info(f"Starting multimodal scoring test on {len(articles)} articles")
        start_time = time.time()
        
        all_results = []
        
        # Process in batches to control memory and API usage
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(articles) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} articles")
            
            # Score batch concurrently
            batch_tasks = [self.score_article(article) for article in batch]
            batch_scores = await asyncio.gather(*batch_tasks)
            
            # Process batch results
            for j, article_scores in enumerate(batch_scores):
                article = batch[j]
                
                result = {
                    'article_url': article.get('url'),
                    'article_title': article.get('title'),
                    'published_at': article.get('published_at_utc'),
                    'scores': {
                        asset: self._score_to_dict(score_components)
                        for asset, score_components in article_scores.items()
                    },
                    'processing_batch': batch_num,
                    'processing_index': i + j
                }
                
                all_results.append(result)
            
            logger.info(f"Batch {batch_num} completed successfully")
        
        processing_time = time.time() - start_time
        logger.info(f"Multimodal scoring completed in {processing_time:.1f} seconds")
        
        return all_results
    
    def _score_to_dict(self, score_components: MultimodalScoreComponents) -> Dict:
        """Convert unknown score component types to dictionaries or raise clearly."""
        from dataclasses import is_dataclass, asdict as dc_asdict
        # Already a dict
        if isinstance(score_components, dict):
            return score_components
        # Dataclass
        if is_dataclass(score_components):
            return dc_asdict(score_components)
        # Has to_dict or dict method
        for method_name in ('to_dict', 'dict'):
            if hasattr(score_components, method_name):
                maybe = getattr(score_components, method_name)()
                if isinstance(maybe, dict):
                    return maybe
        # Has __dict__
        if hasattr(score_components, '__dict__'):
            return dict(score_components.__dict__)
        # Unknown type
        raise TypeError(f"Unsupported score component type: {type(score_components)!r}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze and summarize multimodal scoring results."""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Collect all scores by asset
        asset_scores = {asset: [] for asset in self.assets}
        asset_confidences = {asset: [] for asset in self.assets}
        component_scores = {
            'text': [], 'image': [], 'market': [], 'meta': []
        }
        
        successful_articles = 0
        total_articles = len(results)
        
        for result in results:
            scores = result.get('scores', {})
            article_successful = False
            
            for asset in self.assets:
                if asset in scores:
                    score_data = scores[asset]
                    
                    # Extract scores
                    final_score = score_data.get('final_flowscore', 0.0)
                    confidence = score_data.get('overall_confidence', 0.0)
                    
                    if confidence > 0.3:  # Only count confident scores
                        asset_scores[asset].append(final_score)
                        asset_confidences[asset].append(confidence)
                        article_successful = True
                        
                        # Collect component scores
                        component_scores['text'].append(score_data.get('text_score', 0.0))
                        component_scores['image'].append(score_data.get('image_score', 0.0))
                        component_scores['market'].append(score_data.get('market_context_score', 0.0))
                        component_scores['meta'].append(score_data.get('meta_signals_score', 0.0))
            
            if article_successful:
                successful_articles += 1
        
        # Calculate statistics
        analysis = {
            'summary': {
                'total_articles': total_articles,
                'successful_articles': successful_articles,
                'success_rate': successful_articles / total_articles * 100 if total_articles > 0 else 0,
                'processing_timestamp': datetime.now().isoformat()
            },
            'asset_statistics': {},
            'component_statistics': {},
            'distribution_analysis': {}
        }
        
        # Asset-specific statistics
        for asset in self.assets:
            scores = asset_scores[asset]
            confidences = asset_confidences[asset]
            
            if scores:
                analysis['asset_statistics'][asset] = {
                    'count': len(scores),
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'min_score': float(np.min(scores)),
                    'max_score': float(np.max(scores)),
                    'mean_confidence': float(np.mean(confidences)),
                    'bullish_ratio': sum(1 for s in scores if s > 0.1) / len(scores),
                    'bearish_ratio': sum(1 for s in scores if s < -0.1) / len(scores),
                    'neutral_ratio': sum(1 for s in scores if -0.1 <= s <= 0.1) / len(scores),
                    'high_confidence_ratio': sum(1 for c in confidences if c > 0.7) / len(confidences)
                }
        
        # Component analysis
        for component, scores in component_scores.items():
            if scores:
                analysis['component_statistics'][component] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'contribution': abs(np.mean(scores)) * multimodal_scorer.COMPONENT_WEIGHTS.get(component, 0.0)
                }
        
        # Distribution analysis
        all_final_scores = []
        for asset in self.assets:
            all_final_scores.extend(asset_scores[asset])
        
        if all_final_scores:
            analysis['distribution_analysis'] = {
                'total_scores': len(all_final_scores),
                'score_distribution': {
                    'strong_bullish': sum(1 for s in all_final_scores if s > 0.5),
                    'moderate_bullish': sum(1 for s in all_final_scores if 0.1 < s <= 0.5),
                    'neutral': sum(1 for s in all_final_scores if -0.1 <= s <= 0.1),
                    'moderate_bearish': sum(1 for s in all_final_scores if -0.5 <= s < -0.1),
                    'strong_bearish': sum(1 for s in all_final_scores if s < -0.5)
                },
                'percentiles': {
                    'p10': float(np.percentile(all_final_scores, 10)),
                    'p25': float(np.percentile(all_final_scores, 25)),
                    'p50': float(np.percentile(all_final_scores, 50)),
                    'p75': float(np.percentile(all_final_scores, 75)),
                    'p90': float(np.percentile(all_final_scores, 90))
                }
            }
        
        # Get processing statistics from scorer
        processing_stats = self.scorer.get_processing_stats()
        analysis['processing_statistics'] = processing_stats
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict):
        """Print human-readable analysis summary."""
        print("\n=== MULTIMODAL SCORING ANALYSIS ===")
        
        summary = analysis.get('summary', {})
        print(f"Total articles: {summary.get('total_articles')}")
        print(f"Successful articles: {summary.get('successful_articles')}")
        print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
        
        print("\n=== ASSET PERFORMANCE ===")
        asset_stats = analysis.get('asset_statistics', {})
        for asset, stats in asset_stats.items():
            print(f"{asset}:")
            print(f"  Mean score: {stats.get('mean_score', 0):.3f}")
            print(f"  Mean confidence: {stats.get('mean_confidence', 0):.3f}")
            print(f"  Bullish ratio: {stats.get('bullish_ratio', 0):.1%}")
            print(f"  Bearish ratio: {stats.get('bearish_ratio', 0):.1%}")
            print(f"  High confidence ratio: {stats.get('high_confidence_ratio', 0):.1%}")
        
        print("\n=== COMPONENT CONTRIBUTIONS ===")
        comp_stats = analysis.get('component_statistics', {})
        weights = multimodal_scorer.COMPONENT_WEIGHTS
        for component, stats in comp_stats.items():
            weight = weights.get(component, 0.0)
            contribution = stats.get('contribution', 0.0)
            print(f"{component.capitalize()} ({weight:.0%} weight):")
            print(f"  Mean score: {stats.get('mean', 0):.3f}")
            print(f"  Contribution: {contribution:.3f}")
        
        print("\n=== SCORE DISTRIBUTION ===")
        dist = analysis.get('distribution_analysis', {})
        if 'score_distribution' in dist:
            score_dist = dist['score_distribution']
            total = sum(score_dist.values())
            if total > 0:
                for category, count in score_dist.items():
                    print(f"{category.replace('_', ' ').title()}: {count} ({count/total:.1%})")
    
    def save_results(self, results: List[Dict], analysis: Dict, output_file: str):
        """Save detailed results and analysis to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_articles': len(results),
                    'assets_tested': self.assets,
                    'multimodal_weights': multimodal_scorer.COMPONENT_WEIGHTS
                },
                'analysis': analysis,
                'detailed_results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def main_async(args: argparse.Namespace):
    """Main async testing process."""
    # Parse assets
    assets = [asset.strip() for asset in args.assets.split(',')]
    
    # Initialize tester
    tester = MultimodalTester(assets)
    
    # Load articles
    articles = tester.load_articles(args.cleaned_json, args.limit)
    if not articles:
        logger.error("No articles to process")
        return False
    
    # Run scoring test
    results = await tester.run_batch_test(articles, args.batch_size)
    
    # Analyze results
    analysis = tester.analyze_results(results)
    
    # Print summary
    tester.print_analysis_summary(analysis)
    
    # Save detailed results
    tester.save_results(results, analysis, args.output_file)
    
    return len(results) > 0


def main():
    """Main multimodal scoring test CLI."""
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        success = asyncio.run(main_async(args))
        logger.info("Multimodal scoring test completed" + (" successfully" if success else " with issues"))
        return success
        
    except Exception as e:
        logger.error(f"Multimodal scoring test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
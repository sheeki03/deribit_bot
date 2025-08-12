#!/usr/bin/env python3
"""
Smart Resume Scraper - Continue scraping from where we left off
Avoids duplicate processing and efficiently uses GPU acceleration
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Set

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.logging import logger
from app.scrapers.bulletproof_scraper import bulletproof_scraper
from app.validation.data_validator import data_validator
from app.market_data.price_correlator import price_correlator


"""Set environment variables once at startup to configure GPU acceleration."""
os.environ.setdefault('EASYOCR_GPU', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')


class SmartResumeScraper:
    """Resume scraping from the last processed article."""
    
    def __init__(self):
        self.data_dir = Path("./scraped_data")
        self.data_dir.mkdir(exist_ok=True)
        
    def load_discovered_articles(self) -> List[Dict]:
        """Load all discovered articles."""
        discovered_file = self.data_dir / "all_126_articles_discovered.json"
        if discovered_file.exists():
            with open(discovered_file, 'r') as f:
                articles = json.load(f)
            print(f"📚 Loaded {len(articles)} discovered articles")
            return articles
        return []
    
    def load_processed_articles(self) -> List[Dict]:
        """Load already processed articles from the latest progress file."""
        # Check both old and new progress file patterns
        old_progress_files = list(self.data_dir.glob("progress_articles_*_of_*.json"))
        new_progress_files = list(self.data_dir.glob("resume_progress_*_of_*.json"))
        
        all_progress_files = old_progress_files + new_progress_files
        
        if not all_progress_files:
            print("📝 No previous progress found, starting fresh")
            return []
        
        # Find the latest progress file
        latest_file = max(all_progress_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            processed_articles = json.load(f)
        
        print(f"📈 Loaded {len(processed_articles)} already processed articles from {latest_file.name}")
        return processed_articles
    
    def get_remaining_articles(self, all_articles: List[Dict], processed_articles: List[Dict]) -> List[Dict]:
        """Get articles that still need to be processed."""
        
        # Create a set of processed URLs for fast lookup
        processed_urls = {article.get('url', '').rstrip('/') for article in processed_articles}
        
        # Filter out articles that have already been processed
        remaining_articles = []
        
        for article in all_articles:
            article_url = article.get('url', '').rstrip('/')
            
            # Skip non-article URLs (pagination, navigation, etc.)
            if any(skip in article_url for skip in ['#content', '#/', '/page/', '.jpg']):
                continue
                
            # Skip if already processed
            if article_url in processed_urls:
                continue
                
            # Only include actual option-flow articles
            if '/option-flow' in article_url:
                remaining_articles.append(article)
        
        print(f"🎯 Found {len(remaining_articles)} articles remaining to process")
        return remaining_articles
    
    async def process_remaining_articles(self, remaining_articles: List[Dict], already_processed: List[Dict]) -> Dict:
        """Process only the remaining unprocessed articles."""
        
        if not remaining_articles:
            print("✅ All articles already processed!")
            return {
                'total_target': len(already_processed),
                'successful_scrapes': len(already_processed),
                'failed_scrapes': 0,
                'total_images': sum(len(a.get('images', [])) for a in already_processed),
                'articles_processed': already_processed
            }
        
        print(f"🚀 Processing {len(remaining_articles)} remaining articles with GPU acceleration...")
        print("=" * 80)
        
        # Start with already processed articles
        all_processed = already_processed.copy()
        successful_scrapes = len(already_processed)
        failed_scrapes = 0
        total_images = sum(len(a.get('images', [])) for a in already_processed)
        
        for i, article in enumerate(remaining_articles, 1):
            current_total = successful_scrapes + i
            print(f"\n📄 [{current_total}/{len(remaining_articles) + len(already_processed)}] {article['title'][:60]}...")
            print(f"   🔗 {article['url']}")
            
            try:
                async with bulletproof_scraper:
                    content = await bulletproof_scraper.scrape_article_content(article['url'])
                
                if content:
                    # Combine metadata with content
                    full_article = {**article, **content}
                    all_processed.append(full_article)
                    successful_scrapes += 1
                    
                    # Count images
                    image_count = len(content.get('images', []))
                    total_images += image_count
                    
                    # Quick validation
                    validation = data_validator.validate_article(full_article)
                    
                    # Analyze price correlation for this article
                    print(f"   🔍 Analyzing price correlation...")
                    try:
                        price_analyses = await price_correlator.analyze_article_price_correlation(full_article)
                        full_article['price_analyses'] = price_analyses
                        
                        # Display price correlation results
                        if price_analyses:
                            for asset, analysis in price_analyses.items():
                                print(f"   📊 {asset}: {analysis.weekly_return:+.2%} weekly return, "
                                      f"accuracy: {analysis.prediction_accuracy:.2f}")
                        else:
                            print(f"   ⚠️ No price correlation data available")
                    except Exception as e:
                        print(f"   ⚠️ Price analysis failed: {str(e)[:50]}...")
                    
                    print(f"   ✅ Content: {len(content.get('body_markdown', ''))} chars")
                    print(f"   ✅ Images: {image_count} found")
                    print(f"   ✅ Quality: {validation.quality_score:.1f}/100")
                    
                    # Save progress every 10 articles
                    if i % 10 == 0:
                        await self._save_progress(all_processed, current_total, len(remaining_articles) + len(already_processed))
                
                else:
                    failed_scrapes += 1
                    print(f"   ❌ Failed to scrape content")
                
                # Rate limiting for politeness
                await asyncio.sleep(0.5)
                
            except Exception as e:
                failed_scrapes += 1
                print(f"   💥 Error: {str(e)[:100]}...")
                continue
        
        # Final save
        await self._save_final_results(all_processed, successful_scrapes, failed_scrapes, total_images)
        
        return {
            'total_target': len(remaining_articles) + len(already_processed),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': failed_scrapes,
            'total_images': total_images,
            'articles_processed': all_processed
        }
    
    async def _save_progress(self, articles_processed: List[Dict], current: int, total: int):
        """Save progress."""
        progress_file = self.data_dir / f"resume_progress_{current}_of_{total}.json"
        with open(progress_file, 'w') as f:
            json.dump(articles_processed, f, indent=2, default=str)
        print(f"   💾 Progress saved: {current}/{total} articles")
    
    async def _save_final_results(self, articles_processed: List[Dict], successful: int, failed: int, total_images: int):
        """Save final comprehensive results."""
        
        # Save all processed articles
        final_file = self.data_dir / "all_articles_final_complete.json"
        with open(final_file, 'w') as f:
            json.dump(articles_processed, f, indent=2, default=str)
        
        # Generate comprehensive statistics
        print(f"\n📊 FINAL COMPREHENSIVE SCRAPING RESULTS")
        print("=" * 80)
        print(f"✅ Successfully scraped: {successful} articles")
        print(f"❌ Failed scrapes: {failed} articles")
        print(f"📈 Success rate: {(successful/(successful+failed)*100) if (successful+failed) > 0 else 0:.1f}%")
        print(f"📸 Total images processed: {total_images}")
        
        if articles_processed:
            total_content = sum(len(a.get('body_markdown', '')) for a in articles_processed)
            print(f"📝 Total content: {total_content:,} characters")
            
            # Quality analysis
            validations = [data_validator.validate_article(a) for a in articles_processed]
            valid_articles = sum(1 for v in validations if v.is_valid)
            avg_quality = sum(v.quality_score for v in validations) / len(validations) if validations else 0
            
            print(f"✅ Valid articles: {valid_articles}/{len(articles_processed)}")
            print(f"📊 Average quality: {avg_quality:.1f}/100")
        
        print(f"\n💾 Complete dataset saved to: {final_file}")
        print(f"🎉 MISSION ACCOMPLISHED: {successful} articles with {total_images} images!")


async def main():
    """Resume scraping from where we left off."""
    scraper = SmartResumeScraper()
    
    try:
        # Load discovered articles
        all_articles = scraper.load_discovered_articles()
        if not all_articles:
            print("❌ No discovered articles found. Run the main scraper first.")
            return False
        
        # Load already processed articles
        processed_articles = scraper.load_processed_articles()
        
        # Get remaining articles to process
        remaining_articles = scraper.get_remaining_articles(all_articles, processed_articles)
        
        # Process remaining articles
        results = await scraper.process_remaining_articles(remaining_articles, processed_articles)
        
        print(f"\n{'🎉 SUCCESS' if results['successful_scrapes'] > 0 else '⚠️ COMPLETE'}: {results['successful_scrapes']} total articles processed")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Scraper crashed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
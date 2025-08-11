#!/usr/bin/env python3
"""
Complete Scraper for ALL 126 Deribit Option Flows Articles

This script uses Firecrawl to properly crawl paginated pages and discover
all 126 articles, then scrapes them with GPU-accelerated OCR processing.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Set
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.logging import logger
from app.scrapers.bulletproof_scraper import bulletproof_scraper
from app.scrapers.firecrawl_client import FirecrawlClient
from app.validation.data_validator import data_validator
from app.market_data.price_correlator import price_correlator


class ComprehensiveOptionFlowsScraper:
    """High-performance scraper for all 126 Option Flows articles."""
    
    def __init__(self):
        self.firecrawl = FirecrawlClient()
        self.discovered_urls: Set[str] = set()
        self.data_dir = Path("./scraped_data")
        self.data_dir.mkdir(exist_ok=True)
    
    async def discover_all_articles(self) -> List[Dict]:
        """Discover all 126 articles from paginated pages using Firecrawl."""
        print("🔍 Discovering ALL Option Flows articles using Firecrawl...")
        
        # Base pagination URLs to crawl
        urls_to_crawl = [
            "https://insights.deribit.com/option-flows/",
            "https://insights.deribit.com/option-flows/page/2/",
            "https://insights.deribit.com/option-flows/page/3/",
            "https://insights.deribit.com/option-flows/page/4/",
            "https://insights.deribit.com/option-flows/page/5/",
            "https://insights.deribit.com/option-flows/page/6/",
            "https://insights.deribit.com/option-flows/page/7/",
            "https://insights.deribit.com/option-flows/page/8/",
            "https://insights.deribit.com/option-flows/page/9/",
            "https://insights.deribit.com/option-flows/page/10/",
        ]
        
        all_articles = []
        
        for page_url in urls_to_crawl:
            try:
                print(f"   📄 Crawling: {page_url}")
                
                # Use Firecrawl to get the page content
                page_content = await self.firecrawl.scrape_url(page_url)
                
                if not page_content:
                    print(f"   ❌ Failed to crawl {page_url}")
                    continue
                
                # Extract article URLs from the page
                page_articles = self._extract_articles_from_page(page_content, page_url)
                
                print(f"   ✅ Found {len(page_articles)} articles on this page")
                all_articles.extend(page_articles)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   💥 Error crawling {page_url}: {e}")
                continue
        
        # Remove duplicates based on URL
        unique_articles = []
        seen_urls = set()
        
        for article in all_articles:
            if article['url'] not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(article['url'])
        
        print(f"🎯 Total unique articles discovered: {len(unique_articles)}")
        
        # Save discovered articles
        discovered_file = self.data_dir / "all_126_articles_discovered.json"
        with open(discovered_file, 'w') as f:
            json.dump(unique_articles, f, indent=2, default=str)
        
        print(f"💾 All discovered articles saved to: {discovered_file}")
        return unique_articles
    
    def _extract_articles_from_page(self, page_content: Dict, page_url: str) -> List[Dict]:
        """Extract article metadata from a page's content."""
        articles = []
        
        # Get the markdown content from Firecrawl
        content = page_content.get('data', {}).get('markdown', '') or page_content.get('markdown', '')
        
        if not content:
            return articles
        
        # Multiple patterns to extract article URLs from Firecrawl markdown
        patterns = [
            r'\[([^\]]+)\]\((https://insights\.deribit\.com/option-flows/[^)]+)\)',  # [title](url) format
            r'https://insights\.deribit\.com/option-flows/[a-zA-Z0-9-/]+',  # Direct URLs
            r'insights\.deribit\.com/option-flows/([a-zA-Z0-9-/]+)',  # Domain with path
        ]
        
        found_articles = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            
            for match in matches:
                if isinstance(match, tuple):
                    # Pattern with title and URL
                    title, url = match
                    clean_url = url.rstrip('/')
                else:
                    # Just URL
                    url = match if match.startswith('http') else f"https://insights.deribit.com/option-flows/{match}"
                    clean_url = url.rstrip('/')
                    
                    # Extract title from URL slug
                    slug = clean_url.split('/')[-1]
                    title = slug.replace('-', ' ').replace('option flow', 'Option Flow').title()
                
                # Only include if it's actually an article URL and not already found
                if '/option-flow' in clean_url and clean_url not in found_articles:
                    found_articles.add(clean_url)
                    
                    # Ensure URL ends with slash
                    if not clean_url.endswith('/'):
                        clean_url += '/'
                    
                    articles.append({
                        'url': clean_url,
                        'title': title,
                        'author': 'Tony Stewart',
                        'source': 'firecrawl_pagination',
                        'discovery_method': 'firecrawl',
                        'discovered_from_page': page_url
                    })
        
        # Also check for any line that mentions option flow articles
        lines = content.split('\n')
        for line in lines:
            if 'option-flow' in line.lower():
                # Try to extract URLs from this line
                url_matches = re.findall(r'https://[^\s)]+', line)
                for url in url_matches:
                    if 'option-flow' in url and url not in found_articles:
                        found_articles.add(url)
                        
                        slug = url.split('/')[-1]
                        title = slug.replace('-', ' ').title()
                        
                        articles.append({
                            'url': url + ('/' if not url.endswith('/') else ''),
                            'title': title,
                            'author': 'Tony Stewart',
                            'source': 'firecrawl_pagination',
                            'discovery_method': 'firecrawl',
                            'discovered_from_page': page_url
                        })
        
        return articles
    
    async def scrape_all_articles(self, articles: List[Dict]) -> Dict:
        """Scrape all articles with GPU-accelerated processing."""
        print(f"\n🚀 Scraping ALL {len(articles)} articles with GPU acceleration...")
        print("=" * 60)
        
        # Enable GPU for OCR if available
        import os
        os.environ['EASYOCR_GPU'] = '1'  # Force GPU usage for EasyOCR
        
        articles_processed = []
        successful_scrapes = 0
        failed_scrapes = 0
        total_images = 0
        
        for i, article in enumerate(articles, 1):
            print(f"\n📄 [{i}/{len(articles)}] {article['title'][:60]}...")
            print(f"   🔗 {article['url']}")
            
            try:
                async with bulletproof_scraper:
                    content = await bulletproof_scraper.scrape_article_content(article['url'])
                
                if content:
                    # Combine metadata with content
                    full_article = {**article, **content}
                    articles_processed.append(full_article)
                    successful_scrapes += 1
                    
                    # Count images
                    image_count = len(content.get('images', []))
                    total_images += image_count
                    
                    # Quick validation
                    validation = data_validator.validate_article(full_article)
                    
                    # Analyze price correlation for this article
                    print(f"   🔍 Analyzing price correlation...")
                    price_analyses = await price_correlator.analyze_article_price_correlation(full_article)
                    
                    # Add price analysis to article data
                    full_article['price_analyses'] = price_analyses
                    
                    print(f"   ✅ Content: {len(content.get('body_markdown', ''))} chars")
                    print(f"   ✅ Images: {image_count} found")
                    print(f"   ✅ Quality: {validation.quality_score:.1f}/100")
                    
                    # Display price correlation results
                    if price_analyses:
                        for asset, analysis in price_analyses.items():
                            print(f"   📊 {asset}: {analysis.weekly_return:+.2%} weekly return, "
                                  f"accuracy: {analysis.prediction_accuracy:.2f}")
                    else:
                        print(f"   ⚠️ No price correlation data available")
                    
                    # Save progress every 10 articles
                    if i % 10 == 0:
                        await self._save_progress(articles_processed, i, len(articles))
                
                else:
                    failed_scrapes += 1
                    print(f"   ❌ Failed to scrape content")
                
                # Minimal rate limiting for speed
                await asyncio.sleep(0.5)
                
            except Exception as e:
                failed_scrapes += 1
                print(f"   💥 Error: {str(e)[:100]}...")
                continue
        
        # Final save
        await self._save_final_results(articles_processed, successful_scrapes, failed_scrapes, total_images)
        
        return {
            'total_target': len(articles),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': failed_scrapes,
            'total_images': total_images,
            'articles_processed': articles_processed
        }
    
    async def _save_progress(self, articles_processed: List[Dict], current: int, total: int):
        """Save progress every 10 articles."""
        progress_file = self.data_dir / f"progress_articles_{current}_of_{total}.json"
        with open(progress_file, 'w') as f:
            json.dump(articles_processed, f, indent=2, default=str)
        print(f"   💾 Progress saved: {current}/{total} articles")
    
    async def _save_final_results(self, articles_processed: List[Dict], successful: int, failed: int, total_images: int):
        """Save final comprehensive results."""
        
        # Save all processed articles
        final_file = self.data_dir / "all_126_articles_processed.json"
        with open(final_file, 'w') as f:
            json.dump(articles_processed, f, indent=2, default=str)
        
        # Generate comprehensive statistics
        print(f"\n📊 COMPREHENSIVE SCRAPING RESULTS")
        print("=" * 60)
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
            
            # Calculate overall price correlation statistics
            all_price_analyses = []
            for article in articles_processed:
                if 'price_analyses' in article:
                    for asset, analysis in article['price_analyses'].items():
                        all_price_analyses.append(analysis)
            
            if all_price_analyses:
                correlation_stats = price_correlator.calculate_correlation_statistics(all_price_analyses)
                print(f"\n📈 PRICE CORRELATION ANALYSIS:")
                print(f"   • Articles with price data: {correlation_stats.get('total_articles_analyzed', 0)}")
                print(f"   • Overall prediction accuracy: {correlation_stats.get('overall_accuracy', 0):.2%}")
                print(f"   • Hit rate (>60% accuracy): {correlation_stats.get('hit_rate', 0):.2%}")
                print(f"   • Average weekly return: {correlation_stats.get('avg_weekly_return', 0):+.2%}")
                print(f"   • Average weekly volatility: {correlation_stats.get('avg_weekly_volatility', 0):.2%}")
                
                sentiment_acc = correlation_stats.get('sentiment_accuracy', {})
                print(f"   • Bullish accuracy: {sentiment_acc.get('bullish', 0):.2%}")
                print(f"   • Bearish accuracy: {sentiment_acc.get('bearish', 0):.2%}")
                print(f"   • Neutral accuracy: {sentiment_acc.get('neutral', 0):.2%}")
        
        print(f"\n💾 Complete dataset saved to: {final_file}")
        print(f"🎉 MISSION ACCOMPLISHED: {successful} articles with {total_images} images and price correlation data!")


async def main():
    """Run the comprehensive scraper for all 126 articles."""
    scraper = ComprehensiveOptionFlowsScraper()
    
    try:
        # Phase 1: Discover all articles using Firecrawl
        all_articles = await scraper.discover_all_articles()
        
        if not all_articles:
            print("❌ No articles discovered. Check Firecrawl configuration.")
            return False
        
        print(f"\n🎯 Discovered {len(all_articles)} articles total")
        
        # Phase 2: Scrape all articles with GPU acceleration  
        results = await scraper.scrape_all_articles(all_articles)
        
        # Success if we got at least 90 articles (70%+ success rate)
        success_threshold = max(90, len(all_articles) * 0.7)
        success = results['successful_scrapes'] >= success_threshold
        
        print(f"\n{'🎉 SUCCESS' if success else '⚠️ PARTIAL SUCCESS'}: {results['successful_scrapes']} articles scraped")
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraping interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Scraper crashed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
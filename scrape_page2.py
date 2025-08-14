#!/usr/bin/env python3

import asyncio
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.getcwd())

from app.scrapers.playwright_article_scraper import PlaywrightArticleScraper
from dataclasses import asdict

async def scrape_page_2():
    print('🚀 Scraping additional articles from page 2...')
    
    async with PlaywrightArticleScraper(headless=True) as scraper:
        # Navigate to page 2 directly
        page = await scraper.context.new_page()
        page_2_url = 'https://insights.deribit.com/option-flows/page/2/'
        
        print(f'Discovering URLs from: {page_2_url}')
        
        try:
            await page.goto(page_2_url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Get all links on page 2
            links = await page.query_selector_all('a[href*="option-flow"]')
            page_2_urls = set()
            
            for link in links:
                try:
                    href = await link.get_attribute('href')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            full_url = f'https://insights.deribit.com{href}'
                        elif href.startswith('option-flow-'):
                            full_url = f'https://insights.deribit.com/option-flows/{href}'
                        else:
                            full_url = href
                        
                        # Only include option flow article URLs
                        if '/option-flows/option-flow-' in full_url and full_url.endswith('/'):
                            page_2_urls.add(full_url)
                            
                except Exception as e:
                    continue
            
            await page.close()
            
            print(f'Found {len(page_2_urls)} URLs on page 2')
            
            # Show sample URLs
            if page_2_urls:
                print('Sample page 2 URLs:')
                for i, url in enumerate(list(page_2_urls)[:3], 1):
                    print(f'  {i}. {url.split("/")[-2]}...')
            
            # Now scrape each article from page 2
            page_2_articles = []
            print(f'\nScraping {len(page_2_urls)} articles from page 2...')
            
            for i, url in enumerate(page_2_urls, 1):
                print(f'Scraping {i}/{len(page_2_urls)}: {url.split("/")[-2]}...')
                article = await scraper.scrape_article(url)
                if article:
                    page_2_articles.append(article)
                    print(f'  ✅ {article.title} - {article.publication_date} ({article.readable_date}) - conf: {article.extraction_confidence:.2f}')
                else:
                    print(f'  ❌ Failed')
                    
                # Small delay to be respectful
                await asyncio.sleep(1)
            
            print(f'\nSuccessfully scraped {len(page_2_articles)} articles from page 2')
            
            # Save page 2 results
            if page_2_articles:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'page2_articles_{timestamp}.json'
                filepath = scraper.scraped_data_dir / filename
                
                # Convert to dictionaries
                articles_data = [asdict(article) for article in page_2_articles]
                
                output_data = {
                    'scrape_metadata': {
                        'scrape_timestamp': datetime.now().isoformat(),
                        'total_articles': len(page_2_articles),
                        'source_page': 'page/2/',
                        'date_range': {
                            'earliest': min(a.publication_date for a in page_2_articles if a.publication_date != "1970-01-01"),
                            'latest': max(a.publication_date for a in page_2_articles if a.publication_date != "1970-01-01")
                        },
                        'extraction_confidence_avg': sum(a.extraction_confidence for a in page_2_articles) / len(page_2_articles),
                        'scraper_type': 'playwright'
                    },
                    'articles': articles_data
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f'Saved page 2 results to: {filepath}')
            
            return page_2_articles
            
        except Exception as e:
            print(f'Error scraping page 2: {e}')
            import traceback
            traceback.print_exc()
            return []

if __name__ == "__main__":
    page_2_articles = asyncio.run(scrape_page_2())
    print(f'\n✅ Page 2 scraping completed: {len(page_2_articles)} additional articles')
#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from app.scrapers.playwright_article_scraper import PlaywrightArticleScraper

async def test_small_batch():
    try:
        async with PlaywrightArticleScraper(headless=True) as scraper:
            # Get first 5 URLs
            print('Discovering URLs...')
            all_urls = await scraper.discover_article_urls()
            test_urls = all_urls[:5] if all_urls else []
            
            print(f'Testing with {len(test_urls)} articles...')
            
            articles = []
            for i, url in enumerate(test_urls, 1):
                print(f'Scraping article {i}/{len(test_urls)}: {url.split("/")[-2]}...')
                article = await scraper.scrape_article(url)
                if article:
                    articles.append(article)
                    print(f'  ✅ {article.title} - {article.publication_date} ({article.readable_date}) - conf: {article.extraction_confidence:.2f}')
                else:
                    print(f'  ❌ Failed')
            
            print(f'Successfully scraped {len(articles)}/{len(test_urls)} articles')
            
            # Show results with proper date format
            for article in articles:
                if article.readable_date != 'Unknown Date':
                    print(f'Date format example: {article.readable_date} -> {article.publication_date}')
                    break
                    
        return articles
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    articles = asyncio.run(test_small_batch())
    print(f'Test completed with {len(articles)} articles')
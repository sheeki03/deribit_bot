#!/usr/bin/env python3

import asyncio
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, os.getcwd())

from app.scrapers.playwright_article_scraper import PlaywrightArticleScraper
from dataclasses import asdict

# All page 2 URLs
page_2_urls = [
    'https://insights.deribit.com/option-flows/option-flow-a-helping-hand/',
    'https://insights.deribit.com/option-flows/option-flow-a-perfect-storm/',
    'https://insights.deribit.com/option-flows/option-flow-a-picture-tells-a-thousand-words/',
    'https://insights.deribit.com/option-flows/option-flow-afraid-of-aftershocks/',
    'https://insights.deribit.com/option-flows/option-flow-all-eyes-on-gbtc/',
    'https://insights.deribit.com/option-flows/option-flow-allocators-spoilt-for-choice/',
    'https://insights.deribit.com/option-flows/option-flow-bets-placed/',
    'https://insights.deribit.com/option-flows/option-flow-btc-etf-and-halving-narrative/',
    'https://insights.deribit.com/option-flows/option-flow-btc-halving-play/',
    'https://insights.deribit.com/option-flows/option-flow-btc-spot-etf-approved/',
    'https://insights.deribit.com/option-flows/option-flow-bullish-exposure/',
    'https://insights.deribit.com/option-flows/option-flow-buying-flurry/',
    'https://insights.deribit.com/option-flows/option-flow-call-buying-bias/',
    'https://insights.deribit.com/option-flows/option-flow-calm-before-the-storm/',
    'https://insights.deribit.com/option-flows/option-flow-connect-the-dots/',
    'https://insights.deribit.com/option-flows/option-flow-conspicuous-volatility/',
    'https://insights.deribit.com/option-flows/option-flow-deadline-approval-approaching/',
    'https://insights.deribit.com/option-flows/option-flow-dilemma-trading/',
    'https://insights.deribit.com/option-flows/option-flow-easy-gamma/',
    'https://insights.deribit.com/option-flows/option-flow-eth-june-400-puts/',
    'https://insights.deribit.com/option-flows/option-flow-eth-rotation/',
    'https://insights.deribit.com/option-flows/option-flow-eth-saw-etf-related-hedges/',
    'https://insights.deribit.com/option-flows/option-flow-eth-unlock/',
    'https://insights.deribit.com/option-flows/option-flow-everything-changed/',
    'https://insights.deribit.com/option-flows/option-flow-fast-money-accumulation/',
    'https://insights.deribit.com/option-flows/option-flow-fast-money/',
    'https://insights.deribit.com/option-flows/option-flow-full-circle/',
    'https://insights.deribit.com/option-flows/option-flow-gamma-buying/',
    'https://insights.deribit.com/option-flows/option-flow-gamma-squeeze/',
    'https://insights.deribit.com/option-flows/option-flow-gifting-dealers-gamma/',
    'https://insights.deribit.com/option-flows/option-flow-impatient-longs/',
    'https://insights.deribit.com/option-flows/option-flow-low-vol/',
    'https://insights.deribit.com/option-flows/option-flow-macro-week/',
    'https://insights.deribit.com/option-flows/option-flow-market-support/',
    'https://insights.deribit.com/option-flows/option-flow-mix-it-up/',
    'https://insights.deribit.com/option-flows/option-flow-mixed-signals/',
    'https://insights.deribit.com/option-flows/option-flow-mixed-trades/',
    'https://insights.deribit.com/option-flows/option-flow-observing/',
    'https://insights.deribit.com/option-flows/option-flow-pre-fomc/',
    'https://insights.deribit.com/option-flows/option-flow-put-skew-gamma-iv-increased/',
    'https://insights.deribit.com/option-flows/option-flow-resistance-is-upon-us/',
    'https://insights.deribit.com/option-flows/option-flow-scrambling-for-gamma/',
    'https://insights.deribit.com/option-flows/option-flow-sell-in-may/',
    'https://insights.deribit.com/option-flows/option-flow-sell-the-news/',
    'https://insights.deribit.com/option-flows/option-flow-speculations/',
    'https://insights.deribit.com/option-flows/option-flow-spot-buying-spree/',
    'https://insights.deribit.com/option-flows/option-flow-spot-vs-resistance/',
    'https://insights.deribit.com/option-flows/option-flow-the-game-is-afoot/',
    'https://insights.deribit.com/option-flows/option-flow-the-over-write-entity/',
    'https://insights.deribit.com/option-flows/option-flow-upside-biased/',
    'https://insights.deribit.com/option-flows/option-flow-vol-buying-via-call-spreads-in-btc-and-eth/',
    'https://insights.deribit.com/option-flows/option-flow-vol-discount/',
    'https://insights.deribit.com/option-flows/option-flow-week-36-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-37-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-39-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-40-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-41-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-43-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-44-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-45-2022/',
    'https://insights.deribit.com/option-flows/option-flow-week-47-2022/',
    'https://insights.deribit.com/option-flows/option-flow-weekend-solutions/',
    'https://insights.deribit.com/option-flows/option-flow-when-will-the-tipping-point-be/'
]

async def scrape_page_2_batch():
    print(f'🚀 Scraping {len(page_2_urls)} articles from page 2...')
    
    async with PlaywrightArticleScraper(headless=True) as scraper:
        all_articles = []
        batch_size = 10
        
        # Process in batches
        for batch_start in range(0, len(page_2_urls), batch_size):
            batch_end = min(batch_start + batch_size, len(page_2_urls))
            batch_urls = page_2_urls[batch_start:batch_end]
            
            print(f'\\nProcessing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end}/{len(page_2_urls)})...')
            
            for i, url in enumerate(batch_urls, batch_start + 1):
                print(f'  Scraping {i}/{len(page_2_urls)}: {url.split("/")[-2]}...')
                article = await scraper.scrape_article(url)
                if article:
                    all_articles.append(article)
                    print(f'    ✅ {article.title} - {article.publication_date} ({article.readable_date})')
                else:
                    print(f'    ❌ Failed')
                
                # Small delay between articles
                await asyncio.sleep(0.5)
            
            print(f'  Batch {batch_start//batch_size + 1} completed: {len(all_articles)} total articles scraped')
            
            # Small delay between batches
            await asyncio.sleep(2)
        
        print(f'\\n🎉 Page 2 scraping completed!')
        print(f'Total articles scraped: {len(all_articles)}/{len(page_2_urls)}')
        
        # Save results
        if all_articles:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'page2_articles_{timestamp}.json'
            filepath = scraper.scraped_data_dir / filename
            
            # Convert to dictionaries
            articles_data = [asdict(article) for article in all_articles]
            
            # Add metadata
            output_data = {
                'scrape_metadata': {
                    'scrape_timestamp': datetime.now().isoformat(),
                    'total_articles': len(all_articles),
                    'source_page': 'page/2/',
                    'date_range': {
                        'earliest': min(a.publication_date for a in all_articles if a.publication_date != "1970-01-01"),
                        'latest': max(a.publication_date for a in all_articles if a.publication_date != "1970-01-01")
                    } if all_articles else {},
                    'extraction_confidence_avg': sum(a.extraction_confidence for a in all_articles) / len(all_articles) if all_articles else 0,
                    'scraper_type': 'playwright'
                },
                'articles': articles_data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f'\\nResults saved to: {filepath}')
            
            # Show summary
            print(f'\\n📊 SUMMARY:')
            if all_articles:
                dates = [a.publication_date for a in all_articles if a.publication_date != "1970-01-01"]
                if dates:
                    print(f'   Date range: {min(dates)} to {max(dates)}')
                    print(f'   Unique dates: {len(set(dates))}')
                print(f'   Average confidence: {sum(a.extraction_confidence for a in all_articles) / len(all_articles):.3f}')
        
        return all_articles

if __name__ == "__main__":
    articles = asyncio.run(scrape_page_2_batch())
    print(f'\\n✅ Completed: {len(articles)} additional articles from page 2')
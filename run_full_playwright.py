#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from app.scrapers.playwright_article_scraper import scrape_all_with_playwright

async def run_full_scraper():
    print('🚀 Starting comprehensive Playwright scraping...')
    print('This may take several minutes due to respectful crawling delays...')
    
    try:
        articles, report = await scrape_all_with_playwright(headless=True)
        
        print(f'\n=== SCRAPING COMPLETED ===')
        print(f'Total articles scraped: {len(articles)}')
        
        if articles:
            print(f'\nSample scraped articles:')
            for i, article in enumerate(articles[:5]):
                print(f'  {i+1}. {article.title}')
                print(f'     Date: {article.publication_date} ({article.readable_date})')
                print(f'     Confidence: {article.extraction_confidence:.2f}')
                print(f'     URL: {article.url.split("/")[-2]}...')
            
            print(f'\n=== SCRAPING REPORT ===')
            summary = report.get('summary', {})
            print(f'Successfully dated articles: {summary.get("successfully_dated", 0)}')
            print(f'Dating success rate: {summary.get("dating_success_rate", 0):.1f}%')
            print(f'Average confidence: {summary.get("average_confidence", 0):.2f}')
            
            # Date range
            date_coverage = report.get('date_coverage', {})
            if not date_coverage.get('error'):
                print(f'Date range: {date_coverage.get("earliest", "N/A")} to {date_coverage.get("latest", "N/A")}')
                print(f'Unique dates: {date_coverage.get("unique_dates", 0)}')
            
            # Confidence distribution
            conf_dist = report.get('confidence_distribution', {})
            print(f'High confidence (≥0.8): {conf_dist.get("high_confidence", 0)}')
            print(f'Medium confidence (0.5-0.8): {conf_dist.get("medium_confidence", 0)}')
            print(f'Low confidence (<0.5): {conf_dist.get("low_confidence", 0)}')
            
            print(f'\nResults saved to: {report.get("saved_to", "N/A")}')
        
        return articles, report
        
    except Exception as e:
        print(f'❌ Error during scraping: {e}')
        import traceback
        traceback.print_exc()
        return [], {}

if __name__ == "__main__":
    articles, report = asyncio.run(run_full_scraper())
    print(f'\n✅ Completed: Scraped {len(articles)} articles with ultra-accurate date extraction')
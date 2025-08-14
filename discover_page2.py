#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, os.getcwd())

from app.scrapers.playwright_article_scraper import PlaywrightArticleScraper

async def discover_page_2_urls():
    print('🔍 Discovering URLs from page 2...')
    
    async with PlaywrightArticleScraper(headless=True) as scraper:
        page = await scraper.context.new_page()
        page_2_url = 'https://insights.deribit.com/option-flows/page/2/'
        
        print(f'Accessing: {page_2_url}')
        
        try:
            await page.goto(page_2_url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Get all links on page 2
            links = await page.query_selector_all('a')
            page_2_urls = set()
            
            for link in links:
                try:
                    href = await link.get_attribute('href')
                    if href and 'option-flow' in href:
                        # Convert to full URL if relative
                        if href.startswith('/'):
                            full_url = f'https://insights.deribit.com{href}'
                        elif href.startswith('option-flow-'):
                            full_url = f'https://insights.deribit.com/option-flows/{href}'
                        else:
                            full_url = href
                        
                        # Only include proper option flow article URLs
                        if '/option-flows/option-flow-' in full_url:
                            page_2_urls.add(full_url)
                            
                except:
                    continue
            
            await page.close()
            
            print(f'Found {len(page_2_urls)} URLs on page 2')
            
            if page_2_urls:
                print('\nPage 2 URLs found:')
                for i, url in enumerate(sorted(page_2_urls), 1):
                    print(f'  {i:2}. {url}')
            
            return list(page_2_urls)
            
        except Exception as e:
            print(f'Error: {e}')
            import traceback
            traceback.print_exc()
            return []

if __name__ == "__main__":
    urls = asyncio.run(discover_page_2_urls())
    print(f'\n✅ Discovery completed: {len(urls)} URLs found on page 2')
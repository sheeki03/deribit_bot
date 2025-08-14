"""
Ultimate Playwright-Based Article Scraper
Uses Playwright for JavaScript-heavy sites with ultra-accurate date extraction.
"""
from __future__ import annotations

import asyncio
import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime, date, timedelta
import logging
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import concurrent.futures
from bs4 import BeautifulSoup

from playwright.async_api import async_playwright, Page, BrowserContext

logger = logging.getLogger(__name__)

@dataclass
class PlaywrightArticle:
    """Represents an article scraped with Playwright."""
    title: str
    url: str
    publication_date: str  # YYYY-MM-DD format
    readable_date: str     # "August 11, 2025" format
    category: str          # "Option Flows"
    author: str
    content: str
    images: List[Dict[str, str]]
    extraction_confidence: float
    scrape_timestamp: str
    html_snapshot: str     # For debugging

class PlaywrightArticleScraper:
    """Ultimate Playwright-based article scraper."""
    
    def __init__(self, headless: bool = True, slow_mo: int = 100):
        """Initialize Playwright scraper.
        
        Args:
            headless: Whether to run in headless mode
            slow_mo: Slow motion delay in milliseconds
        """
        self.base_url = "https://insights.deribit.com/option-flows/"
        self.headless = headless
        self.slow_mo = slow_mo
        
        # Output directory
        self.scraped_data_dir = Path(__file__).parent.parent.parent / 'scraped_data' / 'playwright'
        self.scraped_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Browser and context will be initialized when needed
        self.browser = None
        self.context = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.playwright = await async_playwright().start()
        
        # Launch browser with optimal settings
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create context with realistic browser profile
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def discover_article_urls(self) -> List[str]:
        """Discover all article URLs using Playwright."""
        logger.info("Discovering article URLs with Playwright...")
        
        page = await self.context.new_page()
        article_urls = set()
        
        try:
            # Navigate to main page
            await page.goto(self.base_url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to load
            await page.wait_for_timeout(3000)
            
            # Scroll to load any lazy-loaded content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            
            # Get all links on the page
            links = await page.query_selector_all('a[href*="option-flow"]')
            
            for link in links:
                try:
                    href = await link.get_attribute('href')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            full_url = f"https://insights.deribit.com{href}"
                        elif href.startswith('option-flow-'):
                            full_url = urljoin(self.base_url, href)
                        else:
                            full_url = href
                        
                        # Only include option flow article URLs
                        if '/option-flows/option-flow-' in full_url and full_url.endswith('/'):
                            article_urls.add(full_url)
                        
                except Exception as e:
                    logger.debug(f"Error processing link: {e}")
                    continue
            
            # If no URLs found, try pagination
            if not article_urls:
                logger.warning("No URLs found on main page, trying pagination...")
                article_urls = await self._discover_paginated_urls(page)
            
            # If still no URLs, try alternative selectors
            if not article_urls:
                logger.warning("Trying alternative link discovery methods...")
                article_urls = await self._discover_alternative_selectors(page)
            
        except Exception as e:
            logger.error(f"Error discovering URLs: {e}")
        
        finally:
            await page.close()
        
        article_urls = list(article_urls)
        logger.info(f"Discovered {len(article_urls)} article URLs")
        
        # Show sample URLs for verification
        if article_urls:
            logger.info("Sample URLs discovered:")
            for i, url in enumerate(article_urls[:5], 1):
                logger.info(f"  {i}. {url}")
        
        return article_urls
    
    async def _discover_paginated_urls(self, page: Page) -> set:
        """Discover URLs from paginated pages."""
        urls = set()
        current_page = 1
        max_pages = 10
        
        while current_page <= max_pages:
            try:
                if current_page > 1:
                    page_url = f"{self.base_url}page/{current_page}/"
                    await page.goto(page_url, wait_until='networkidle', timeout=30000)
                    await page.wait_for_timeout(2000)
                
                # Look for article links on this page
                links = await page.query_selector_all('a')
                page_urls = set()
                
                for link in links:
                    try:
                        href = await link.get_attribute('href')
                        text = await link.text_content()
                        
                        if href and 'option-flow' in href:
                            if href.startswith('/'):
                                full_url = f"https://insights.deribit.com{href}"
                            else:
                                full_url = href
                            
                            if '/option-flows/option-flow-' in full_url:
                                urls.add(full_url)
                                page_urls.add(full_url)
                    
                    except:
                        continue
                
                # If no URLs found on this page, we've reached the end
                if not page_urls:
                    break
                
                current_page += 1
                
            except Exception as e:
                logger.warning(f"Error on page {current_page}: {e}")
                break
        
        return urls
    
    async def _discover_alternative_selectors(self, page: Page) -> set:
        """Try alternative selectors for finding article links."""
        urls = set()
        
        # Alternative selectors to try
        selectors = [
            'a[title*="Option Flow"]',
            'a[href*="/option-flows/"]',
            '.post-title a',
            '.entry-title a',
            'h2 a',
            'h3 a',
            '.article-title a'
        ]
        
        for selector in selectors:
            try:
                links = await page.query_selector_all(selector)
                for link in links:
                    href = await link.get_attribute('href')
                    if href and 'option-flow' in href:
                        if href.startswith('/'):
                            full_url = f"https://insights.deribit.com{href}"
                        else:
                            full_url = href
                        urls.add(full_url)
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        return urls
    
    async def scrape_article(self, url: str) -> Optional[PlaywrightArticle]:
        """Scrape individual article with Playwright.
        
        Args:
            url: Article URL to scrape
            
        Returns:
            PlaywrightArticle or None if failed
        """
        logger.info(f"Scraping article: {url}")
        
        page = await self.context.new_page()
        
        try:
            # Navigate to article
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to fully load
            await page.wait_for_timeout(3000)
            
            # Scroll to ensure all content is loaded
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1000)
            
            # Get page HTML for parsing
            html_content = await page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract data using advanced methods
            title = await self._extract_title_playwright(page, soup)
            publication_date, readable_date, confidence = await self._extract_date_playwright(page, soup, url)
            category = await self._extract_category_playwright(page, soup)
            author = await self._extract_author_playwright(page, soup)
            content = await self._extract_content_playwright(page, soup)
            images = await self._extract_images_playwright(page, soup, url)
            
            if not title:
                logger.warning(f"No title extracted for {url}")
                return None
            
            if not publication_date:
                logger.warning(f"No publication date extracted for {url}")
                # Don't fail completely, but set low confidence
                publication_date = "1970-01-01"
                readable_date = "Unknown Date"
                confidence = 0.1
            
            article = PlaywrightArticle(
                title=title,
                url=url,
                publication_date=publication_date,
                readable_date=readable_date,
                category=category,
                author=author,
                content=content,
                images=images,
                extraction_confidence=confidence,
                scrape_timestamp=datetime.now().isoformat(),
                html_snapshot=html_content[:5000]  # First 5000 chars for debugging
            )
            
            logger.info(f"Successfully scraped: {title} ({publication_date})")
            return article
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
            
        finally:
            await page.close()
    
    async def _extract_title_playwright(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract title using Playwright + BeautifulSoup."""
        
        # Method 1: Try to get title via Playwright selectors
        title_selectors = [
            'h1.entry-title',
            'h1.post-title', 
            'h1.article-title',
            'h1',
            '.entry-header h1',
            '.post-header h1'
        ]
        
        for selector in title_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    title = await element.text_content()
                    if title and 'Option Flow' in title:
                        return title.strip()
            except:
                continue
        
        # Method 2: Use BeautifulSoup fallback
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            text = h1.get_text(strip=True)
            if text and 'Option Flow' in text:
                return text
        
        # Method 3: Page title
        try:
            title = await page.title()
            if title and 'Option Flow' in title:
                # Clean up title
                title = re.sub(r'\s*\|\s*Deribit.*$', '', title)
                return title
        except:
            pass
        
        return ""
    
    async def _extract_date_playwright(self, page: Page, soup: BeautifulSoup, url: str) -> Tuple[str, str, float]:
        """Ultra-accurate date extraction using Playwright + BeautifulSoup."""
        
        methods = []
        
        # Method 1: Look for time elements via Playwright
        try:
            time_elements = await page.query_selector_all('time')
            for element in time_elements:
                datetime_attr = await element.get_attribute('datetime')
                text_content = await element.text_content()
                
                if datetime_attr:
                    try:
                        dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                        readable = text_content.strip() if text_content else dt.strftime('%B %d, %Y')
                        methods.append((dt.strftime('%Y-%m-%d'), readable, 0.95))
                    except:
                        pass
        except:
            pass
        
        # Method 2: Look for the exact pattern we expect
        try:
            page_text = await page.evaluate("document.body.innerText")
        except:
            page_text = ""
        
        # Look for "Month DD, YYYY|Option Flows" pattern
        date_patterns = [
            r'([A-Za-z]+ \d{1,2}, \d{4})\s*\|\s*Option Flows',
            r'([A-Za-z]+ \d{1,2}, \d{4})\s*\|',
            r'([A-Za-z]+ \d{1,2}, \d{4})',
            r'(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}',
            r'(\d{4}-\d{2}-\d{2})',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, page_text)
            for match in matches:
                try:
                    readable_date = match
                    
                    if re.match(r'[A-Za-z]+ \d{1,2}, \d{4}', readable_date):
                        # Parse "August 11, 2025" format
                        dt = datetime.strptime(readable_date, '%B %d, %Y')
                        iso_date = dt.strftime('%Y-%m-%d')
                        methods.append((iso_date, readable_date, 0.9))
                        
                    elif re.match(r'\d{4}-\d{2}-\d{2}', readable_date):
                        # ISO format
                        dt = datetime.strptime(readable_date.split('T')[0], '%Y-%m-%d')
                        readable = dt.strftime('%B %d, %Y')
                        methods.append((readable_date.split('T')[0], readable, 0.85))
                        
                except ValueError:
                    continue
        
        # Method 3: Look in meta tags via BeautifulSoup
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            if meta.get('property') in ['article:published_time', 'datePublished'] or \
               meta.get('name') in ['date', 'publish_date', 'publication_date']:
                content = meta.get('content', '')
                if content:
                    try:
                        # Try various date formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%B %d, %Y']:
                            try:
                                dt = datetime.strptime(content.split('+')[0].split('Z')[0], fmt)
                                iso_date = dt.strftime('%Y-%m-%d')
                                readable = dt.strftime('%B %d, %Y')
                                methods.append((iso_date, readable, 0.8))
                                break
                            except:
                                continue
                    except:
                        pass
        
        # Method 4: Try to extract from JSON-LD structured data
        try:
            json_scripts = await page.query_selector_all('script[type="application/ld+json"]')
            for script in json_scripts:
                script_content = await script.text_content()
                if script_content:
                    try:
                        data = json.loads(script_content)
                        date_published = data.get('datePublished', '')
                        if date_published:
                            dt = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                            iso_date = dt.strftime('%Y-%m-%d')
                            readable = dt.strftime('%B %d, %Y')
                            methods.append((iso_date, readable, 0.9))
                    except:
                        continue
        except:
            pass
        
        # Filter and sort by confidence
        if methods:
            # Filter to reasonable dates (2021-2026)
            valid_methods = []
            for iso_date, readable, confidence in methods:
                try:
                    year = int(iso_date.split('-')[0])
                    if 2021 <= year <= 2026:
                        valid_methods.append((iso_date, readable, confidence))
                except:
                    continue
            
            if valid_methods:
                # Sort by confidence
                valid_methods.sort(key=lambda x: x[2], reverse=True)
                return valid_methods[0]
        
        return "", "", 0.0
    
    async def _extract_category_playwright(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract category using Playwright."""
        
        # Try category selectors
        category_selectors = [
            '.post-category',
            '.entry-category',
            '.category',
            '[class*="category"]'
        ]
        
        for selector in category_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and 'Option' in text:
                        return text.strip()
            except:
                continue
        
        return 'Option Flows'  # Default
    
    async def _extract_author_playwright(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract author using Playwright."""
        
        # Try author selectors
        author_selectors = [
            '.author',
            '.byline',
            '.post-author',
            '.entry-author',
            '[class*="author"]'
        ]
        
        for selector in author_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and len(text.strip()) < 50:
                        return text.strip()
            except:
                continue
        
        # Look for "By" pattern in text
        try:
            page_text = await page.evaluate("document.body.innerText")
        except:
            page_text = ""
        author_match = re.search(r'By\s+([^|\n]+)', page_text)
        if author_match:
            author = author_match.group(1).strip()
            if len(author) < 50:
                return author
        
        return 'Tony Stewart'  # Default for Option Flows
    
    async def _extract_content_playwright(self, page: Page, soup: BeautifulSoup) -> str:
        """Extract main content using Playwright."""
        
        # Try content selectors
        content_selectors = [
            '.entry-content',
            '.post-content',
            '.article-content',
            '.content',
            'main'
        ]
        
        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    if text and len(text.strip()) > 100:
                        return text.strip()
            except:
                continue
        
        # Fallback: get all paragraph text
        try:
            paragraphs = await page.query_selector_all('p')
            content_parts = []
            
            for p in paragraphs:
                text = await p.text_content()
                if text and len(text.strip()) > 20:
                    content_parts.append(text.strip())
            
            return '\n\n'.join(content_parts)
            
        except:
            return ""
    
    async def _extract_images_playwright(self, page: Page, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images using Playwright."""
        images = []
        
        try:
            img_elements = await page.query_selector_all('img')
            
            for img in img_elements:
                src = await img.get_attribute('src')
                alt = await img.get_attribute('alt')
                
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('/'):
                        src = urljoin(base_url, src)
                    
                    images.append({
                        'url': src,
                        'alt_text': alt or ''
                    })
        except:
            pass
        
        return images
    
    async def scrape_all_articles(self, max_concurrent: int = 3) -> List[PlaywrightArticle]:
        """Scrape all articles with controlled concurrency.
        
        Args:
            max_concurrent: Maximum concurrent scraping tasks
            
        Returns:
            List of scraped articles
        """
        logger.info("Starting comprehensive article scraping with Playwright...")
        
        # Discover URLs
        article_urls = await self.discover_article_urls()
        
        if not article_urls:
            logger.error("No article URLs discovered!")
            return []
        
        logger.info(f"Scraping {len(article_urls)} articles...")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url):
            async with semaphore:
                await asyncio.sleep(1)  # Be respectful
                return await self.scrape_article(url)
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(url) for url in article_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        articles = []
        for i, result in enumerate(results):
            if isinstance(result, PlaywrightArticle):
                articles.append(result)
                logger.info(f"✅ Scraped: {result.title} ({result.publication_date})")
            elif isinstance(result, Exception):
                logger.error(f"❌ Error scraping {article_urls[i]}: {result}")
            else:
                logger.warning(f"❌ Failed to scrape: {article_urls[i]}")
        
        # Sort by publication date
        articles.sort(key=lambda x: x.publication_date if x.publication_date != "1970-01-01" else "9999-12-31", reverse=True)
        
        logger.info(f"Successfully scraped {len(articles)} articles")
        return articles
    
    def save_articles(self, articles: List[PlaywrightArticle]) -> Path:
        """Save scraped articles to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"playwright_articles_{timestamp}.json"
        filepath = self.scraped_data_dir / filename
        
        # Convert to dictionaries
        articles_data = [asdict(article) for article in articles]
        
        # Add metadata
        output_data = {
            'scrape_metadata': {
                'scrape_timestamp': datetime.now().isoformat(),
                'total_articles': len(articles),
                'date_range': {
                    'earliest': min(a.publication_date for a in articles if a.publication_date != "1970-01-01") if articles else None,
                    'latest': max(a.publication_date for a in articles if a.publication_date != "1970-01-01") if articles else None
                },
                'extraction_confidence_avg': sum(a.extraction_confidence for a in articles) / len(articles) if articles else 0,
                'scraper_type': 'playwright'
            },
            'articles': articles_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath


# Async function to run the scraper
async def scrape_all_with_playwright(headless: bool = True) -> Tuple[List[PlaywrightArticle], Dict]:
    """Main function to scrape all articles with Playwright.
    
    Args:
        headless: Whether to run browser in headless mode
        
    Returns:
        Tuple of (articles, report)
    """
    async with PlaywrightArticleScraper(headless=headless) as scraper:
        articles = await scraper.scrape_all_articles(max_concurrent=2)
        
        # Generate report
        report = generate_scraping_report(articles)
        
        # Save results
        if articles:
            saved_path = scraper.save_articles(articles)
            report['saved_to'] = str(saved_path)
        
        return articles, report


def generate_scraping_report(articles: List[PlaywrightArticle]) -> Dict:
    """Generate comprehensive scraping report."""
    if not articles:
        return {'error': 'No articles scraped'}
    
    # Date analysis
    dated_articles = [a for a in articles if a.publication_date != "1970-01-01"]
    
    # Confidence analysis
    confidences = [a.extraction_confidence for a in articles]
    avg_confidence = sum(confidences) / len(confidences)
    
    high_conf = sum(1 for c in confidences if c >= 0.8)
    med_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)
    
    # Date range
    if dated_articles:
        dates = [a.publication_date for a in dated_articles]
        date_range = {
            'earliest': min(dates),
            'latest': max(dates),
            'unique_dates': len(set(dates)),
            'dated_articles': len(dated_articles)
        }
    else:
        date_range = {'error': 'No dated articles'}
    
    return {
        'summary': {
            'total_articles': len(articles),
            'successfully_dated': len(dated_articles),
            'dating_success_rate': len(dated_articles) / len(articles) * 100,
            'average_confidence': avg_confidence,
        },
        'confidence_distribution': {
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf
        },
        'date_coverage': date_range,
        'sample_articles': [
            {
                'title': a.title,
                'date': a.publication_date,
                'readable_date': a.readable_date,
                'confidence': a.extraction_confidence,
                'url_slug': a.url.split('/')[-2] if '/' in a.url else a.url
            }
            for a in sorted(articles, key=lambda x: x.extraction_confidence, reverse=True)[:10]
        ]
    }


# Synchronous wrapper
def run_playwright_scraper(headless: bool = True) -> Tuple[List[PlaywrightArticle], Dict]:
    """Synchronous wrapper for the Playwright scraper."""
    return asyncio.run(scrape_all_with_playwright(headless=headless))
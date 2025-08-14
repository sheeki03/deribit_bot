"""
Ultra-Accurate Article Scraper with Proper Date Extraction
Crawls each individual article sub-link and extracts dates with high precision.
"""
from __future__ import annotations

import requests
import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime, date
import logging
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import concurrent.futures
from bs4 import BeautifulSoup

# Try to import firecrawl if available
try:
    from firecrawl import FirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    print("⚠️ Firecrawl not available, will use requests + BeautifulSoup")

logger = logging.getLogger(__name__)

@dataclass
class AccurateArticle:
    """Represents an accurately scraped article with proper date."""
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

class AccurateArticleScraper:
    """Ultra-accurate article scraper for Deribit Option Flows."""
    
    def __init__(self, firecrawl_api_key: Optional[str] = None):
        """Initialize accurate article scraper.
        
        Args:
            firecrawl_api_key: Optional Firecrawl API key for enhanced scraping
        """
        self.firecrawl_api_key = firecrawl_api_key
        self.base_url = "https://insights.deribit.com/option-flows/"
        self.scraped_data_dir = Path(__file__).parent.parent.parent / 'scraped_data' / 'accurate'
        self.scraped_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scraping client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        if FIRECRAWL_AVAILABLE and firecrawl_api_key:
            self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
            logger.info("Initialized with Firecrawl")
        else:
            self.firecrawl = None
            logger.info("Initialized with requests + BeautifulSoup")
    
    def discover_all_article_urls(self) -> List[str]:
        """Discover all article URLs from the main Option Flows page."""
        logger.info("Discovering all article URLs from main page...")
        
        article_urls = []
        
        try:
            if self.firecrawl:
                # Use Firecrawl for robust extraction
                result = self.firecrawl.scrape_url(
                    self.base_url,
                    params={
                        'formats': ['markdown', 'html'],
                        'includeTags': ['a'],
                        'onlyMainContent': True
                    }
                )
                
                if result and 'html' in result:
                    soup = BeautifulSoup(result['html'], 'html.parser')
                else:
                    # Fallback to markdown parsing
                    content = result.get('markdown', '') if result else ''
                    # Extract URLs from markdown
                    url_pattern = r'\[([^\]]+)\]\(([^)]+option-flow-[^)]+)\)'
                    matches = re.findall(url_pattern, content)
                    article_urls.extend([match[1] for match in matches])
            else:
                # Use requests + BeautifulSoup
                response = self.session.get(self.base_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
            
            if 'soup' in locals():
                # Extract all article links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Look for option flow article URLs
                    if '/option-flows/option-flow-' in href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in article_urls:
                            article_urls.append(full_url)
                    
                    # Also check for relative URLs
                    elif href.startswith('option-flow-'):
                        full_url = urljoin(self.base_url, href)
                        if full_url not in article_urls:
                            article_urls.append(full_url)
        
        except Exception as e:
            logger.error(f"Failed to discover article URLs: {e}")
            
            # Fallback: try to scrape pagination
            try:
                article_urls = self._scrape_paginated_urls()
            except Exception as e2:
                logger.error(f"Fallback pagination scraping failed: {e2}")
        
        logger.info(f"Discovered {len(article_urls)} article URLs")
        return list(set(article_urls))  # Remove duplicates
    
    def _scrape_paginated_urls(self) -> List[str]:
        """Scrape URLs from paginated results."""
        article_urls = []
        page = 1
        max_pages = 10  # Safety limit
        
        while page <= max_pages:
            page_url = f"{self.base_url}page/{page}/" if page > 1 else self.base_url
            
            try:
                response = self.session.get(page_url)
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for article links on this page
                page_articles = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/option-flows/option-flow-' in href:
                        full_url = urljoin(self.base_url, href)
                        if full_url not in article_urls:
                            article_urls.append(full_url)
                            page_articles.append(full_url)
                
                # If no new articles found, we've reached the end
                if not page_articles:
                    break
                
                page += 1
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                logger.warning(f"Failed to scrape page {page}: {e}")
                break
        
        return article_urls
    
    def scrape_individual_article(self, url: str) -> Optional[AccurateArticle]:
        """Scrape an individual article with ultra-accurate date extraction.
        
        Args:
            url: Article URL to scrape
            
        Returns:
            AccurateArticle with proper date extraction or None if failed
        """
        logger.info(f"Scraping individual article: {url}")
        
        try:
            if self.firecrawl:
                # Use Firecrawl for robust content extraction
                result = self.firecrawl.scrape_url(
                    url,
                    params={
                        'formats': ['markdown', 'html'],
                        'includeTags': ['h1', 'h2', 'h3', 'p', 'time', 'span', 'div'],
                        'onlyMainContent': True,
                        'waitFor': 2000
                    }
                )
                
                if not result:
                    raise Exception("Firecrawl returned no data")
                
                html_content = result.get('html', '')
                markdown_content = result.get('markdown', '')
                
            else:
                # Use requests + BeautifulSoup
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                html_content = response.text
                markdown_content = ""
            
            # Parse with BeautifulSoup for precise extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup, markdown_content)
            
            # Extract date with multiple methods
            publication_date, readable_date, confidence = self._extract_date_ultra_accurate(
                soup, markdown_content, url
            )
            
            # Extract category
            category = self._extract_category(soup, markdown_content)
            
            # Extract author
            author = self._extract_author(soup, markdown_content)
            
            # Extract main content
            content = self._extract_content(soup, markdown_content)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            if not title or not publication_date:
                logger.warning(f"Missing critical data for {url}: title={bool(title)}, date={bool(publication_date)}")
                return None
            
            article = AccurateArticle(
                title=title,
                url=url,
                publication_date=publication_date,
                readable_date=readable_date,
                category=category,
                author=author,
                content=content,
                images=images,
                extraction_confidence=confidence,
                scrape_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Successfully scraped: {title} ({publication_date})")
            return article
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, markdown: str) -> str:
        """Extract article title with multiple fallbacks."""
        
        # Method 1: Look for h1 tags
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            text = h1.get_text(strip=True)
            if text and 'Option Flow' in text:
                return text
        
        # Method 2: Look for title tag
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Clean up title
            title = re.sub(r'\s*\|\s*Deribit.*$', '', title)
            if title and 'Option Flow' in title:
                return title
        
        # Method 3: Extract from markdown
        if markdown:
            lines = markdown.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line and 'Option Flow' in line and not line.startswith('#'):
                    return line
        
        # Method 4: Look for article header
        article_headers = soup.find_all(['h1', 'h2'], class_=re.compile(r'title|heading|header'))
        for header in article_headers:
            text = header.get_text(strip=True)
            if text and 'Option Flow' in text:
                return text
        
        return ""
    
    def _extract_date_ultra_accurate(self, soup: BeautifulSoup, markdown: str, url: str) -> Tuple[str, str, float]:
        """Ultra-accurate date extraction with multiple methods.
        
        Returns:
            Tuple of (YYYY-MM-DD, readable_date, confidence)
        """
        methods = []
        
        # Method 1: Look for time tags
        time_tags = soup.find_all('time')
        for time_tag in time_tags:
            datetime_attr = time_tag.get('datetime')
            if datetime_attr:
                try:
                    dt = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                    readable = time_tag.get_text(strip=True)
                    methods.append((dt.strftime('%Y-%m-%d'), readable, 0.95))
                except:
                    pass
        
        # Method 2: Look for date patterns in specific locations
        # Search for the exact format: "August 11, 2025|Option Flows"
        date_patterns = [
            # Full month name with year
            r'([A-Za-z]+ \d{1,2}, \d{4})\s*\|\s*Option Flows',
            r'([A-Za-z]+ \d{1,2}, \d{4})\s*\|',
            r'([A-Za-z]+ \d{1,2}, \d{4})',
            
            # ISO format
            r'(\d{4}-\d{2}-\d{2})',
            
            # Other common formats
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}-\d{1,2}-\d{4})',
        ]
        
        # Search in various elements
        search_elements = [
            soup.find('div', class_=re.compile(r'date|published|meta')),
            soup.find('span', class_=re.compile(r'date|published|meta')),
            soup.find('p', string=re.compile(r'[A-Za-z]+ \d{1,2}, \d{4}')),
        ]
        
        # Add all text content for pattern matching
        search_texts = [soup.get_text(), markdown]
        
        # Add specific elements to search texts
        for element in search_elements:
            if element:
                search_texts.append(element.get_text())
        
        for text in search_texts:
            if not text:
                continue
                
            for pattern in date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Try to parse the date
                        readable_date = match
                        
                        # Convert to standard format
                        if re.match(r'[A-Za-z]+ \d{1,2}, \d{4}', readable_date):
                            # Parse "August 11, 2025" format
                            dt = datetime.strptime(readable_date, '%B %d, %Y')
                            iso_date = dt.strftime('%Y-%m-%d')
                            methods.append((iso_date, readable_date, 0.9))
                            
                        elif re.match(r'\d{4}-\d{2}-\d{2}', readable_date):
                            # Already in ISO format
                            dt = datetime.strptime(readable_date, '%Y-%m-%d')
                            readable = dt.strftime('%B %d, %Y')
                            methods.append((readable_date, readable, 0.85))
                            
                        elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', readable_date):
                            # MM/DD/YYYY format
                            dt = datetime.strptime(readable_date, '%m/%d/%Y')
                            iso_date = dt.strftime('%Y-%m-%d')
                            readable = dt.strftime('%B %d, %Y')
                            methods.append((iso_date, readable, 0.8))
                            
                    except ValueError:
                        continue
        
        # Method 3: Extract from URL if it contains date
        url_date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
        if url_date_match:
            year, month, day = url_date_match.groups()
            iso_date = f"{year}-{month}-{day}"
            try:
                dt = datetime.strptime(iso_date, '%Y-%m-%d')
                readable = dt.strftime('%B %d, %Y')
                methods.append((iso_date, readable, 0.7))
            except:
                pass
        
        # Method 4: Look in meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            if meta.get('property') in ['article:published_time', 'datePublished'] or \
               meta.get('name') in ['date', 'publish_date', 'publication_date']:
                content = meta.get('content', '')
                if content:
                    try:
                        # Try to parse various formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%B %d, %Y']:
                            try:
                                dt = datetime.strptime(content.split('+')[0].split('Z')[0], fmt)
                                iso_date = dt.strftime('%Y-%m-%d')
                                readable = dt.strftime('%B %d, %Y')
                                methods.append((iso_date, readable, 0.85))
                                break
                            except:
                                continue
                    except:
                        pass
        
        # Sort by confidence and return best match
        if methods:
            methods.sort(key=lambda x: x[2], reverse=True)
            
            # Filter to only reasonable dates (2021-2026)
            valid_methods = []
            for iso_date, readable, confidence in methods:
                try:
                    year = int(iso_date.split('-')[0])
                    if 2021 <= year <= 2026:
                        valid_methods.append((iso_date, readable, confidence))
                except:
                    continue
            
            if valid_methods:
                return valid_methods[0]
        
        return "", "", 0.0
    
    def _extract_category(self, soup: BeautifulSoup, markdown: str) -> str:
        """Extract article category."""
        
        # Look for category indicators
        category_selectors = [
            'span[class*="category"]',
            'div[class*="category"]',
            'span[class*="tag"]',
            '.post-category',
            '.category'
        ]
        
        for selector in category_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and 'Option' in text:
                    return text
        
        # Look for "Option Flows" text pattern
        text_content = soup.get_text()
        if 'Option Flows' in text_content:
            return 'Option Flows'
        
        return 'Option Flows'  # Default for these articles
    
    def _extract_author(self, soup: BeautifulSoup, markdown: str) -> str:
        """Extract article author."""
        
        # Look for author patterns
        author_patterns = [
            r'By\s+([^|]+)\|',
            r'Author:\s*([^\\n]+)',
            r'Written by\s*([^\\n]+)',
        ]
        
        search_texts = [soup.get_text(), markdown]
        
        for text in search_texts:
            if not text:
                continue
            for pattern in author_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    author = matches[0].strip()
                    if author and len(author) < 50:  # Reasonable author name length
                        return author
        
        # Look for specific author elements
        author_selectors = [
            '.author',
            '.byline',
            '[class*="author"]',
            'span[class*="by"]'
        ]
        
        for selector in author_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) < 50 and not any(x in text.lower() for x in ['comment', 'share', 'follow']):
                    return text
        
        return 'Tony Stewart'  # Default for Option Flows articles
    
    def _extract_content(self, soup: BeautifulSoup, markdown: str) -> str:
        """Extract main article content."""
        
        # Use markdown if available (cleaner)
        if markdown:
            # Remove metadata and keep main content
            lines = markdown.split('\n')
            content_lines = []
            skip_next = False
            
            for line in lines:
                line = line.strip()
                
                # Skip headers, metadata
                if line.startswith('#') or line.startswith('---') or '|' in line[:50]:
                    continue
                
                # Keep substantial content
                if len(line) > 20:
                    content_lines.append(line)
            
            if content_lines:
                return '\n'.join(content_lines)
        
        # Fallback to HTML extraction
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        
        # Look for main content areas
        content_selectors = [
            '.post-content',
            '.entry-content',
            '.article-content',
            'main',
            '.content'
        ]
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                return content_div.get_text(strip=True)
        
        # Fallback: get all paragraph text
        paragraphs = soup.find_all('p')
        content_paragraphs = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 50:  # Substantial paragraphs only
                content_paragraphs.append(text)
        
        return '\n\n'.join(content_paragraphs)
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract article images."""
        images = []
        
        img_tags = soup.find_all('img')
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if src:
                # Convert relative URLs to absolute
                if src.startswith('/'):
                    src = urljoin(base_url, src)
                
                alt_text = img.get('alt', '')
                
                images.append({
                    'url': src,
                    'alt_text': alt_text
                })
        
        return images
    
    def scrape_all_articles(self, max_workers: int = 5, delay: float = 2.0) -> List[AccurateArticle]:
        """Scrape all articles with proper concurrency control.
        
        Args:
            max_workers: Maximum concurrent workers
            delay: Delay between requests (seconds)
            
        Returns:
            List of accurately scraped articles
        """
        logger.info("Starting comprehensive article scraping...")
        
        # Discover all URLs
        article_urls = self.discover_all_article_urls()
        
        if not article_urls:
            logger.error("No article URLs discovered!")
            return []
        
        logger.info(f"Found {len(article_urls)} articles to scrape")
        
        articles = []
        
        # Use ThreadPoolExecutor for controlled concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {
                executor.submit(self._scrape_with_delay, url, delay): url 
                for url in article_urls
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    article = future.result()
                    if article:
                        articles.append(article)
                        logger.info(f"✅ Scraped: {article.title} ({article.publication_date})")
                    else:
                        logger.warning(f"❌ Failed to scrape: {url}")
                except Exception as e:
                    logger.error(f"❌ Error scraping {url}: {e}")
        
        # Sort by publication date
        articles.sort(key=lambda x: x.publication_date, reverse=True)
        
        logger.info(f"Successfully scraped {len(articles)} articles")
        return articles
    
    def _scrape_with_delay(self, url: str, delay: float) -> Optional[AccurateArticle]:
        """Scrape article with delay to be respectful."""
        time.sleep(delay)
        return self.scrape_individual_article(url)
    
    def save_scraped_articles(self, articles: List[AccurateArticle]) -> Path:
        """Save scraped articles to JSON file.
        
        Args:
            articles: List of scraped articles
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"accurate_articles_{timestamp}.json"
        filepath = self.scraped_data_dir / filename
        
        # Convert to dictionaries for JSON serialization
        articles_data = [asdict(article) for article in articles]
        
        # Add metadata
        output_data = {
            'scrape_metadata': {
                'scrape_timestamp': datetime.now().isoformat(),
                'total_articles': len(articles),
                'date_range': {
                    'earliest': min(a.publication_date for a in articles) if articles else None,
                    'latest': max(a.publication_date for a in articles) if articles else None
                },
                'extraction_confidence_avg': sum(a.extraction_confidence for a in articles) / len(articles) if articles else 0
            },
            'articles': articles_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {filepath}")
        return filepath
    
    def generate_accuracy_report(self, articles: List[AccurateArticle]) -> Dict:
        """Generate accuracy report for scraped articles."""
        if not articles:
            return {'error': 'No articles to analyze'}
        
        # Confidence analysis
        confidences = [a.extraction_confidence for a in articles]
        avg_confidence = sum(confidences) / len(confidences)
        
        high_confidence = sum(1 for c in confidences if c >= 0.8)
        medium_confidence = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.6)
        
        # Date range analysis
        dates = [a.publication_date for a in articles if a.publication_date]
        date_range = {
            'earliest': min(dates) if dates else None,
            'latest': max(dates) if dates else None,
            'unique_dates': len(set(dates)) if dates else 0,
            'total_dated_articles': len(dates)
        }
        
        # Content analysis
        content_lengths = [len(a.content) for a in articles]
        avg_content_length = sum(content_lengths) / len(content_lengths) if content_lengths else 0
        
        # Recent articles (last 30 days from latest)
        if dates:
            latest_date = datetime.strptime(max(dates), '%Y-%m-%d')
            recent_cutoff = (latest_date - timedelta(days=30)).strftime('%Y-%m-%d')
            recent_articles = sum(1 for d in dates if d >= recent_cutoff)
        else:
            recent_articles = 0
        
        return {
            'summary': {
                'total_articles': len(articles),
                'successfully_dated': len(dates),
                'dating_success_rate': len(dates) / len(articles) * 100 if articles else 0,
                'average_confidence': avg_confidence,
                'average_content_length': avg_content_length
            },
            'confidence_distribution': {
                'high_confidence': high_confidence,
                'medium_confidence': medium_confidence, 
                'low_confidence': low_confidence
            },
            'date_coverage': date_range,
            'recent_articles_30d': recent_articles,
            'sample_articles': [
                {
                    'title': a.title,
                    'date': a.publication_date,
                    'confidence': a.extraction_confidence,
                    'url_slug': a.url.split('/')[-2] if '/' in a.url else a.url
                }
                for a in sorted(articles, key=lambda x: x.extraction_confidence, reverse=True)[:5]
            ]
        }


# Global instance - will be initialized when used
accurate_scraper = None

def initialize_scraper(firecrawl_api_key: Optional[str] = None) -> AccurateArticleScraper:
    """Initialize the accurate scraper."""
    global accurate_scraper
    accurate_scraper = AccurateArticleScraper(firecrawl_api_key=firecrawl_api_key)
    return accurate_scraper

def scrape_all_accurate_articles(firecrawl_api_key: Optional[str] = None) -> Tuple[List[AccurateArticle], Dict]:
    """Scrape all articles with ultra-accuracy and return results + report."""
    scraper = initialize_scraper(firecrawl_api_key)
    articles = scraper.scrape_all_articles(max_workers=3, delay=1.5)
    report = scraper.generate_accuracy_report(articles)
    
    # Save results
    if articles:
        saved_path = scraper.save_scraped_articles(articles)
        report['saved_to'] = str(saved_path)
    
    return articles, report
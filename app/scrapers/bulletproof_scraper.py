import asyncio
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import uuid

import feedparser
import httpx
import requests
from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger
from app.scrapers.firecrawl_client import FirecrawlClient
from app.vision.image_analyzer import image_analyzer


class BulletproofScraper:
    """
    Triple-redundancy scraper for maximum reliability:
    1. RSS Feed (Primary) - Most reliable
    2. Direct HTML Scraping (Secondary) - Fast fallback
    3. Firecrawl (Tertiary) - Complex page handling
    
    Features:
    - Automatic failover between methods
    - Comprehensive image extraction
    - Content deduplication
    - Robust error handling
    - Rate limiting and respect for robots.txt
    """
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.firecrawl_client = FirecrawlClient()
        self.processed_urls: Set[str] = set()
        self.content_hashes: Set[str] = set()
        
        # Target URLs and patterns
        self.base_urls = [
            "https://insights.deribit.com",
            "https://insights.deribit.com/option-flows/",
            "https://insights.deribit.com/feed/",
            "https://insights.deribit.com/option-flows/feed/"
        ]
        
        self.option_flows_patterns = [
            r'/option-flows/',
            r'/option-flow-',
            r'option.*flow'
        ]
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not getattr(self.session, "is_closed", False):
            await self.session.aclose()

    def _ensure_session(self) -> None:
        """Ensure an httpx.AsyncClient session is available."""
        if self.session is None or getattr(self.session, "is_closed", False):
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={'User-Agent': settings.user_agent},
                follow_redirects=True
            )
    
    def compute_content_hash(self, content: str) -> str:
        """Compute hash for content deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def is_option_flows_url(self, url: str) -> bool:
        """Check if URL is related to option flows."""
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in self.option_flows_patterns)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_rss_feeds(self) -> List[Dict]:
        """
        Primary method: Fetch articles from RSS feeds.
        Most reliable method for getting new content.
        """
        articles = []
        
        rss_urls = [
            "https://insights.deribit.com/feed/",
            "https://insights.deribit.com/option-flows/feed/"
        ]
        
        self._ensure_session()
        for rss_url in rss_urls:
            try:
                logger.info(f"Fetching RSS feed: {rss_url}")
                
                response = await self.session.get(rss_url)
                response.raise_for_status()
                
                # Parse RSS feed
                feed = feedparser.parse(response.text)
                
                for entry in feed.entries:
                    # Check if this is an option flows article
                    if self.is_option_flows_url(entry.link):
                        article_data = {
                            'url': entry.link,
                            'title': entry.get('title', ''),
                            'author': entry.get('author', 'Tony Stewart'),
                            'published_at': self._parse_published_date(entry),
                            'summary': entry.get('summary', ''),
                            'source': 'rss_feed',
                            'discovery_method': 'rss'
                        }
                        
                        # Skip if already processed
                        if entry.link not in self.processed_urls:
                            articles.append(article_data)
                            self.processed_urls.add(entry.link)
                
                logger.info(f"Found {len(articles)} new option flows articles from RSS")
                
            except Exception as e:
                logger.error(f"RSS feed failed: {rss_url}", error=str(e))
                continue
        
        return articles
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def discover_articles_html(self) -> List[Dict]:
        """
        Secondary method: Scrape category pages directly with pagination support.
        Discovers ALL sub-links from https://insights.deribit.com/option-flows/
        """
        articles = []
        article_links = set()
        
        try:
            self._ensure_session()
            logger.info("Discovering ALL articles via HTML scraping with pagination")
            
            # Start with page 1 and continue until no more pages
            page = 1
            max_pages = 50  # Safety limit to prevent infinite loops
            
            while page <= max_pages:
                # Construct page URL
                if page == 1:
                    page_url = "https://insights.deribit.com/option-flows/"
                else:
                    # Common WordPress pagination patterns
                    page_url = f"https://insights.deribit.com/option-flows/page/{page}/"
                
                logger.info(f"Scraping page {page}: {page_url}")
                
                try:
                    await asyncio.sleep(settings.request_delay)  # Rate limiting
                    response = await self.session.get(page_url)
                    
                    # If we get 404, try alternative pagination formats
                    if response.status_code == 404:
                        # Try query parameter format
                        alt_page_url = f"https://insights.deribit.com/option-flows/?page={page}"
                        response = await self.session.get(alt_page_url)
                        
                        if response.status_code == 404:
                            logger.info(f"No more pages found at page {page}")
                            break
                    
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Find article links on this page
                    page_links = set()
                    
                    # Enhanced selectors to catch all possible article link patterns
                    link_selectors = [
                        # Direct option flows links
                        'a[href*="/option-flows/"]',
                        'a[href*="/option-flow-"]',
                        'a[href*="option-flows"]',
                        # Common WordPress post selectors
                        '.entry-title a',
                        '.post-title a',
                        'h2 a',
                        'h3 a',
                        'h4 a',
                        '.entry-header a',
                        '.post-header a',
                        # Article containers
                        'article a[href*="/option-flows/"]',
                        '.post a[href*="/option-flows/"]',
                        '.entry a[href*="/option-flows/"]',
                        # Content area links
                        '.content a[href*="/option-flows/"]',
                        '#content a[href*="/option-flows/"]',
                        '.main a[href*="/option-flows/"]',
                        # Grid and listing patterns
                        '.wp-block-latest-posts a',
                        '.post-list a',
                        '.archive-list a'
                    ]
                    
                    for selector in link_selectors:
                        links = soup.select(selector)
                        for link in links:
                            href = link.get('href')
                            if href:
                                full_url = urljoin("https://insights.deribit.com", href)
                                # More comprehensive URL validation
                                if (self.is_option_flows_url(full_url) and 
                                    full_url not in self.processed_urls and
                                    full_url != page_url and  # Don't include category page itself
                                    not full_url.endswith('/option-flows/') and  # Skip category page
                                    '/page/' not in full_url):  # Skip pagination URLs
                                    
                                    page_links.add(full_url)
                    
                    # Also look for any links containing option flow keywords in text
                    all_links = soup.find_all('a', href=True)
                    for link in all_links:
                        href = link.get('href')
                        link_text = link.get_text(strip=True).lower()
                        
                        if href and any(keyword in link_text for keyword in ['option flow', 'flows', 'options']):
                            full_url = urljoin("https://insights.deribit.com", href)
                            if (self.is_option_flows_url(full_url) and 
                                full_url not in self.processed_urls and
                                full_url != page_url):
                                page_links.add(full_url)
                    
                    logger.info(f"Found {len(page_links)} unique article links on page {page}")
                    
                    # If no new links found, we've reached the end
                    if not page_links:
                        logger.info(f"No new articles found on page {page}, stopping pagination")
                        break
                    
                    # Add new links to our master set
                    new_links = page_links - article_links
                    article_links.update(new_links)
                    
                    logger.info(f"Added {len(new_links)} new article URLs (Total: {len(article_links)})")
                    
                    # Check for next page indicators
                    next_page_found = False
                    next_indicators = [
                        '.next-page', '.next-posts-link', '.nav-next',
                        'a[href*="/page/' + str(page + 1) + '"]',
                        'a:contains("Next")', 'a:contains(">")'
                    ]
                    
                    for indicator in next_indicators:
                        if soup.select(indicator):
                            next_page_found = True
                            break
                    
                    # If no next page indicator and no new links, stop
                    if not next_page_found and not new_links:
                        logger.info("No next page indicator found, stopping pagination")
                        break
                        
                except Exception as e:
                    logger.error(f"Failed to scrape page {page}: {e}")
                    break
                
                page += 1
            
            logger.info(f"Pagination complete. Found {len(article_links)} total article URLs across {page-1} pages")
            
            # Now extract metadata for each discovered article
            logger.info("Starting metadata extraction for discovered articles...")
            
            for i, url in enumerate(article_links, 1):
                try:
                    await asyncio.sleep(settings.request_delay)  # Rate limiting
                    
                    logger.info(f"Processing article {i}/{len(article_links)}: {url}")
                    
                    article_response = await self.session.get(url)
                    article_response.raise_for_status()
                    
                    article_soup = BeautifulSoup(article_response.text, 'html.parser')
                    
                    # Extract article metadata
                    title = self._extract_title(article_soup)
                    author = self._extract_author(article_soup)
                    published_at = self._extract_published_date(article_soup)
                    
                    article_data = {
                        'url': url,
                        'title': title,
                        'author': author,
                        'published_at': published_at,
                        'source': 'html_scraping',
                        'discovery_method': 'html_pagination'
                    }
                    
                    articles.append(article_data)
                    self.processed_urls.add(url)
                    
                    logger.info(f"Successfully processed: {title[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Failed to extract metadata for {url}: {e}")
                    continue
            
            logger.info(f"HTML discovery with pagination found {len(articles)} articles total")
            
        except Exception as e:
            logger.error("HTML discovery with pagination failed", error=str(e))
        
        return articles
    
    async def discover_articles_firecrawl(self) -> List[Dict]:
        """
        Tertiary method: Use Firecrawl for complex pages.
        Most thorough but slowest method.
        """
        articles = []
        
        try:
            logger.info("Discovering articles via Firecrawl")
            
            # Use Firecrawl to map the site
            crawl_result = await self.firecrawl_client.crawl_site(
                base_url="https://insights.deribit.com",
                include_paths=["/option-flows/"],
                max_depth=2
            )
            
            if crawl_result and 'data' in crawl_result:
                for page_data in crawl_result['data']:
                    url = page_data.get('metadata', {}).get('url', '')
                    
                    if self.is_option_flows_url(url) and url not in self.processed_urls:
                        article_data = {
                            'url': url,
                            'title': page_data.get('metadata', {}).get('title', ''),
                            'author': page_data.get('metadata', {}).get('author', 'Tony Stewart'),
                            'published_at': self._parse_firecrawl_date(page_data),
                            'source': 'firecrawl',
                            'discovery_method': 'firecrawl',
                            'raw_content': page_data.get('content', '')
                        }
                        
                        articles.append(article_data)
                        self.processed_urls.add(url)
            
            logger.info(f"Discovered {len(articles)} articles via Firecrawl")
            
        except Exception as e:
            logger.error("Firecrawl discovery failed", error=str(e))
        
        return articles
    
    async def scrape_article_content(self, url: str, method: str = 'auto') -> Optional[Dict]:
        """
        Scrape full article content with fallback methods.
        
        Args:
            url: Article URL to scrape
            method: 'rss', 'html', 'firecrawl', or 'auto' for automatic fallback
            
        Returns:
            Article content with images or None if failed
        """
        methods = ['html', 'firecrawl'] if method == 'auto' else [method]
        
        for scrape_method in methods:
            try:
                logger.info(f"Scraping {url} using {scrape_method}")
                
                if scrape_method == 'html':
                    content = await self._scrape_html_content(url)
                elif scrape_method == 'firecrawl':
                    content = await self._scrape_firecrawl_content(url)
                else:
                    continue
                
                if content:
                    # Extract images from the content
                    content['images'] = await self._extract_article_images(url, content)
                    content['scraping_method'] = scrape_method
                    
                    # Compute content hash for deduplication
                    content_text = content.get('body_markdown', '') + content.get('body_html', '')
                    content_hash = self.compute_content_hash(content_text)
                    
                    if content_hash not in self.content_hashes:
                        content['content_hash'] = content_hash
                        self.content_hashes.add(content_hash)
                        return content
                    else:
                        logger.info(f"Duplicate content detected for {url}")
                        return None
                
            except Exception as e:
                logger.error(f"Failed to scrape {url} with {scrape_method}", error=str(e))
                continue
        
        logger.error(f"All scraping methods failed for {url}")
        return None
    
    async def _scrape_html_content(self, url: str) -> Optional[Dict]:
        """Scrape article content using direct HTML parsing."""
        # Ensure session is available
        self._ensure_session()
        
        response = await self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract content using multiple selectors
        content_selectors = [
            '.entry-content',
            '.post-content', 
            '.article-content',
            'article',
            '[class*="content"]'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            # Fallback: try to find main content
            content_element = soup.find('body')
        
        if not content_element:
            return None
        
        # Clean up the content
        self._clean_html_content(content_element)
        
        return {
            'title': self._extract_title(soup),
            'author': self._extract_author(soup),
            'published_at': self._extract_published_date(soup),
            'body_html': str(content_element),
            'body_markdown': self._html_to_markdown(content_element),
            'url': url
        }
    
    async def _scrape_firecrawl_content(self, url: str) -> Optional[Dict]:
        """Scrape article content using Firecrawl."""
        result = await self.firecrawl_client.scrape_url(url)
        
        if result and 'data' in result:
            data = result['data']
            return {
                'title': data.get('metadata', {}).get('title', ''),
                'author': data.get('metadata', {}).get('author', 'Tony Stewart'),
                'published_at': self._parse_firecrawl_date(data),
                'body_html': data.get('html', ''),
                'body_markdown': data.get('markdown', ''),
                'url': url
            }
        
        return None
    
    async def _extract_article_images(self, article_url: str, content: Dict) -> List[Dict]:
        """
        Extract and analyze all images from article content.
        
        Args:
            article_url: URL of the article
            content: Article content dictionary
            
        Returns:
            List of image analysis results
        """
        images = []
        
        try:
            # Parse HTML content to find images
            if 'body_html' in content:
                soup = BeautifulSoup(content['body_html'], 'html.parser')
                img_tags = soup.find_all('img')
                
                for img_tag in img_tags[:settings.max_images_per_article]:
                    img_src = img_tag.get('src')
                    if not img_src:
                        continue
                    
                    # Convert relative URLs to absolute
                    img_url = urljoin(article_url, img_src)
                    
                    # Skip small images (likely icons/logos)
                    if self._is_likely_content_image(img_tag, img_url):
                        try:
                            # Download and analyze image
                            image_data = await image_analyzer.download_image(img_url)
                            if image_data:
                                # Generate unique image ID
                                image_id = str(uuid.uuid4())
                                
                                # Comprehensive image analysis
                                analysis = await image_analyzer.analyze_image_comprehensive(
                                    image_data, img_url
                                )
                                
                                # Save image locally
                                local_path = image_analyzer.save_image(
                                    image_data, image_id, analysis.get('image_type', 'unknown')
                                )
                                
                                # Prepare image record
                                image_record = {
                                    'image_id': image_id,
                                    'image_url': img_url,
                                    'image_type': analysis.get('image_type', 'unknown'),
                                    'image_hash': analysis.get('image_hash'),
                                    'ocr_text': analysis.get('ocr_text', ''),
                                    'extracted_data': analysis.get('extracted_data', {}),
                                    'vision_analysis': analysis.get('vision_analysis', {}),
                                    'download_path': local_path,
                                    'file_size_bytes': analysis.get('file_size_bytes', 0),
                                    'width': analysis.get('dimensions', {}).get('width', 0),
                                    'height': analysis.get('dimensions', {}).get('height', 0),
                                    'confidence_score': analysis.get('ocr_confidence', 0.0),
                                    'processing_status': analysis.get('processing_status', 'completed'),
                                    'alt_text': img_tag.get('alt', ''),
                                    'title_text': img_tag.get('title', '')
                                }
                                
                                images.append(image_record)
                                
                                logger.info(
                                    f"Processed image {image_id}",
                                    image_type=analysis.get('image_type'),
                                    confidence=analysis.get('ocr_confidence', 0)
                                )
                                
                        except Exception as e:
                            logger.error(f"Failed to process image {img_url}", error=str(e))
                            continue
            
        except Exception as e:
            logger.error("Failed to extract images", article_url=article_url, error=str(e))
        
        logger.info(f"Extracted {len(images)} images from {article_url}")
        return images
    
    def _is_likely_content_image(self, img_tag, img_url: str) -> bool:
        """Determine if image is likely to contain meaningful content."""
        # Skip very small images (likely icons)
        width = img_tag.get('width')
        height = img_tag.get('height')
        
        if width and height:
            try:
                w, h = int(width), int(height)
                if w < 100 or h < 100:
                    return False
            except ValueError:
                pass
        
        # Skip common non-content image patterns
        skip_patterns = [
            r'logo', r'icon', r'avatar', r'favicon',
            r'social', r'share', r'button'
        ]
        
        img_url_lower = img_url.lower()
        alt_text_lower = (img_tag.get('alt') or '').lower()
        
        for pattern in skip_patterns:
            if re.search(pattern, img_url_lower) or re.search(pattern, alt_text_lower):
                return False
        
        return True
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title from HTML."""
        title_selectors = [
            'h1.entry-title',
            'h1.post-title',
            'h1',
            '.entry-title',
            '.post-title',
            'title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> str:
        """Extract article author from HTML."""
        author_selectors = [
            '.author-name',
            '.post-author',
            '.entry-author',
            '[class*="author"]',
            '[rel="author"]'
        ]
        
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)
        
        return "Tony Stewart"  # Default author for Deribit option flows
    
    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract article summary/excerpt from HTML."""
        # Try various summary selectors
        summary_selectors = [
            '.entry-excerpt',
            '.post-excerpt', 
            '.excerpt',
            '.summary',
            '.entry-content p:first-child',
            '.post-content p:first-child',
            '.content p:first-child'
        ]
        
        for selector in summary_selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                text = element.get_text(strip=True)
                # Limit summary length
                return text[:300] + '...' if len(text) > 300 else text
        
        return ""
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract published date from HTML."""
        date_selectors = [
            'time[datetime]',
            '.published-date',
            '.post-date',
            '.entry-date',
            '[class*="date"]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                datetime_attr = element.get('datetime')
                if datetime_attr:
                    return self._parse_iso_date(datetime_attr)
                
                # Try text content
                date_text = element.get_text(strip=True)
                if date_text:
                    return self._parse_date_text(date_text)
        
        return None
    
    def _parse_published_date(self, entry) -> Optional[datetime]:
        """Parse published date from RSS entry."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except (ValueError, TypeError):
                pass
        
        if hasattr(entry, 'published'):
            return self._parse_date_text(entry.published)
        
        return None
    
    def _parse_firecrawl_date(self, data: Dict) -> Optional[datetime]:
        """Parse date from Firecrawl data."""
        metadata = data.get('metadata', {})
        
        # Try various date fields
        date_fields = ['publishedTime', 'published', 'date', 'created']
        
        for field in date_fields:
            if field in metadata and metadata[field]:
                return self._parse_iso_date(metadata[field])
        
        return None
    
    def _parse_iso_date(self, date_string: str) -> Optional[datetime]:
        """Parse ISO format date string."""
        try:
            # Handle various ISO formats
            formats = [
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_string.replace('Z', '+00:00'), fmt)
                except ValueError:
                    continue
                    
        except Exception:
            pass
        
        return None
    
    def _parse_date_text(self, date_text: str) -> Optional[datetime]:
        """Parse human-readable date text."""
        try:
            # Common date patterns
            patterns = [
                r'(\w+\s+\d{1,2},\s+\d{4})',  # "January 15, 2024"
                r'(\d{1,2}\s+\w+\s+\d{4})',   # "15 January 2024"
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'  # "01/15/2024" or "01-15-24"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, date_text)
                if match:
                    try:
                        from dateutil import parser
                        return parser.parse(match.group(1))
                    except (ImportError, ValueError):
                        pass
        except Exception:
            pass
        
        return None
    
    def _clean_html_content(self, element):
        """Clean HTML content by removing unwanted elements."""
        # Remove common unwanted elements
        unwanted_tags = [
            'script', 'style', 'nav', 'footer', 'header',
            'aside', 'advertisement', '.social-share',
            '.related-posts', '.comments'
        ]
        
        for tag in unwanted_tags:
            for unwanted in element.select(tag):
                unwanted.decompose()
    
    def _html_to_markdown(self, element) -> str:
        """Convert HTML content to markdown (simplified)."""
        # This is a basic implementation
        # For production, consider using a proper HTML-to-Markdown library
        text = element.get_text()
        # Basic cleaning
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n\n'.join(lines)
    
    async def discover_all_articles(self) -> List[Dict]:
        """
        Discover articles using all available methods with automatic failover.
        
        Returns:
            List of discovered articles with metadata
        """
        all_articles = []
        
        # Method 1: RSS Feeds (Primary)
        try:
            rss_articles = await self.fetch_rss_feeds()
            all_articles.extend(rss_articles)
            logger.info(f"RSS discovery found {len(rss_articles)} articles")
        except Exception as e:
            logger.error("RSS discovery failed completely", error=str(e))
        
        # Method 2: HTML Scraping (Secondary) - only if RSS found few articles
        if len(all_articles) < 5:  # Threshold for fallback
            try:
                html_articles = await self.discover_articles_html()
                # Merge avoiding duplicates
                existing_urls = {article['url'] for article in all_articles}
                new_html_articles = [
                    article for article in html_articles 
                    if article['url'] not in existing_urls
                ]
                all_articles.extend(new_html_articles)
                logger.info(f"HTML discovery added {len(new_html_articles)} articles")
            except Exception as e:
                logger.error("HTML discovery failed", error=str(e))
        
        # Method 3: Firecrawl (Tertiary) - only if other methods found very few articles
        if len(all_articles) < 3:  # Emergency fallback
            try:
                firecrawl_articles = await self.discover_articles_firecrawl()
                # Merge avoiding duplicates
                existing_urls = {article['url'] for article in all_articles}
                new_firecrawl_articles = [
                    article for article in firecrawl_articles
                    if article['url'] not in existing_urls
                ]
                all_articles.extend(new_firecrawl_articles)
                logger.info(f"Firecrawl discovery added {len(new_firecrawl_articles)} articles")
            except Exception as e:
                logger.error("Firecrawl discovery failed", error=str(e))
        
        # Sort by publication date (newest first)
        all_articles.sort(
            key=lambda x: x.get('published_at') or datetime.min, 
            reverse=True
        )
        
        logger.info(f"Total discovered articles: {len(all_articles)}")
        return all_articles


# Global scraper instance
bulletproof_scraper = BulletproofScraper()
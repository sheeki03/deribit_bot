import asyncio
import json
from typing import Dict, List, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class FirecrawlClient:
    """
    Client for interacting with your hosted Firecrawl instance.
    Handles crawling, scraping, and webhook management.
    """
    
    def __init__(self):
        self.base_url = settings.firecrawl_base_url.rstrip('/')
        self.api_url = settings.firecrawl_api_url
        self.api_key = settings.firecrawl_api_key
        
        # Build headers - only add Authorization if API key exists
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.client = httpx.AsyncClient(
            timeout=120.0,
            headers=headers
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def map_site(self, url: str, limit: int = 1000) -> Optional[Dict]:
        """
        Map a website to discover all URLs.
        
        Args:
            url: Base URL to map
            limit: Maximum number of URLs to return
            
        Returns:
            Mapping results or None if failed
        """
        try:
            payload = {
                "url": url,
                "ignoreSitemap": False,
                "sitemapOnly": False,
                "includeSubdomains": False,
                "limit": limit,
                "timeout": 60000
            }
            
            logger.info(f"Mapping site: {url}")
            
            response = await self.client.post(f"{self.base_url}/v1/map", json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Site mapping completed: {len(result.get('links', []))} URLs found")
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Firecrawl mapping failed with status {e.response.status_code}", 
                        url=url, error=str(e))
            return None
        except Exception as e:
            logger.error("Firecrawl mapping failed", url=url, error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def crawl_site(self, 
                         base_url: str, 
                         include_paths: List[str] = None,
                         exclude_paths: List[str] = None,
                         max_depth: int = 2,
                         limit: int = 500) -> Optional[Dict]:
        """
        Crawl a website with specified parameters.
        
        Args:
            base_url: Base URL to crawl
            include_paths: Paths to include in crawl
            exclude_paths: Paths to exclude from crawl
            max_depth: Maximum crawl depth
            limit: Maximum pages to crawl
            
        Returns:
            Crawl results or None if failed
        """
        try:
            payload = {
                "url": base_url,
                "maxDepth": max_depth,
                "ignoreSitemap": False,
                "limit": limit,
                "scrapeOptions": {
                    "onlyMainContent": True,
                    "formats": ["markdown", "html"],
                    "timeout": 120000,
                    "storeInCache": True
                }
            }
            
            if include_paths:
                payload["includePaths"] = include_paths
            
            if exclude_paths:
                payload["excludePaths"] = exclude_paths
            
            logger.info(f"Starting crawl: {base_url}")
            
            # Start crawl job
            response = await self.client.post(f"{self.base_url}/v1/crawl", json=payload)
            response.raise_for_status()
            
            crawl_response = response.json()
            job_id = crawl_response.get("id")
            
            if not job_id:
                logger.error("No job ID returned from crawl request")
                return None
            
            logger.info(f"Crawl job started: {job_id}")
            
            # Poll for completion
            max_wait_time = 300  # 5 minutes
            wait_interval = 10   # 10 seconds
            waited = 0
            
            while waited < max_wait_time:
                await asyncio.sleep(wait_interval)
                waited += wait_interval
                
                # Check job status
                status_response = await self.client.get(f"{self.base_url}/v1/crawl/{job_id}")
                status_response.raise_for_status()
                
                status_data = status_response.json()
                status = status_data.get("status")
                
                logger.info(f"Crawl job {job_id} status: {status}")
                
                if status == "completed":
                    logger.info(f"Crawl completed: {len(status_data.get('data', []))} pages")
                    return status_data
                elif status == "failed":
                    logger.error(f"Crawl job failed: {status_data.get('error', 'Unknown error')}")
                    return None
                elif status in ["scraping", "processing"]:
                    continue  # Keep waiting
                else:
                    logger.warning(f"Unknown crawl status: {status}")
            
            logger.error(f"Crawl job timed out after {max_wait_time} seconds")
            return None
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Firecrawl crawl failed with status {e.response.status_code}",
                        base_url=base_url, error=str(e))
            return None
        except Exception as e:
            logger.error("Firecrawl crawl failed", base_url=base_url, error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_url(self, 
                        url: str, 
                        include_images: bool = True,
                        extract_schema: bool = False) -> Optional[Dict]:
        """
        Scrape a single URL with detailed content extraction.
        
        Args:
            url: URL to scrape
            include_images: Whether to extract image information
            extract_schema: Whether to extract structured data
            
        Returns:
            Scraping results or None if failed
        """
        try:
            # Use simplified payload format that works with this Firecrawl instance
            payload = {
                "url": url,
                "formats": ["markdown", "html"]
            }
            
            # Add images to formats if needed (Firecrawl typically includes images in markdown automatically)
            if include_images:
                # Images are usually included automatically in the markdown/html output
                pass
            
            logger.info(f"Scraping URL: {url}")
            
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Validate response structure
            if "data" not in result:
                logger.error(f"Invalid response structure from Firecrawl for {url}")
                return None
            
            logger.info(f"Successfully scraped {url}")
            return result
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Rate limited by Firecrawl for {url}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
            else:
                logger.error(f"Firecrawl scraping failed with status {e.response.status_code}",
                            url=url, error=str(e))
            return None
        except Exception as e:
            logger.error("Firecrawl scraping failed", url=url, error=str(e))
            return None
    
    async def batch_scrape_urls(self, urls: List[str], max_concurrent: int = 5) -> List[Dict]:
        """
        Scrape multiple URLs concurrently with rate limiting.
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of scraping results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> Dict:
            async with semaphore:
                result = await self.scrape_url(url)
                await asyncio.sleep(1)  # Rate limiting
                return {"url": url, "result": result, "success": result is not None}
        
        logger.info(f"Starting batch scrape of {len(urls)} URLs")
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and failed results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Batch scrape task failed", error=str(result))
            elif result.get("success"):
                successful_results.append(result["result"])
            else:
                failed_count += 1
        
        logger.info(f"Batch scrape completed: {len(successful_results)} successful, {failed_count} failed")
        return successful_results
    
    async def get_crawl_status(self, job_id: str) -> Optional[Dict]:
        """Get the status of a crawl job."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/crawl/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get crawl status for {job_id}", error=str(e))
            return None
    
    async def cancel_crawl(self, job_id: str) -> bool:
        """Cancel a running crawl job."""
        try:
            response = await self.client.delete(f"{self.base_url}/v1/crawl/{job_id}")
            response.raise_for_status()
            logger.info(f"Crawl job {job_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel crawl {job_id}", error=str(e))
            return False


# Global Firecrawl client instance
firecrawl_client = FirecrawlClient()
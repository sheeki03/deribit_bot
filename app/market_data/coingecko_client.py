import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger


class CoinGeckoClient:
    """
    CoinGecko API client for fetching BTC/ETH price data.
    Perfect for comparing with option flows publication timestamps.
    
    Features:
    - Historical price data
    - Real-time price monitoring
    - Rate limiting compliance
    - Robust error handling
    """
    
    def __init__(self):
        self.base_url = settings.coingecko_base_url
        self.api_key = settings.coingecko_api_key
        
        headers = {
            'User-Agent': settings.user_agent,
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['x-cg-pro-api-key'] = self.api_key
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            headers=headers
        )
        
        # CoinGecko coin IDs
        self.coin_ids = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum'
        }
        
        # Rate limiting (free tier: 10-50 calls/min, pro: 500 calls/min)
        self.rate_limit_delay = 1.2 if not self.api_key else 0.2
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_current_prices(self, assets: List[str] = None) -> Dict[str, float]:
        """
        Get current prices for specified assets.
        
        Args:
            assets: List of assets ['BTC', 'ETH']. If None, gets both.
            
        Returns:
            Dictionary with asset prices in USD
        """
        if assets is None:
            assets = ['BTC', 'ETH']
        
        try:
            coin_ids = [self.coin_ids[asset] for asset in assets if asset in self.coin_ids]
            
            if not coin_ids:
                logger.warning("No valid assets provided", assets=assets)
                return {}
            
            # Make API call
            response = await self.client.get(
                "/simple/price",
                params={
                    'ids': ','.join(coin_ids),
                    'vs_currencies': 'usd',
                    'include_last_updated_at': 'true'
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Convert response to our format
            prices = {}
            for asset in assets:
                coin_id = self.coin_ids.get(asset)
                if coin_id in data:
                    prices[asset] = {
                        'price': data[coin_id]['usd'],
                        'timestamp': datetime.fromtimestamp(data[coin_id]['last_updated_at']),
                        'source': 'coingecko_current'
                    }
            
            logger.info(f"Fetched current prices for {len(prices)} assets")
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            return prices
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CoinGecko rate limit hit, waiting longer")
                await asyncio.sleep(60)
            logger.error(f"CoinGecko API error: {e.response.status_code}", error=str(e))
            return {}
        except Exception as e:
            logger.error("Failed to fetch current prices", error=str(e))
            return {}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_historical_prices(self, 
                                   asset: str, 
                                   from_timestamp: datetime,
                                   to_timestamp: datetime,
                                   interval: str = 'hourly') -> List[Dict]:
        """
        Get historical price data for a specific time range.
        
        Args:
            asset: Asset symbol ('BTC' or 'ETH')
            from_timestamp: Start timestamp
            to_timestamp: End timestamp  
            interval: 'hourly' or 'daily'
            
        Returns:
            List of price data points
        """
        if asset not in self.coin_ids:
            logger.error(f"Unsupported asset: {asset}")
            return []
        
        try:
            coin_id = self.coin_ids[asset]
            
            # Convert timestamps to Unix
            from_unix = int(from_timestamp.timestamp())
            to_unix = int(to_timestamp.timestamp())
            
            # Determine the appropriate endpoint
            time_diff = to_timestamp - from_timestamp
            
            if time_diff <= timedelta(days=1):
                # Use market_chart/range for detailed data
                endpoint = f"/coins/{coin_id}/market_chart/range"
                params = {
                    'vs_currency': 'usd',
                    'from': from_unix,
                    'to': to_unix
                }
            else:
                # Use market_chart for longer periods
                endpoint = f"/coins/{coin_id}/market_chart"
                days = min(time_diff.days, 365)  # CoinGecko limit
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': interval
                }
            
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse price data
            prices = []
            if 'prices' in data:
                for timestamp_ms, price in data['prices']:
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                    
                    # Filter to requested time range
                    if from_timestamp <= timestamp <= to_timestamp:
                        prices.append({
                            'timestamp': timestamp,
                            'price': price,
                            'asset': asset,
                            'source': 'coingecko_historical'
                        })
            
            logger.info(f"Fetched {len(prices)} historical prices for {asset}")
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            return prices
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
                await asyncio.sleep(60)
            logger.error(f"CoinGecko historical API error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error("Failed to fetch historical prices", asset=asset, error=str(e))
            return []
    
    async def get_price_at_timestamp(self, 
                                    asset: str, 
                                    target_timestamp: datetime,
                                    tolerance_minutes: int = 60) -> Optional[Dict]:
        """
        Get the price closest to a specific timestamp.
        Perfect for option flows event studies.
        
        Args:
            asset: Asset symbol ('BTC' or 'ETH')
            target_timestamp: Target timestamp
            tolerance_minutes: Maximum time difference to accept
            
        Returns:
            Price data or None if not found within tolerance
        """
        try:
            # Define search window
            from_time = target_timestamp - timedelta(minutes=tolerance_minutes)
            to_time = target_timestamp + timedelta(minutes=tolerance_minutes)
            
            # Get historical data for the window
            prices = await self.get_historical_prices(asset, from_time, to_time)
            
            if not prices:
                logger.warning(f"No price data found for {asset} around {target_timestamp}")
                return None
            
            # Find closest price
            closest_price = min(
                prices,
                key=lambda p: abs((p['timestamp'] - target_timestamp).total_seconds())
            )
            
            # Check if within tolerance
            time_diff = abs((closest_price['timestamp'] - target_timestamp).total_seconds())
            if time_diff <= tolerance_minutes * 60:
                closest_price['time_difference_seconds'] = time_diff
                return closest_price
            else:
                logger.warning(f"Closest price for {asset} is {time_diff/60:.1f} minutes away")
                return None
                
        except Exception as e:
            logger.error("Failed to get price at timestamp", asset=asset, error=str(e))
            return None
    
    async def get_forward_returns(self, 
                                 asset: str,
                                 base_timestamp: datetime,
                                 forward_hours: List[int] = [4, 24, 72, 168]) -> Dict[str, Optional[float]]:
        """
        Calculate forward returns from a base timestamp.
        Essential for option flows event studies.
        
        Args:
            asset: Asset symbol
            base_timestamp: Starting timestamp (e.g., article publication)
            forward_hours: List of forward looking hours [4h, 24h, 72h, 7d=168h]
            
        Returns:
            Dictionary with forward returns
        """
        returns = {}
        
        try:
            # Get base price
            base_price_data = await self.get_price_at_timestamp(asset, base_timestamp)
            
            if not base_price_data:
                logger.warning(f"No base price found for {asset} at {base_timestamp}")
                return {f"ret_{h}h": None for h in forward_hours}
            
            base_price = base_price_data['price']
            
            # Calculate returns for each forward period
            for hours in forward_hours:
                target_timestamp = base_timestamp + timedelta(hours=hours)
                
                forward_price_data = await self.get_price_at_timestamp(asset, target_timestamp)
                
                if forward_price_data:
                    forward_price = forward_price_data['price']
                    log_return = float('inf')
                    
                    if base_price > 0:
                        import math
                        log_return = math.log(forward_price / base_price)
                    
                    returns[f"ret_{hours}h"] = log_return
                    
                    logger.debug(f"{asset} {hours}h return: {log_return:.4f}")
                else:
                    returns[f"ret_{hours}h"] = None
            
            return returns
            
        except Exception as e:
            logger.error("Failed to calculate forward returns", asset=asset, error=str(e))
            return {f"ret_{h}h": None for h in forward_hours}
    
    async def get_price_data_for_articles(self, articles_data: List[Dict]) -> Dict[str, Dict]:
        """
        Batch fetch price data for multiple articles.
        Optimized for event study analysis.
        
        Args:
            articles_data: List of articles with timestamps and URLs
            
        Returns:
            Dictionary mapping article_id to price data
        """
        results = {}
        
        try:
            # Group by asset and batch process
            for article in articles_data:
                article_id = article.get('article_id') or article.get('url')
                published_at = article.get('published_at_utc') or article.get('published_at')
                
                if not published_at:
                    continue
                
                # Get prices for both BTC and ETH
                article_prices = {}
                
                for asset in ['BTC', 'ETH']:
                    # Get base price at publication
                    base_price = await self.get_price_at_timestamp(asset, published_at)
                    
                    if base_price:
                        # Get forward returns
                        forward_returns = await self.get_forward_returns(asset, published_at)
                        
                        article_prices[asset] = {
                            'base_price': base_price,
                            'forward_returns': forward_returns,
                            'published_at': published_at
                        }
                    
                    # Rate limiting between assets
                    await asyncio.sleep(self.rate_limit_delay)
                
                if article_prices:
                    results[article_id] = article_prices
                
                logger.info(f"Processed price data for article {article_id}")
            
            logger.info(f"Completed price data fetch for {len(results)} articles")
            
        except Exception as e:
            logger.error("Failed to batch process article price data", error=str(e))
        
        return results
    
    async def health_check(self) -> bool:
        """Check if CoinGecko API is accessible."""
        try:
            response = await self.client.get("/ping")
            response.raise_for_status()
            data = response.json()
            
            is_healthy = data.get('gecko_says') == '(V3) To the Moon!'
            logger.info(f"CoinGecko health check: {'✓' if is_healthy else '✗'}")
            
            return is_healthy
        except Exception as e:
            logger.error("CoinGecko health check failed", error=str(e))
            return False


# Global CoinGecko client instance
coingecko_client = CoinGeckoClient()
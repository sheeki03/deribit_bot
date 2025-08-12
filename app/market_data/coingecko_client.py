import asyncio
import time
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
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
        
        # Cache & rate/quota
        self.cache_dir = Path(settings.data_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._daily_cache: Dict[str, Dict[str, float]] = {}
        self.quota_path = self.cache_dir / "coingecko_quota.json"
        # Rate limiting: ≤ 30 req/min
        self.rate_limit_delay = 2.2 if not self.api_key else 0.3
        self.max_requests_per_minute = 30
        self._last_request_time: float = 0.0

        # Asset inception dates (approx)
        # We only need data from 2022 onwards per project scope
        self.asset_start_date = {
            'BTC': date(2022, 1, 1),
            'ETH': date(2022, 1, 1),
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # --------------- Quota & Rate helpers ---------------
    def _load_quota(self) -> dict:
        if self.quota_path.exists():
            try:
                return json.loads(self.quota_path.read_text())
            except Exception:
                return {}
        return {}

    def _save_quota(self, data: dict) -> None:
        try:
            self.quota_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    async def _respect_rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _check_and_inc_quota(self, n: int = 1) -> bool:
        today_str = datetime.now(timezone.utc).date().isoformat()
        data = self._load_quota()
        day_record = data.get(today_str, 0)
        if day_record + n > 10000:
            return False
        data[today_str] = day_record + n
        self._save_quota(data)
        return True

    # --------------- Daily cache helpers ---------------
    def _daily_cache_path(self, asset: str) -> Path:
        return self.cache_dir / f"coingecko_daily_{asset}.json"

    def _load_daily_cache(self, asset: str) -> Dict[str, float]:
        if asset in self._daily_cache:
            return self._daily_cache[asset]
        path = self._daily_cache_path(asset)
        cache: Dict[str, float] = {}
        if path.exists():
            try:
                cache = json.loads(path.read_text())
            except Exception:
                cache = {}
        self._daily_cache[asset] = cache
        return cache

    def _save_daily_cache(self, asset: str) -> None:
        cache = self._daily_cache.get(asset, {})
        path = self._daily_cache_path(asset)
        try:
            path.write_text(json.dumps(cache, indent=2))
        except Exception:
            pass

    def _clamp_dates(self, asset: str, start: date, end: date) -> Tuple[date, date]:
        start0 = self.asset_start_date.get(asset, date(2012, 1, 1))
        today = datetime.now(timezone.utc).date()
        s = max(start, start0)
        e = min(end, today)
        if e < s:
            e = s
        return s, e

    async def ensure_daily_cache_for_range(self, asset: str, start: date, end: date) -> None:
        s, e_ = self._clamp_dates(asset, start, end)
        cache = self._load_daily_cache(asset)
        # Detect missing days
        missing: List[date] = []
        d = s
        while d <= e_:
            if d.isoformat() not in cache:
                missing.append(d)
            d += timedelta(days=1)
        if not missing:
            return
        # Fetch in ~60-day windows using range API; fill cache with last price per day
        win = missing[0]
        while win <= missing[-1]:
            win_end = min(win + timedelta(days=60), e_)
            from_dt = datetime.combine(win, datetime.min.time(), tzinfo=timezone.utc)
            to_dt = datetime.combine(win_end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            if not self._check_and_inc_quota(1):
                logger.warning("Quota reached for CoinGecko daily prefetch")
                break
            await self._respect_rate_limit()
            try:
                coin_id = self.coin_ids[asset]
                resp = await self.client.get(
                    f"/coins/{coin_id}/market_chart/range",
                    params={'vs_currency': 'usd', 'from': int(from_dt.timestamp()), 'to': int(to_dt.timestamp())}
                )
                resp.raise_for_status()
                data = resp.json()
                day_last: Dict[str, float] = {}
                for ts_ms, price in data.get('prices', []):
                    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                    day_last[ts.date().isoformat()] = price
                dd = win
                while dd <= win_end:
                    k = dd.isoformat()
                    if k in day_last:
                        cache[k] = day_last[k]
                    dd += timedelta(days=1)
                self._save_daily_cache(asset)
            except httpx.HTTPStatusError as e:
                logger.error(f"CoinGecko range error: {e.response.status_code}")
            except Exception as e:
                logger.error("Daily cache prefetch failed", error=str(e))
            win = win_end + timedelta(days=1)
    
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
                                    interval: str = 'hourly',
                                    daily_average: bool = False) -> List[Dict]:
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
            
            if daily_average:
                # Force daily resolution using market_chart with 'daily' interval
                endpoint = f"/coins/{coin_id}/market_chart"
                days = max(1, min(time_diff.days + 1, 90))
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily'
                }
            elif time_diff <= timedelta(days=1):
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
            
            # Quota & rate limiting for all network calls
            if not self._check_and_inc_quota(1):
                logger.warning("CoinGecko daily quota reached; aborting historical fetch")
                return []
            await self._respect_rate_limit()
            response = await self.client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse price data with optional daily averaging
            prices = []
            if 'prices' in data:
                if daily_average:
                    # Use the daily points returned by API (already averaged per day by CoinGecko)
                    for timestamp_ms, price in data['prices']:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                        if from_timestamp.date() <= timestamp.date() <= to_timestamp.date():
                            prices.append({
                                'timestamp': timestamp,
                                'price': price,
                                'asset': asset,
                                'source': 'coingecko_daily'
                            })
                else:
                    for timestamp_ms, price in data['prices']:
                        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
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
                                    tolerance_minutes: int = 60,
                                    use_daily: bool = True) -> Optional[Dict]:
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
            if use_daily:
                from_time = target_timestamp - timedelta(days=1)
                to_time = target_timestamp + timedelta(days=1)
            else:
                from_time = target_timestamp - timedelta(minutes=tolerance_minutes)
                to_time = target_timestamp + timedelta(minutes=tolerance_minutes)
            
            # Get historical data for the window
            prices = await self.get_historical_prices(asset, from_time, to_time, daily_average=use_daily)
            
            if not prices:
                logger.warning(f"No price data found for {asset} around {target_timestamp}")
                return None
            
            # Find closest price (by day when daily mode)
            if use_daily:
                target_day = target_timestamp.date()
                same_day = [p for p in prices if p['timestamp'].date() == target_day]
                chosen = same_day[0] if same_day else min(
                    prices, key=lambda p: abs((p['timestamp'] - target_timestamp).total_seconds())
                )
                closest_price = chosen
            else:
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
            base_price_data = await self.get_price_at_timestamp(asset, base_timestamp, use_daily=True)
            
            if not base_price_data:
                logger.warning(f"No base price found for {asset} at {base_timestamp}")
                return {f"ret_{h}h": None for h in forward_hours}
            
            base_price = base_price_data['price']
            
            # Calculate returns for each forward period
            for hours in forward_hours:
                target_timestamp = base_timestamp + timedelta(hours=hours)
                
                forward_price_data = await self.get_price_at_timestamp(asset, target_timestamp, use_daily=True)
                
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
    
    async def get_historical_price_range(self, coin_id: str, from_date: str, to_date: str, vs_currency: str = "usd") -> Optional[List]:
        """
        Get historical price data for a date range (daily data).
        
        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            from_date: Start date in DD-MM-YYYY format
            to_date: End date in DD-MM-YYYY format
            vs_currency: Currency to price against
            
        Returns:
            List of [timestamp, price] pairs or None
        """
        try:
            # Validate and parse date inputs
            try:
                parsed_from_date = datetime.strptime(from_date, '%d-%m-%Y')
                parsed_to_date = datetime.strptime(to_date, '%d-%m-%Y')
            except ValueError as e:
                raise ValueError(f"Invalid date format. Expected DD-MM-YYYY, got from_date='{from_date}', to_date='{to_date}'. Error: {e}")
            
            # Convert parsed dates to timestamps
            from_timestamp = int(parsed_from_date.timestamp())
            to_timestamp = int(parsed_to_date.timestamp())
            
            response = await self.client.get(
                f"/coins/{coin_id}/market_chart/range",
                params={
                    'vs_currency': vs_currency,
                    'from': from_timestamp,
                    'to': to_timestamp
                }
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Return price data (array of [timestamp, price])
            prices = data.get('prices', [])
            logger.info(f"Fetched {len(prices)} price points for {coin_id} from {from_date} to {to_date}")
            
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            return prices
            
        except Exception as e:
            logger.error("Failed to get historical price range", coin_id=coin_id, 
                        from_date=from_date, to_date=to_date, error=str(e))
            return None

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
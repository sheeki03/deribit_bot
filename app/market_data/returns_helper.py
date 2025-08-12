from __future__ import annotations

from datetime import datetime, timezone, timedelta, date
import logging
from typing import Dict, List, Optional

from app.market_data.coingecko_client import coingecko_client

logger = logging.getLogger(__name__)

# Asset inception (mirror of coingecko_client and event_study)
# Safe default start date used when an asset is missing from ASSET_START.
# Chosen as an early reasonable bound to avoid querying far past history unnecessarily.
DEFAULT_ASSET_START_DATE = date(2012, 1, 1)
ASSET_START = {
    'BTC': date(2022, 1, 1),
    'ETH': date(2022, 1, 1),
}


def _parse_iso_dt(pub_dt: datetime | str) -> Optional[datetime]:
    if isinstance(pub_dt, datetime):
        return pub_dt
    if isinstance(pub_dt, str):
        try:
            # Accept trailing Z as UTC
            return datetime.fromisoformat(pub_dt.replace('Z', '+00:00'))
        except Exception:
            return None
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def get_realized_returns_at(
    pub_dt: datetime | str,
    horizons: List[int] | None = None,
    assets: List[str] | None = None,
    client=coingecko_client,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute realized forward returns for each asset at publication timestamp with caching.

    Behavior:
    - Parses input timestamp robustly (handles trailing 'Z')
    - Clamps publication date to 'now' (UTC) to avoid future lookups
    - Prefetches daily cache across [pub_day, pub_day + max(horizons)] per asset
    - Clamps to asset inception dates

    Returns: { 'BTC': {'ret_24h': ..., 'ret_72h': ...}, 'ETH': {...} }
    """
    if horizons is None:
        horizons = [24, 72, 168]
    if assets is None:
        assets = ['BTC', 'ETH']

    dt = _parse_iso_dt(pub_dt)
    if dt is None:
        return {a: {f"ret_{h}h": None for h in horizons} for a in assets}

    # Ensure timezone-aware (assume UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Clamp future dates to now
    now_utc = _utc_now()
    if dt > now_utc:
        dt = now_utc

    # Prefetch daily cache for [pub_day, pub_day + max_h]
    try:
        pub_day = dt.date()
        fwd_days = max(horizons) // 24 + 1
        end_day = pub_day + timedelta(days=fwd_days)
        for a in assets:
            start = pub_day
            # Clamp to asset inception
            start = max(start, ASSET_START.get(a, DEFAULT_ASSET_START_DATE))
            await client.ensure_daily_cache_for_range(a, start, end_day)
    except Exception:
        # Prefetch failures should not abort returns computation, but log the error for visibility
        logger.exception("Failed to prefetch daily cache for forward returns", extra={
            'assets': assets,
            'horizons': horizons,
            'publication_dt': dt.isoformat(),
        })

    results: Dict[str, Dict[str, Optional[float]]] = {}
    for a in assets:
        try:
            rets = await client.get_forward_returns(a, dt, horizons)
        except Exception:
            rets = {f"ret_{h}h": None for h in horizons}
        results[a] = rets
    return results

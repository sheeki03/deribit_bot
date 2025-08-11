import argparse
import asyncio
import csv
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import List, Dict, Any

from app.market_data.coingecko_client import coingecko_client

# Asset inception dates
ASSET_START = {
    'BTC': date(2012, 1, 1),
    'ETH': date(2015, 8, 7),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute forward returns for cleaned articles")
    p.add_argument("--cleaned-json", default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--hours", type=str, default="4,24,72,168", help="Comma-separated forward hours")
    p.add_argument("--output", default=str(Path("test_results/event_study.csv").resolve()))
    return p.parse_args()


async def compute_for_article(article: Dict[str, Any], forward_hours: List[int]) -> Dict[str, Any]:
    url = article.get("url")
    title = article.get("title")
    pub = article.get("published_at_utc") or article.get("published_at")
    if not pub:
        return {"url": url, "title": title, "published_at_utc": None}
    if isinstance(pub, str):
        pub_dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
    else:
        pub_dt = pub

    result = {"url": url, "title": title, "published_at_utc": pub_dt.isoformat()}
    for asset in ["BTC", "ETH"]:
        base = await coingecko_client.get_price_at_timestamp(asset, pub_dt, use_daily=True)
        result[f"{asset}_price_at_pub"] = base.get("price") if base else None
        forwards = await coingecko_client.get_forward_returns(asset, pub_dt, forward_hours)
        for h in forward_hours:
            result[f"{asset}_ret_{h}h"] = forwards.get(f"ret_{h}h")
    return result


async def main_async(args: argparse.Namespace):
    import json
    cleaned = Path(args.cleaned_json)
    articles: List[Dict[str, Any]] = json.loads(cleaned.read_text())
    forward_hours = [int(x.strip()) for x in args.hours.split(',') if x.strip()]
    subset = articles[: args.limit]

    # Build distinct valid dates and prefetch caches
    def clamp_day(dt: datetime) -> date:
        d = dt.date()
        today = datetime.now(timezone.utc).date()
        if d > today:
            d = today
        return d

    pub_days = []
    for a in subset:
        pub = a.get('published_at_utc') or a.get('published_at')
        if not pub:
            continue
        if isinstance(pub, str):
            try:
                dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
            except Exception:
                continue
        else:
            dt = pub
        pub_days.append(clamp_day(dt))

    if pub_days:
        min_day = min(pub_days)
        max_day = max(pub_days)
        # Expand by +7 days for forward returns
        prefetch_start_btc = max(min_day, ASSET_START['BTC'])
        prefetch_start_eth = max(min_day, ASSET_START['ETH'])
        await coingecko_client.ensure_daily_cache_for_range('BTC', prefetch_start_btc, max_day + timedelta(days=7))
        await coingecko_client.ensure_daily_cache_for_range('ETH', prefetch_start_eth, max_day + timedelta(days=7))

    tasks = [compute_for_article(a, forward_hours) for a in subset]
    rows = await asyncio.gather(*tasks)

    # Write CSV
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Collect all keys
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = ["url", "title", "published_at_utc"] + sorted(k for k in keys if k not in {"url", "title", "published_at_utc"})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()



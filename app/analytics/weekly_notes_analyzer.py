from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from app.validation.article_filters import is_weekly_note
from app.ml.sentiment_service import score_article_text
from app.market_data.returns_helper import get_realized_returns_at
 


@dataclass
class AnalysisDeps:
    sentiment_scorer: Callable[[Optional[str], Optional[str]], Dict[str, Any]]
    returns_helper: Callable[[datetime | str, List[int] | None, List[str] | None], Awaitable[Dict[str, Dict[str, Optional[float]]]]]


def is_weekly_article(article: Dict[str, Any]) -> bool:
    url = article.get("url")
    title = article.get("title")
    rss_cat = article.get("rss_category") or article.get("category")
    return is_weekly_note(url, title, rss_cat)


def _format_published_at(pub) -> str:
    """Format published_at value to string with explicit checks."""
    if isinstance(pub, str):
        return pub
    elif pub is not None and hasattr(pub, 'isoformat') and callable(getattr(pub, 'isoformat')):
        return pub.isoformat()
    else:
        return ""


async def analyze_article(article: Dict[str, Any], deps: AnalysisDeps, horizons: List[int]) -> Optional[Dict[str, Any]]:
    if not is_weekly_article(article):
        return None

    title = article.get("title")
    body = article.get("body_text") or article.get("body_markdown") or article.get("body_html") or ""
    pub = article.get("published_at_utc") or article.get("published_at")
    if not pub or not body:
        return None

    # Sentiment
    sent = deps.sentiment_scorer(title, body)
    sentiment = sent.get("sentiment", "neutral")
    confidence = float(sent.get("confidence", 0.0))

    # Returns (BTC/ETH)
    rets = await deps.returns_helper(pub, horizons, ["BTC", "ETH"])  # type: ignore[arg-type]

    row: Dict[str, Any] = {
        "url": article.get("url"),
        "title": title,
        "published_at_utc": _format_published_at(pub),
        "sentiment": sentiment,
        "confidence": confidence,
    }
    for asset in ["BTC", "ETH"]:
        asset_rets = rets.get(asset, {})
        for h in horizons:
            row[f"{asset}_ret_{h}h"] = asset_rets.get(f"ret_{h}h")
    return row


async def analyze_articles(articles: Iterable[Dict[str, Any]], deps: AnalysisDeps, horizons: List[int]) -> List[Dict[str, Any]]:
    tasks = [analyze_article(a, deps, horizons) for a in articles]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def default_deps() -> AnalysisDeps:
    return AnalysisDeps(
        sentiment_scorer=lambda title, body: score_article_text(title, body),
        returns_helper=lambda pub_dt, horizons=None, assets=None: get_realized_returns_at(pub_dt, horizons=horizons, assets=assets),
    )


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Collect all fieldnames
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    # Prefer consistent ordering
    preferred = ["url", "title", "published_at_utc", "sentiment", "confidence"]
    fieldnames = preferred + [k for k in keys if k not in preferred]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weekly Notes: per-article sentiment + returns analyzer")
    p.add_argument("--cleaned-json", default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()))
    p.add_argument("--output", default=str(Path("test_results/weekly_notes_analysis.csv").resolve()))
    p.add_argument("--hours", default="24,72,168", help="Comma-separated forward hours list")
    p.add_argument("--limit", type=int, default=0, help="Optional limit of articles to process (0=all)")
    p.add_argument("--persist-db", action="store_true", help="If set, upsert results into Postgres (DATABASE_URL required)")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    import json
    cleaned = Path(args.cleaned_json)
    if not cleaned.exists():
        raise SystemExit(f"Cleaned JSON not found: {cleaned}")

    articles: List[Dict[str, Any]] = json.loads(cleaned.read_text())
    if args.limit and args.limit > 0:
        articles = articles[: args.limit]

    horizons = [int(x.strip()) for x in args.hours.split(',') if x.strip()]

    deps = default_deps()
    rows = asyncio.run(analyze_articles(articles, deps, horizons))

    out = Path(args.output)
    write_csv(rows, out)
    print(f"Wrote {len(rows)} rows to {out}")

    # Optional DB persistence
    if args.persist_db:
        try:
            from app.analytics import db_weekly_notes
            with db_weekly_notes.connect() as conn:
                db_weekly_notes.ensure_table(conn)
                upserted = db_weekly_notes.upsert_weekly_note_analyses(conn, rows)
                print(f"Upserted {upserted} rows into weekly_note_analyses")
        except Exception as e:
            # Do not fail CLI if DB not available
            print(f"Warning: DB persistence skipped due to error: {e}")


if __name__ == "__main__":
    main()

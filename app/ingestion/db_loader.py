import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg.extras import execute_values

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned articles/images into Postgres")
    parser.add_argument(
        "--cleaned-json",
        default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()),
        help="Path to cleaned articles JSON",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Postgres connection URL. If not provided, uses DATABASE_URL from environment.",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _to_bytes_from_hex(hex_str: Optional[str]) -> Optional[bytes]:
    if not hex_str:
        return None
    try:
        return bytes.fromhex(hex_str)
    except Exception:
        return None


def connect(db_url: str, retries: int = 10, delay: float = 2.0):
    last_err = None
    for _ in range(retries):
        try:
            conn = psycopg.connect(db_url, autocommit=False)
            return conn
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err


def upsert_articles(conn, articles: List[Dict[str, Any]], batch_size: int = 200) -> Dict[str, str]:
    """Upsert articles in a single transaction using a bulk VALUES insert.

    Returns a mapping of url -> article_id (as string).
    """
    sql = """
        INSERT INTO articles (
            url, slug, title, author, published_at_utc, body_markdown, body_html,
            content_hash, image_count, processing_status
        ) VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            published_at_utc = EXCLUDED.published_at_utc,
            body_markdown = EXCLUDED.body_markdown,
            body_html = EXCLUDED.body_html,
            content_hash = EXCLUDED.content_hash,
            image_count = EXCLUDED.image_count,
            processing_status = EXCLUDED.processing_status,
            updated_at = now()
        RETURNING article_id, url;
    """

    values: List[Tuple[Any, ...]] = []
    for art in articles:
        values.append(
            (
                art.get("url"),
                None,  # slug
                art.get("title"),
                art.get("author"),
                art.get("published_at_utc"),
                art.get("body_text"),
                None,  # body_html
                _to_bytes_from_hex(art.get("content_hash")),
                len(art.get("images") or []),
                "completed",
            )
        )

    article_id_by_url: Dict[str, str] = {}
    if not values:
        return article_id_by_url

    with conn.cursor() as cur:
        try:
            execute_values(
                cur,
                sql,
                values,
                template="(" + ",".join(["%s"] * 10) + ")",
                page_size=batch_size,
            )
            returned = cur.fetchall()
            for article_id, url in returned:
                article_id_by_url[url] = str(article_id)
        except Exception as e:
            logger.exception("Failed to upsert articles")
            raise
    return article_id_by_url


def insert_images(conn, articles: List[Dict[str, Any]], article_id_by_url: Dict[str, str], batch_size: int = 500) -> None:
    """Delete existing images for affected articles and bulk insert new ones in a single transaction."""
    # Build list of affected article IDs
    affected_article_ids = [article_id_by_url.get(a.get("url")) for a in articles]
    affected_article_ids = [aid for aid in affected_article_ids if aid]

    insert_sql = """
        INSERT INTO article_images (
            article_id, image_url, image_type, image_hash, ocr_text,
            extracted_data, vision_analysis, download_path, file_size_bytes,
            width, height, processing_status, confidence_score
        ) VALUES %s
    """

    values: List[Tuple[Any, ...]] = []
    for art in articles:
        aid = article_id_by_url.get(art.get("url"))
        if not aid:
            continue
        for img in art.get("images") or []:
            values.append(
                (
                    aid,
                    img.get("image_url"),
                    img.get("image_type") or "unknown",
                    _to_bytes_from_hex(img.get("image_hash")),
                    img.get("ocr_text"),
                    Jsonb(img.get("extracted_data")) if img.get("extracted_data") is not None else None,
                    Jsonb(img.get("vision_analysis")) if img.get("vision_analysis") is not None else None,
                    img.get("download_path_cleaned") or img.get("download_path"),
                    img.get("file_size_bytes"),
                    img.get("width"),
                    img.get("height"),
                    img.get("processing_status") or "completed",
                    img.get("confidence_score") or 0.0,
                )
            )

    with conn.cursor() as cur:
        try:
            if affected_article_ids:
                # Delete existing images for these articles to ensure idempotency
                cur.execute(
                    "DELETE FROM article_images WHERE article_id = ANY(%s)",
                    (affected_article_ids,),
                )

            if values:
                execute_values(
                    cur,
                    insert_sql,
                    values,
                    template="(" + ",".join(["%s"] * 13) + ")",
                    page_size=batch_size,
                )
        except Exception:
            logger.exception("Failed to insert images")
            raise


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cleaned_path = Path(args.cleaned_json)
    if not cleaned_path.exists():
        print(f"Cleaned JSON not found: {cleaned_path}")
        sys.exit(2)

    with cleaned_path.open("r", encoding="utf-8") as f:
        articles = json.load(f)
    if not isinstance(articles, list):
        print("Cleaned JSON must be a list of articles")
        sys.exit(2)

    if args.dry_run:
        print(f"DRY RUN: would load {len(articles)} articles")
        sys.exit(0)

    # Resolve database URL: CLI argument or environment
    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        print("Error: database URL must be provided via --db-url or DATABASE_URL environment variable", file=sys.stderr)
        sys.exit(2)

    conn = connect(db_url)
    try:
        try:
            mapping = upsert_articles(conn, articles, batch_size=args.batch_size)
            insert_images(conn, articles, mapping, batch_size=args.batch_size * 2)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.exception("DB load failed")
            raise

        print(json.dumps({
            "loaded_articles": len(mapping),
            "loaded_images": sum(len(a.get("images") or []) for a in articles if a.get("url") in mapping),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }))
    finally:
        conn.close()


if __name__ == "__main__":
    main()



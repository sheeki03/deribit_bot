import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned articles/images into Postgres")
    parser.add_argument(
        "--cleaned-json",
        default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()),
        help="Path to cleaned articles JSON",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv(
            "DATABASE_URL",
            "postgresql://deribit_user:deribit_pass@localhost:5432/deribit_flows",
        ),
        help="Postgres connection URL",
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


def upsert_articles(conn, articles: List[Dict[str, Any]], batch_size: int = 200):
    sql = """
    INSERT INTO articles (
        url, slug, title, author, published_at_utc, body_markdown, body_html,
        content_hash, image_count, processing_status
    ) VALUES (
        %(url)s, %(slug)s, %(title)s, %(author)s, %(published_at_utc)s, %(body_markdown)s, %(body_html)s,
        %(content_hash)s, %(image_count)s, %(processing_status)s
    )
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

    rows = []
    for art in articles:
        rows.append(
            {
                "url": art.get("url"),
                "slug": None,
                "title": art.get("title"),
                "author": art.get("author"),
                "published_at_utc": art.get("published_at_utc"),
                "body_markdown": art.get("body_text"),
                "body_html": None,
                "content_hash": _to_bytes_from_hex(art.get("content_hash")),
                "image_count": len(art.get("images") or []),
                "processing_status": "completed",
            }
        )

    article_id_by_url: Dict[str, str] = {}
    with conn.cursor() as cur:
        for row in rows:
            cur.execute("BEGIN")
            try:
                cur.execute(sql, row)
                rec = cur.fetchone()
                if rec:
                    article_id, url = rec
                    article_id_by_url[url] = str(article_id)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    return article_id_by_url


def insert_images(conn, articles: List[Dict[str, Any]], article_id_by_url: Dict[str, str], batch_size: int = 500):
    # We do simple insert; for idempotency, delete existing images for the article first then insert fresh
    # Delete existing images per article to avoid duplicates
    with conn.cursor() as cur:
        cur.execute("BEGIN")
        try:
            for a in articles:
                aid = article_id_by_url.get(a.get("url"))
                if aid:
                    cur.execute("DELETE FROM article_images WHERE article_id = %s", (aid,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    sql = """
    INSERT INTO article_images (
        article_id, image_url, image_type, image_hash, ocr_text,
        extracted_data, vision_analysis, download_path, file_size_bytes,
        width, height, processing_status, confidence_score
    ) VALUES (
        %(article_id)s, %(image_url)s, %(image_type)s, %(image_hash)s, %(ocr_text)s,
        %(extracted_data)s, %(vision_analysis)s, %(download_path)s, %(file_size_bytes)s,
        %(width)s, %(height)s, %(processing_status)s, %(confidence_score)s
    )
    """

    rows = []
    for art in articles:
        aid = article_id_by_url.get(art.get("url"))
        if not aid:
            continue
        for img in art.get("images") or []:
            rows.append(
                {
                    "article_id": aid,
                    "image_url": img.get("image_url"),
                    "image_type": img.get("image_type") or "unknown",
                    "image_hash": _to_bytes_from_hex(img.get("image_hash")),
                    "ocr_text": img.get("ocr_text"),
                    "extracted_data": Jsonb(img.get("extracted_data")) if img.get("extracted_data") is not None else None,
                    "vision_analysis": Jsonb(img.get("vision_analysis")) if img.get("vision_analysis") is not None else None,
                    "download_path": img.get("download_path_cleaned") or img.get("download_path"),
                    "file_size_bytes": img.get("file_size_bytes"),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "processing_status": img.get("processing_status") or "completed",
                    "confidence_score": img.get("confidence_score") or 0.0,
                }
            )

    with conn.cursor() as cur:
        for r in rows:
            cur.execute("BEGIN")
            try:
                cur.execute(sql, r)
                conn.commit()
            except Exception:
                conn.rollback()
                raise


def main():
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

    conn = connect(args.db_url)
    try:
        mapping = upsert_articles(conn, articles, batch_size=args.batch_size)
        insert_images(conn, articles, mapping, batch_size=args.batch_size * 2)
        conn.commit()
        print(json.dumps({
            "loaded_articles": len(mapping),
            "loaded_images": sum(len(a.get("images") or []) for a in articles if a.get("url") in mapping),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }))
    finally:
        conn.close()


if __name__ == "__main__":
    main()



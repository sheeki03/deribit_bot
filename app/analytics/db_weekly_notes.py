from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple

import psycopg
from psycopg.rows import dict_row
from psycopg.extras import execute_values


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS weekly_note_analyses (
    id BIGSERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    published_at_utc TIMESTAMPTZ,
    sentiment TEXT,
    confidence DOUBLE PRECISION,
    returns JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def connect(db_url: str | None = None):
    dsn = db_url or os.getenv("DATABASE_URL")
    if not dsn:
        raise ValueError("DATABASE_URL not provided")
    
    # Parse and validate DSN
    from urllib.parse import urlparse
    try:
        parsed = urlparse(dsn)
    except Exception as e:
        raise ValueError(f"Invalid DSN format: {e}")
    
    # Verify scheme
    if parsed.scheme not in ("postgres", "postgresql"):
        raise ValueError(f"Invalid scheme '{parsed.scheme}': must be 'postgres' or 'postgresql'")
    
    # Ensure hostname is present
    if not parsed.hostname:
        raise ValueError("DSN must include a hostname")
    
    # Validate port if present
    if parsed.port is not None:
        if not isinstance(parsed.port, int) or parsed.port <= 0 or parsed.port > 65535:
            raise ValueError(f"Invalid port {parsed.port}: must be a positive integer <= 65535")
    
    # Validate netloc format (basic check)
    if not parsed.netloc:
        raise ValueError("DSN netloc missing or malformed")
    
    return psycopg.connect(dsn, autocommit=False)


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()


def _row_to_db_tuple(row: Dict[str, Any]) -> Tuple[Any, ...]:
    # Collect returns keys: any *_ret_* fields into a flat dict
    returns: Dict[str, Any] = {}
    for k, v in row.items():
        if "_ret_" in k:
            returns[k] = v
    pub = row.get("published_at_utc")
    # Normalize timestamp string to ISO
    if isinstance(pub, datetime):
        pub_iso = pub.isoformat()
    else:
        pub_iso = pub
    return (
        row.get("url"),
        row.get("title"),
        pub_iso,
        row.get("sentiment"),
        float(row.get("confidence") or 0.0),
        returns or None,
    )


def upsert_weekly_note_analyses(conn, rows: Iterable[Dict[str, Any]]) -> int:
    sql = """
        INSERT INTO weekly_note_analyses (
            url, title, published_at_utc, sentiment, confidence, returns
        ) VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            title = EXCLUDED.title,
            published_at_utc = EXCLUDED.published_at_utc,
            sentiment = EXCLUDED.sentiment,
            confidence = EXCLUDED.confidence,
            returns = EXCLUDED.returns,
            updated_at = now()
    """
    tuples = [_row_to_db_tuple(r) for r in rows]
    if not tuples:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            sql,
            tuples,
            template="(" + ",".join(["%s"] * 6) + ")",
            page_size=200,
        )
    conn.commit()
    return len(tuples)

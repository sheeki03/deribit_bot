#!/usr/bin/env python3
"""
Enhanced Text Extraction CLI

Extract structured options data from cleaned articles and populate the extractions table.
This implements the Enhanced PRD Phase 2: Intelligence - Text Analysis component.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.ml.feature_extractors import OptionsFeatureExtractor
from app.core.logging import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured options data from articles")
    parser.add_argument(
        "--cleaned-json",
        default=str(Path("scraped_data/cleaned/articles_cleaned.json").resolve()),
        help="Path to cleaned articles JSON"
    )
    parser.add_argument(
        "--db-url", 
        default=None,
        help="Postgres connection URL. Uses DATABASE_URL env var if not provided."
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, help="Limit number of articles to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


class ArticleExtractor:
    """Extract structured options data from articles."""
    
    def __init__(self, db_url: Optional[str] = None):
        self.extractor = OptionsFeatureExtractor()
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        # Allow dry run without database
        if not self.db_url:
            import warnings
            warnings.warn("No database URL provided. Only dry-run mode available.")
    
    def load_cleaned_articles(self, json_path: str, limit: Optional[int] = None) -> List[Dict]:
        """Load cleaned articles from JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            if limit:
                articles = articles[:limit]
            
            logger.info(f"Loaded {len(articles)} articles from {json_path}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to load articles from {json_path}: {e}")
            raise
    
    def extract_article_data(self, article: Dict) -> Optional[Dict]:
        """Extract structured data from a single article."""
        try:
            # Get text content (prefer body_text from cleaned data)
            text_content = article.get('body_text') or article.get('body_markdown', '')
            title = article.get('title', '')
            
            if not text_content:
                logger.warning(f"No text content for article: {article.get('url', 'unknown')}")
                return None
            
            # Extract structured options data
            extraction = self.extractor.extract_structured_options_data(text_content, title)
            
            # Add article metadata
            extraction['article_url'] = article.get('url')
            extraction['article_title'] = title
            extraction['published_at'] = article.get('published_at_utc')
            extraction['content_hash'] = article.get('content_hash')
            
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to extract data from article {article.get('url', 'unknown')}: {e}")
            return None
    
    def store_extractions(self, extractions: List[Dict], dry_run: bool = False):
        """Store extractions in database."""
        if dry_run or not self.db_url:
            logger.info(f"DRY RUN: Would store {len(extractions)} extractions")
            # Print sample extractions
            for i, ext in enumerate(extractions[:3]):
                logger.info(f"Sample extraction {i+1}:")
                logger.info(f"  URL: {ext.get('article_url')}")
                logger.info(f"  Strikes: {len(ext.get('strikes', []))}")
                logger.info(f"  Notionals: {len(ext.get('notionals', []))}")
                logger.info(f"  Flow Direction: {ext.get('flow_direction')}")
                logger.info(f"  Confidence: {ext.get('confidence', 0):.2f}")
            return
        
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Create extractions table if not exists
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS extractions (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            article_url TEXT NOT NULL,
                            article_title TEXT,
                            content_hash BYTEA,
                            strikes JSONB DEFAULT '[]'::jsonb,
                            notionals JSONB DEFAULT '[]'::jsonb,
                            expiries JSONB DEFAULT '[]'::jsonb,
                            greeks JSONB DEFAULT '{}'::jsonb,
                            option_types JSONB DEFAULT '[]'::jsonb,
                            flow_direction TEXT DEFAULT 'neutral',
                            sentiment_indicators JSONB DEFAULT '[]'::jsonb,
                            flow_concentration JSONB DEFAULT '[]'::jsonb,
                            confidence REAL DEFAULT 0.0,
                            extracted_at TIMESTAMPTZ DEFAULT now(),
                            created_at TIMESTAMPTZ DEFAULT now(),
                            UNIQUE(article_url, content_hash)
                        )
                    """)
                    
                    # Prepare insertion data
                    insert_data = []
                    for ext in extractions:
                        # Convert content_hash if present
                        content_hash = None
                        if ext.get('content_hash'):
                            try:
                                content_hash = bytes.fromhex(ext['content_hash'])
                            except Exception:
                                pass
                        
                        insert_data.append((
                            ext['article_url'],
                            ext.get('article_title'),
                            content_hash,
                            Jsonb(ext.get('strikes', [])),
                            Jsonb(ext.get('notionals', [])),
                            Jsonb(ext.get('expiries', [])),
                            Jsonb(ext.get('greeks', {})),
                            Jsonb(ext.get('option_types', [])),
                            ext.get('flow_direction', 'neutral'),
                            Jsonb(ext.get('sentiment_indicators', [])),
                            Jsonb(ext.get('flow_concentration', [])),
                            ext.get('confidence', 0.0)
                        ))
                    
                    # Bulk insert with conflict resolution
                    query = """
                        INSERT INTO extractions (
                            article_url, article_title, content_hash, strikes, notionals,
                            expiries, greeks, option_types, flow_direction,
                            sentiment_indicators, flow_concentration, confidence
                        ) VALUES %s
                        ON CONFLICT (article_url, content_hash) DO UPDATE SET
                            strikes = EXCLUDED.strikes,
                            notionals = EXCLUDED.notionals,
                            expiries = EXCLUDED.expiries,
                            greeks = EXCLUDED.greeks,
                            option_types = EXCLUDED.option_types,
                            flow_direction = EXCLUDED.flow_direction,
                            sentiment_indicators = EXCLUDED.sentiment_indicators,
                            flow_concentration = EXCLUDED.flow_concentration,
                            confidence = EXCLUDED.confidence,
                            extracted_at = now()
                    """
                    
                    from psycopg.extras import execute_values
                    execute_values(cur, query, insert_data, page_size=50)
                    
                    conn.commit()
                    logger.info(f"Successfully stored {len(extractions)} extractions")
                    
        except Exception as e:
            logger.error(f"Failed to store extractions: {e}")
            raise
    
    def run_extraction(self, json_path: str, batch_size: int = 50, 
                      limit: Optional[int] = None, dry_run: bool = False):
        """Run the full extraction process."""
        logger.info("Starting enhanced text extraction process")
        logger.info(f"Target: {limit or 'all'} articles from {json_path}")
        
        # Load articles
        articles = self.load_cleaned_articles(json_path, limit)
        
        if not articles:
            logger.warning("No articles to process")
            return
        
        # Process articles in batches
        all_extractions = []
        successful = 0
        failed = 0
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_extractions = []
            
            logger.info(f"Processing batch {i//batch_size + 1}: articles {i+1}-{min(i+batch_size, len(articles))}")
            
            for article in batch:
                extraction = self.extract_article_data(article)
                if extraction:
                    batch_extractions.append(extraction)
                    successful += 1
                else:
                    failed += 1
            
            if batch_extractions:
                all_extractions.extend(batch_extractions)
        
        logger.info(f"Extraction complete: {successful} successful, {failed} failed")
        
        # Store results
        if all_extractions:
            self.store_extractions(all_extractions, dry_run)
            
            # Print summary statistics
            self.print_extraction_stats(all_extractions)
        
        return all_extractions
    
    def print_extraction_stats(self, extractions: List[Dict]):
        """Print summary statistics about extractions."""
        if not extractions:
            return
        
        total = len(extractions)
        
        # Calculate statistics
        with_strikes = sum(1 for e in extractions if e.get('strikes'))
        with_notionals = sum(1 for e in extractions if e.get('notionals'))
        with_greeks = sum(1 for e in extractions if e.get('greeks'))
        
        # Flow direction distribution
        flow_counts = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        confidence_scores = []
        
        for ext in extractions:
            flow_dir = ext.get('flow_direction', 'neutral')
            flow_counts[flow_dir] = flow_counts.get(flow_dir, 0) + 1
            confidence_scores.append(ext.get('confidence', 0.0))
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        logger.info("=== EXTRACTION STATISTICS ===")
        logger.info(f"Total articles processed: {total}")
        logger.info(f"Articles with strikes: {with_strikes} ({with_strikes/total*100:.1f}%)")
        logger.info(f"Articles with notionals: {with_notionals} ({with_notionals/total*100:.1f}%)")
        logger.info(f"Articles with Greeks: {with_greeks} ({with_greeks/total*100:.1f}%)")
        logger.info(f"Average confidence: {avg_confidence:.3f}")
        logger.info("Flow Direction Distribution:")
        for direction, count in flow_counts.items():
            logger.info(f"  {direction.capitalize()}: {count} ({count/total*100:.1f}%)")


def main():
    """Main extraction CLI."""
    args = parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        extractor = ArticleExtractor(args.db_url)
        extractor.run_extraction(
            json_path=args.cleaned_json,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run
        )
        
        logger.info("Text extraction completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


if __name__ == "__main__":
    import os
    success = main()
    sys.exit(0 if success else 1)
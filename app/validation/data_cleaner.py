import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.logging import logger
from app.validation.data_validator import DataValidator


def read_json_array(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON file expected to contain a list of objects. Return [] on error."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logger.warning(f"File is not a JSON array: {path}")
            return []
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return []


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_html(text: str) -> str:
    if not text:
        return ""
    # Basic tag removal; avoids external dependencies
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(no_tags)


def strip_noise_sections(text: str) -> str:
    """Remove common boilerplate sections like Disclaimer, AUTHOR(S), RECENT ARTICLES, and social links."""
    if not text:
        return ""
    # Normalize newlines for section-based removal
    lines = text.splitlines()
    cleaned: List[str] = []
    skip = False
    noise_headers = [
        re.compile(r"^\s*disclaimer\s*$", re.IGNORECASE),
        re.compile(r"^\s*author\(s\)\s*$", re.IGNORECASE),
        re.compile(r"^\s*recent articles\s*$", re.IGNORECASE),
    ]
    for line in lines:
        if any(pat.search(line) for pat in noise_headers):
            skip = True
            continue
        # Stop skipping when we hit an empty line separating sections or a strong divider
        if skip and (line.strip() == "" or re.search(r"^-{2,}|={2,}|\*{2,}", line)):
            skip = False
            continue
        if not skip:
            cleaned.append(line)
    text2 = "\n".join(cleaned)
    # Remove explicit social link phrases
    text2 = re.sub(r"\bview\s+x\s+thread\b.*", "", text2, flags=re.IGNORECASE)
    text2 = re.sub(r"\bfollow\s+us\b.*", "", text2, flags=re.IGNORECASE)
    return normalize_whitespace(text2)


def compute_content_hash(article: Dict[str, Any]) -> str:
    content = article.get("body_markdown") or article.get("body_html") or ""
    content = strip_html(content)
    content = strip_noise_sections(content)
    content = normalize_whitespace(content.lower())
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def normalize_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize fields, clean noise, and add derived fields. Non-destructive."""
    url = article.get("url", "").strip()
    title = normalize_whitespace(article.get("title", ""))
    author = normalize_whitespace(article.get("author", ""))

    # Dates: prefer ISO in UTC if plausible
    published_raw = article.get("published_at") or article.get("published_at_utc")
    published_at_utc: Optional[str] = None
    if isinstance(published_raw, str) and published_raw.strip():
        s = published_raw.strip().replace("Z", "+00:00")
        dt: Optional[datetime] = None
        try:
            # First try exact "YYYY-mm-dd HH:MM:SS"
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                # Fallback to fromisoformat
                dt = datetime.fromisoformat(s)
            except ValueError:
                dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            published_at_utc = dt.astimezone(timezone.utc).isoformat()

    # Clean bodies
    body_html = article.get("body_html") or ""
    body_markdown = article.get("body_markdown") or ""

    if body_markdown:
        body_clean = strip_noise_sections(body_markdown)
    elif body_html:
        body_clean = strip_noise_sections(strip_html(body_html))
    else:
        body_clean = ""

    summary_html = article.get("summary") or ""
    summary_text = strip_html(summary_html)
    summary_text = strip_noise_sections(summary_text)

    images = article.get("images") or []
    # Deduplicate images in-article by image_hash or URL
    seen_img: Set[str] = set()
    dedup_images: List[Dict[str, Any]] = []
    for img in images:
        key = img.get("image_hash") or img.get("image_url") or img.get("download_path")
        if not key:
            continue
        if key in seen_img:
            continue
        seen_img.add(key)
        dedup_images.append(img)

    normalized = {
        "url": url,
        "title": title,
        "author": author,
        "published_at_utc": published_at_utc,
        "summary_text": summary_text,
        "body_text": body_clean,
        "source": article.get("source"),
        "discovery_method": article.get("discovery_method"),
        "images": dedup_images,
        "scraping_method": article.get("scraping_method"),
        "content_hash": article.get("content_hash") or compute_content_hash(article),
        "original_fields": {
            # Keep a handful of original fields for traceability
            k: article.get(k)
            for k in ["body_markdown", "body_html", "summary", "published_at"]
        },
    }
    return normalized


def merge_and_deduplicate(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge by URL first, then drop content duplicates by content_hash."""
    by_url: Dict[str, Dict[str, Any]] = {}
    for art in articles:
        norm = normalize_article(art)
        url_key = (norm.get("url") or "").strip().lower()
        if not url_key:
            # Use content hash as fallback key
            url_key = f"hash:{norm.get('content_hash')}"
        if url_key in by_url:
            # Prefer the record with longer body_text and more images
            old = by_url[url_key]
            if len(norm.get("body_text", "")) > len(old.get("body_text", "")) or (
                len(norm.get("images", [])) > len(old.get("images", []))
            ):
                by_url[url_key] = norm
        else:
            by_url[url_key] = norm

    # Drop duplicates by content_hash across different URLs (rare, but progress snapshots can cause this)
    seen_hashes: Set[str] = set()
    unique: List[Dict[str, Any]] = []
    for norm in by_url.values():
        ch = norm.get("content_hash")
        if not ch:
            unique.append(norm)
            continue
        if ch in seen_hashes:
            continue
        seen_hashes.add(ch)
        unique.append(norm)
    return unique


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dedupe_images(images_root: Path, output_root: Path) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Deduplicate image files by content hash, copying unique files to output_root preserving subfolders.

    Returns (old_to_new_path_map, stats)
    """
    stats = {"scanned": 0, "unique": 0, "duplicates": 0}
    old_to_new: Dict[str, str] = {}
    seen_hash: Dict[str, str] = {}  # sha256 -> new_path

    if not images_root.exists():
        return old_to_new, stats

    for dirpath, _dirnames, filenames in os.walk(images_root):
        for fn in filenames:
            src = Path(dirpath) / fn
            # Skip non-image extensions
            if not re.search(r"\.(jpe?g|png|gif)$", src.name, re.IGNORECASE):
                continue
            stats["scanned"] += 1
            try:
                digest = sha256_file(src)
            except Exception as e:
                logger.warning(f"Failed hashing {src}: {e}")
                continue
            if digest in seen_hash:
                stats["duplicates"] += 1
                old_to_new[str(src)] = seen_hash[digest]
                continue

            # Create deterministic subfolder by first 2 bytes of hash for spread
            subdir = digest[:2]
            dst_dir = output_root / subdir
            dst_dir.mkdir(parents=True, exist_ok=True)
            # Keep original extension
            dst = dst_dir / f"{digest}{src.suffix.lower()}"
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
                seen_hash[digest] = str(dst)
                old_to_new[str(src)] = str(dst)
                stats["unique"] += 1
            except Exception as e:
                logger.error(f"Failed copying {src} -> {dst}: {e}")

    return old_to_new, stats


def remap_article_image_paths(articles: List[Dict[str, Any]], mapping: Dict[str, str]) -> None:
    for art in articles:
        for img in art.get("images", []) or []:
            old = img.get("download_path")
            if old and old in mapping:
                img["download_path_cleaned"] = mapping[old]


def run_cleaning(scraped_dir: Path, images_dir: Path) -> None:
    # 1) Gather JSON arrays from scraped_dir
    json_files = sorted([p for p in scraped_dir.glob("*.json")])
    all_articles_raw: List[Dict[str, Any]] = []
    for jf in json_files:
        # Many are progress snapshots; merge them all
        all_articles_raw.extend(read_json_array(jf))

    logger.info(f"Loaded {len(all_articles_raw)} raw articles from {len(json_files)} files")

    # 2) Merge and dedupe
    unique_articles = merge_and_deduplicate(all_articles_raw)
    logger.info(f"After deduplication: {len(unique_articles)} articles")

    # 3) Validate to compute quality metadata (non-dropping)
    validator = DataValidator()
    quality_breakdown: List[Dict[str, Any]] = []
    valid_count = 0
    for art in unique_articles:
        result = validator.validate_article({
            "url": art.get("url"),
            "title": art.get("title"),
            "author": art.get("author"),
            "body_markdown": art.get("body_text"),
            "published_at_utc": art.get("published_at_utc"),
            "images": art.get("images"),
        })
        if result.is_valid:
            valid_count += 1
        art["validation"] = {
            "is_valid": result.is_valid,
            "quality_score": result.quality_score,
            "warnings": result.warnings,
            "metadata": result.metadata,
        }
        quality_breakdown.append(art["validation"])

    logger.info(f"Validation complete: {valid_count}/{len(unique_articles)} valid by thresholds")

    # 4) Image filesystem dedupe and remap
    cleaned_images_root = images_dir.parent / "images_cleaned"
    mapping, img_stats = dedupe_images(images_dir, cleaned_images_root)
    remap_article_image_paths(unique_articles, mapping)
    logger.info(
        f"Images scanned={img_stats.get('scanned',0)}, unique={img_stats.get('unique',0)}, duplicates={img_stats.get('duplicates',0)}"
    )

    # 5) Save outputs
    cleaned_dir = scraped_dir / "cleaned"
    save_json(cleaned_dir / "articles_cleaned.json", unique_articles)
    save_json(cleaned_dir / "image_path_mapping.json", mapping)
    save_json(cleaned_dir / "cleaning_stats.json", {
        "raw_articles": len(all_articles_raw),
        "unique_articles": len(unique_articles),
        "valid_articles": valid_count,
        "images": img_stats,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    logger.info(f"Saved cleaned artifacts under: {cleaned_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate and clean scraped articles and images")
    parser.add_argument("--scraped-dir", required=True, help="Absolute path to scraped_data directory")
    parser.add_argument("--images-dir", required=True, help="Absolute path to data/images directory")
    return parser.parse_args()


def main():
    args = parse_args()
    scraped_dir = Path(args.scraped_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve()
    if not scraped_dir.exists():
        raise SystemExit(f"scraped-dir not found: {scraped_dir}")
    if not images_dir.exists():
        logger.warning(f"images-dir not found: {images_dir} (skipping image dedupe)")
    run_cleaning(scraped_dir, images_dir)


if __name__ == "__main__":
    main()



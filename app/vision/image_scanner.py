from __future__ import annotations

import argparse
import json
import logging
import os
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
from datetime import datetime
from contextlib import contextmanager

from app.vision.openrouter_client import OpenRouterVisionClient
from app.vision.ocr_helper import ocr_image, get_optimal_ocr_config_for_charts
from app.vision.performance_monitor import PerformanceMonitor, AdaptiveRateLimiter

logger = logging.getLogger(__name__)

@dataclass
class ScanConfig:
    mapping_json: Optional[Path]
    images_dir: Optional[Path]
    model: Optional[str]
    output: Path
    resume: bool
    limit: Optional[int]
    do_ocr: bool
    # Resume controls
    resume_mode: str = "any"  # 'any' or 'parsed_only'
    start_after: Optional[str] = None
    progress: Optional[Path] = None
    # Performance controls
    batch_size: int = 10  # Process in smaller batches
    progress_interval: int = 3  # Write progress every N items
    memory_cleanup_interval: int = 20  # Run gc every N items


def load_image_list(mapping_json: Optional[Path], images_dir: Optional[Path]) -> List[Tuple[Optional[str], str]]:
    pairs: List[Tuple[Optional[str], str]] = []
    if mapping_json and mapping_json.exists():
        try:
            data = json.loads(mapping_json.read_text())
            # Supported shapes:
            # 1) { url_or_id: ["path1", "path2", ...], ... }
            # 2) [ {"url": ..., "images": [..]}, ... ] or [ {"id":..., "path": ...}, ...]
            if isinstance(data, dict):
                for key, val in data.items():
                    if isinstance(val, list):
                        for p in val:
                            pairs.append((key, p))
                    elif isinstance(val, str):
                        pairs.append((key, val))
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    url = item.get("url") or item.get("article_url") or item.get("id")
                    if "images" in item and isinstance(item["images"], list):
                        for p in item["images"]:
                            pairs.append((url, p))
                    elif "path" in item and isinstance(item["path"], str):
                        pairs.append((url, item["path"]))
        except Exception:
            logger.exception("Failed to load mapping JSON for image list", extra={
                'mapping_json': str(mapping_json) if mapping_json else None,
            })
    if not pairs and images_dir and images_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in images_dir.rglob(ext):
                pairs.append((None, str(p)))
    # Normalize to strings and filter existing only
    norm: List[Tuple[Optional[str], str]] = []
    for url, p in pairs:
        sp = str(p)
        if os.path.exists(sp):
            norm.append((url, sp))
    return norm


@contextmanager
def batch_processor(cfg: ScanConfig):
    """Context manager for efficient batch processing with memory management and monitoring."""
    client = OpenRouterVisionClient(model=cfg.model) if cfg.model else OpenRouterVisionClient()
    
    # Initialize performance monitoring
    monitor = PerformanceMonitor(f"scan_{int(datetime.now().timestamp())}")
    rate_limiter = AdaptiveRateLimiter(initial_rate=2.0, min_rate=0.5, max_rate=5.0)
    
    # Load processed paths for resume functionality
    processed_paths: set[str] = set()
    if cfg.resume and cfg.output.exists():
        try:
            with monitor.track_operation("load_existing_results"):
                with open(cfg.output, 'r') as f:
                    prior = json.load(f)
                for rec in prior.get("images", []) or []:
                    p = rec.get("path")
                    if isinstance(p, str):
                        if cfg.resume_mode == "parsed_only":
                            if rec.get("gpt5") is not None:
                                processed_paths.add(p)
                        else:  # any
                            if rec.get("gpt5") is not None or rec.get("gpt5_raw") is not None:
                                processed_paths.add(p)
        except Exception:
            logger.exception("Failed to load prior image analysis; starting fresh", extra={
                'output_path': str(cfg.output),
            })
    
    try:
        yield client, processed_paths, monitor, rate_limiter
    finally:
        # Save performance metrics
        metrics_path = cfg.output.with_suffix('.metrics.json')
        monitor.save_metrics(metrics_path)
        monitor.log_summary()
        
        # Cleanup
        del client
        gc.collect()

def analyze_images(pairs: List[Tuple[Optional[str], str]], cfg: ScanConfig) -> Dict[str, Any]:
    """Analyze images with improved memory management and progress tracking."""
    with batch_processor(cfg) as (client, processed_paths, monitor, rate_limiter):
        # Load existing results efficiently
        existing_results = []
        if cfg.resume and cfg.output.exists():
            try:
                with open(cfg.output, 'r') as f:
                    prior = json.load(f)
                existing_results = prior.get("images", [])
            except Exception:
                logger.warning("Could not load existing results; starting fresh")
        
        skipped = 0
        started = cfg.start_after is None
        new_records = []
        progress_path: Optional[Path] = cfg.progress

        def write_progress(final: bool = False) -> None:
            """Write progress with enhanced error handling and atomic updates."""
            if not progress_path:
                return
            try:
                payload = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "output": str(cfg.output),
                    "resume_mode": cfg.resume_mode,
                    "start_after": cfg.start_after,
                    "model": cfg.model,
                    "processed_in_run": len(new_records),
                    "skipped_in_run": skipped,
                    "total_images_in_output": len(existing_results) + len(new_records),
                    "last_processed_path": new_records[-1]["path"] if new_records else None,
                    "final": final,
                    "memory_usage_mb": _get_memory_usage_mb_safe()
                }
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic write
                temp_path = progress_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(payload, f, indent=2)
                temp_path.replace(progress_path)
                
            except Exception:
                logger.exception("Failed writing progress file", extra={'progress_path': str(progress_path)})
        
        def write_incremental_output() -> None:
            """Write output incrementally to handle large datasets."""
            if not new_records:
                return
                
            try:
                all_results = existing_results + new_records
                output_data = {
                    "images": all_results,
                    "count": len(new_records),
                    "skipped": skipped,
                    "added": len(new_records),
                    "total": len(all_results)
                }
                
                cfg.output.parent.mkdir(parents=True, exist_ok=True)
                
                # Atomic write
                temp_path = cfg.output.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                temp_path.replace(cfg.output)
                
            except Exception:
                logger.exception("Failed writing incremental output", extra={'output_path': str(cfg.output)})
        # Process images in batches for better memory management
        items_processed = 0
        batch_start_idx = 0
        
        for i, (url, path) in enumerate(pairs):
            # Support manual continuation after a given path
            if not started:
                if path == cfg.start_after:
                    started = True
                continue
                
            if cfg.limit is not None and len(new_records) >= cfg.limit:
                break
                
            rec: Dict[str, Any] = {"article": url, "path": path}
            
            if path in processed_paths:
                skipped += 1
                continue
                
            # Rate limiting before processing
            if not rate_limiter.acquire(timeout=30.0):
                logger.warning(f"Rate limit timeout for {path}")
                rec["error"] = "rate_limit_timeout"
            else:
                # Process the image with monitoring
                try:
                    with monitor.track_operation("vision_analysis", {"path": path, "url": url}):
                        vision_res = client.analyze_image(path)
                        raw = vision_res.get("raw", "")
                        rec["gpt5_raw"] = raw
                        parsed = vision_res.get("parsed")
                        rec["gpt5"] = parsed if isinstance(parsed, dict) else None
                        
                        # Include additional metadata from enhanced client
                        for key in ["model_used", "request_duration", "parsing_success"]:
                            if key in vision_res:
                                rec[key] = vision_res[key]
                    
                    rate_limiter.record_result(True)
                        
                except Exception as e:
                    logger.exception("Vision analysis failed", extra={'path': path, 'article': url})
                    rec["error"] = f"vision_error: {e}"
                    rate_limiter.record_result(False, str(e))
                    
                if cfg.do_ocr:
                    try:
                        with monitor.track_operation("ocr_analysis", {"path": path}):
                            # Use optimized OCR config for charts
                            ocr_config = get_optimal_ocr_config_for_charts()
                            rec["ocr_text"] = ocr_image(path, ocr_config)
                    except Exception:
                        logger.exception("OCR failed", extra={'path': path})
                        rec["ocr_text"] = None
                    
            new_records.append(rec)
            items_processed += 1
            
            # Progress tracking and memory management
            if items_processed % cfg.progress_interval == 0:
                write_progress(final=False)
                
            if items_processed % cfg.memory_cleanup_interval == 0:
                gc.collect()
                logger.debug(f"Memory cleanup at item {items_processed}, memory: {_get_memory_usage_mb():.1f}MB")
                
            # Batch processing for large datasets
            if len(new_records) % cfg.batch_size == 0:
                write_incremental_output()
                logger.info(f"Processed batch {len(new_records)//cfg.batch_size}: {len(new_records)} total items")
        logger.info(
            "Scan summary: processed=%s skipped=%s resume_mode=%s start_after=%s",
            len(new_records), skipped, cfg.resume_mode, cfg.start_after,
        )
        
        # Final writes
        write_incremental_output()
        write_progress(final=True)
        
        # Return results
        return {
            "images": existing_results + new_records,
            "count": len(new_records),
            "skipped": skipped, 
            "added": len(new_records),
            "total": len(existing_results) + len(new_records)
        }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan images with OpenRouter GPT-5 vision and optional OCR")
    p.add_argument("--mapping-json", default=str(Path("scraped_data/cleaned/image_path_mapping.json")))
    p.add_argument("--images-dir", default="")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--model", default=os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-vl-32b-instruct:free"))
    p.add_argument("--output", default=str(Path("test_results/image_analysis.json").resolve()))
    p.add_argument("--no-resume", action="store_true", help="Disable resume; rescan from scratch and overwrite output")
    p.add_argument("--resume-mode", choices=["any", "parsed_only"], default="any", help="Skip items previously touched: 'any' skips records with either parsed JSON or raw output; 'parsed_only' skips only if parsed JSON exists")
    p.add_argument("--start-after", default=None, help="If set, skip everything until this exact image path is seen, then start after it")
    p.add_argument("--progress", default=None, help="Optional path to a progress JSON file that is updated during the run")
    p.add_argument("--batch-size", type=int, default=10, help="Process images in batches of this size")
    p.add_argument("--progress-interval", type=int, default=3, help="Write progress every N items")
    p.add_argument("--memory-cleanup-interval", type=int, default=20, help="Run garbage collection every N items")
    return p


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    output_path = Path(args.output)
    if args.no_resume and output_path.exists():
        output_path.unlink()
    progress_path = Path(args.progress) if args.progress else output_path.with_suffix('.progress.json')
    cfg = ScanConfig(
        mapping_json=Path(args.mapping_json) if args.mapping_json else None,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        model=args.model,
        output=output_path,
        resume=not args.no_resume,
        limit=args.limit,
        do_ocr=args.ocr,
        resume_mode=args.resume_mode,
        start_after=args.start_after,
        progress=progress_path,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
        memory_cleanup_interval=args.memory_cleanup_interval,
    )
    pairs = load_image_list(cfg.mapping_json, cfg.images_dir)
    if not pairs:
        print("No images found to analyze.")
        return 0
    res = analyze_images(pairs, cfg)
    # Results are written incrementally, just print summary
    print(f"Completed: {res.get('count', 0)} new analyses, {res.get('skipped', 0)} skipped, {res.get('total', 0)} total")
    return 0


def _get_memory_usage_mb_safe() -> float:
    """Get current memory usage in MB with error handling."""
    try:
        from .utils import get_memory_usage_mb
        return get_memory_usage_mb()
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return -1.0

if __name__ == "__main__":
    raise SystemExit(main())

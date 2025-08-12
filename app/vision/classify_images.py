#!/usr/bin/env python3
"""
Batch Image Classification CLI

Classify and organize the large collection of unclassified images using the 
BulletproofImageAnalyzer system. This implements Enhanced PRD Phase 2: Intelligence 
- Image Classification component.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import shutil

# Avoid manipulating sys.path; rely on package imports

from app.vision.image_analyzer import image_analyzer
from app.core.logging import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify unclassified images using vision system")
    base_dir = Path(__file__).parent.parent.parent.resolve()
    parser.add_argument(
        "--base-dir",
        default=str(base_dir),
        help="Base directory to resolve default paths"
    )
    parser.add_argument(
        "--unknown-dir",
        default=str((base_dir / "data/images/unknown").resolve()),
        help="Directory containing unclassified images"
    )
    parser.add_argument(
        "--output-dir", 
        default=str((base_dir / "data/images").resolve()),
        help="Base directory for classified images"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=20,
        help="Number of images to process per batch"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        help="Limit number of images to process (for testing)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Don't move files, just classify and report"
    )
    parser.add_argument(
        "--results-file",
        default=str((base_dir / "test_results/image_classification_results.json").resolve()),
        help="File to save classification results"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


class ImageClassifier:
    """Batch image classification using BulletproofImageAnalyzer."""
    
    def __init__(self, unknown_dir: str, output_dir: str):
        self.unknown_dir = Path(unknown_dir)
        self.output_dir = Path(output_dir)
        self.analyzer = image_analyzer
        
        # Ensure output directories exist
        self.category_dirs = {
            'greeks_chart': self.output_dir / 'greeks_chart',
            'flow_heatmap': self.output_dir / 'flow_heatmap',
            'skew_chart': self.output_dir / 'skew_chart',
            'price_chart': self.output_dir / 'price_chart',
            'position_diagram': self.output_dir / 'position_diagram',
            'unclassified': self.output_dir / 'unclassified'
        }
        
        for category_dir in self.category_dirs.values():
            category_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self, limit: Optional[int] = None) -> List[Path]:
        """Get list of image files to process."""
        if not self.unknown_dir.exists():
            logger.warning(f"Unknown directory does not exist: {self.unknown_dir}")
            return []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in self.unknown_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        # Sort for consistent processing
        image_files.sort()
        
        if limit:
            image_files = image_files[:limit]
        
        logger.info(f"Found {len(image_files)} images to process")
        return image_files
    
    async def classify_image(self, image_path: Path) -> Dict:
        """Classify a single image using the vision analyzer."""
        try:
            # Read image file as bytes
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Use the comprehensive analysis from BulletproofImageAnalyzer
            analysis = await self.analyzer.analyze_image_comprehensive(image_data, str(image_path))
            
            if analysis.get('processing_status') == 'completed':
                image_type = analysis.get('image_type', 'unclassified')
                confidence = analysis.get('ocr_confidence', 0.0)
                
                result = {
                    'image_path': str(image_path),
                    'filename': image_path.name,
                    'classification': image_type,
                    'confidence': confidence,
                    'ocr_text': analysis.get('ocr_text', ''),
                    'numerical_data': analysis.get('extracted_data', {}),
                    'sentiment': analysis.get('sentiment_indicators', {}),
                    'success': True,
                    'processing_time': 0.0  # Not provided in current analysis
                }
                
                logger.debug(f"Classified {image_path.name}: {image_type} (confidence: {confidence:.2f})")
                return result
                
            else:
                error_msg = analysis.get('error', 'Unknown classification error')
                logger.warning(f"Classification failed for {image_path.name}: {error_msg}")
                return {
                    'image_path': str(image_path),
                    'filename': image_path.name,
                    'classification': 'unclassified',
                    'confidence': 0.0,
                    'success': False,
                    'error': error_msg
                }
                
        except Exception as e:
            logger.error(f"Exception classifying {image_path.name}: {e}")
            return {
                'image_path': str(image_path),
                'filename': image_path.name,
                'classification': 'unclassified',
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def move_classified_image(self, image_path: Path, classification: str, dry_run: bool = False):
        """Move image to appropriate category directory."""
        target_dir = self.category_dirs.get(classification, self.category_dirs['unclassified'])
        target_path = target_dir / image_path.name
        
        if dry_run:
            logger.debug(f"DRY RUN: Would move {image_path.name} to {target_dir.name}/")
            return True
        
        try:
            if target_path.exists():
                # Handle duplicate names
                stem = target_path.stem
                suffix = target_path.suffix
                counter = 1
                while target_path.exists():
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.move(str(image_path), str(target_path))
            logger.debug(f"Moved {image_path.name} to {target_dir.name}/")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move {image_path.name}: {e}")
            return False
    
    async def classify_batch(self, image_files: List[Path], dry_run: bool = False, concurrency: int = 5) -> List[Dict]:
        """Classify a batch of images.

        Args:
            image_files: list of image paths to classify
            dry_run: if True, do not move files
            concurrency: maximum number of concurrent classification tasks
        """
        batch_results = []
        
        # Process images concurrently but with controlled parallelism
        MAX_CONCURRENCY = 50
        if not isinstance(concurrency, int) or concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
        if concurrency > MAX_CONCURRENCY:
            raise ValueError(f"concurrency must be <= {MAX_CONCURRENCY}")
        
        try:
            semaphore = asyncio.Semaphore(concurrency)  # Limit concurrent processing
        except Exception as e:
            logger.error(f"Failed to create semaphore with concurrency {concurrency}: {e}")
            raise
        
        async def classify_with_semaphore(image_path):
            async with semaphore:
                return await self.classify_image(image_path)
        
        # Execute batch classification
        classification_tasks = [classify_with_semaphore(img) for img in image_files]
        try:
            results = await asyncio.gather(*classification_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Batch classification failed with concurrency {concurrency}: {e}")
            raise
        
        # Process results and move files
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch classification failed for {image_files[i].name}: {result}")
                result = {
                    'image_path': str(image_files[i]),
                    'filename': image_files[i].name,
                    'classification': 'unclassified',
                    'success': False,
                    'error': str(result)
                }
            
            batch_results.append(result)
            
            # Move file to appropriate directory if classification successful
            if result.get('success') and result.get('classification'):
                self.move_classified_image(
                    image_files[i], 
                    result['classification'], 
                    dry_run
                )
        
        return batch_results
    
    async def run_classification(self, batch_size: int = 20, limit: Optional[int] = None, 
                               dry_run: bool = False, concurrency: int = 5) -> Dict:
        """Run the full image classification process."""
        logger.info("Starting batch image classification process")
        MAX_BATCH_SIZE = 1024
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be <= {MAX_BATCH_SIZE}")
        
        # Get images to process
        image_files = self.get_image_files(limit)
        if not image_files:
            logger.warning("No images to process")
            return {'status': 'no_images', 'results': []}
        
        # Process in batches
        all_results = []
        start_time = time.time()
        
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(image_files) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}: {len(batch)} images")
            
            batch_results = await self.classify_batch(batch, dry_run, concurrency=concurrency)
            all_results.extend(batch_results)
            
            # Log batch statistics
            successful = sum(1 for r in batch_results if r.get('success'))
            logger.info(f"Batch {batch_num} complete: {successful}/{len(batch)} successful")
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        stats = self.calculate_statistics(all_results, processing_time)
        
        logger.info("=== CLASSIFICATION STATISTICS ===")
        logger.info(f"Total images processed: {stats['total_processed']}")
        logger.info(f"Successful classifications: {stats['successful']} ({stats['success_rate']:.1f}%)")
        logger.info(f"Processing time: {stats['processing_time']:.1f} seconds")
        logger.info(f"Average time per image: {stats['avg_time_per_image']:.2f} seconds")
        logger.info("\nClassification Distribution:")
        for category, count in stats['category_distribution'].items():
            percentage = count / stats['total_processed'] * 100 if stats['total_processed'] else 0
            logger.info(f"  {category}: {count} ({percentage:.1f}%)")
        
        return {
            'status': 'completed',
            'statistics': stats,
            'results': all_results
        }
    
    def calculate_statistics(self, results: List[Dict], processing_time: float) -> Dict:
        """Calculate comprehensive statistics from classification results."""
        total = len(results)
        successful = sum(1 for r in results if r.get('success'))
        
        # Category distribution
        category_counts = {}
        confidence_scores = []
        
        for result in results:
            category = result.get('classification', 'unclassified')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if result.get('confidence'):
                confidence_scores.append(result['confidence'])
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_processed': total,
            'successful': successful,
            'success_rate': successful / total * 100 if total else 0,
            'processing_time': processing_time,
            'avg_time_per_image': processing_time / total if total else 0,
            'category_distribution': dict(sorted(category_counts.items())),
            'average_confidence': avg_confidence,
            'high_confidence_classifications': sum(1 for c in confidence_scores if c >= 0.7)
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save classification results to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            results['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'unknown_directory': str(self.unknown_dir),
                'output_directory': str(self.output_dir),
                'analyzer_version': getattr(self.analyzer, '__version__', 'BulletproofImageAnalyzer_v1.0')
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise


async def main_async(args: argparse.Namespace):
    """Main async classification process."""
    classifier = ImageClassifier(args.unknown_dir, args.output_dir)
    
    results = await classifier.run_classification(
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        concurrency=5
    )
    
    # Save results
    classifier.save_results(results, args.results_file)
    
    return results['status'] == 'completed'


def main():
    """Main classification CLI."""
    args = parse_args()
    
    # Set up logging using existing handlers if present
    level = logging.DEBUG if args.verbose else logging.INFO
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(level)
    
    try:
        success = asyncio.run(main_async(args))
        logger.info("Image classification completed successfully" if success else "Image classification completed with issues")
        return success
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
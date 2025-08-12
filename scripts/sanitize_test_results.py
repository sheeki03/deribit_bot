#!/usr/bin/env python3
"""
Script to sanitize test results by removing PII and normalizing paths.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import argparse


def sanitize_path(path: str) -> str:
    """Convert absolute paths to relative, safe paths."""
    if not path:
        return path
    
    # Remove user-specific paths and project paths
    path = re.sub(r'/Users/[^/]+/', '', path)
    path = re.sub(r'/home/[^/]+/', '', path)
    path = re.sub(r'.*deribit[_\s]*(option[_\s]*)?bot/', '', path)
    
    # Normalize image paths
    path = re.sub(r'data/images_cleaned/[a-f0-9]{2}/', 'data/images/', path)
    path = path.replace('/images/unknown/', '/data/images/unknown/')
    
    # Clean up duplicated segments
    path = re.sub(r'/data/data/images/', '/data/images/', path)
    
    # Ensure relative path format
    path = path.lstrip('/')
    
    return path


def sanitize_result_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Restructure result entry to separate processing and inference."""
    
    # Create new structure
    sanitized = {}
    
    # Processing section
    processing_success = entry.get('success', True)  
    processing_error = entry.get('error')
    processing_time = entry.get('request_duration', 0.0)
    
    sanitized['processing'] = {
        'success': processing_success,
        'status': 'ok' if processing_success else 'error',
        'error_code': None,
        'error_message': processing_error if processing_error else None,
        'processing_time': max(0.001, processing_time),  # Never 0.0
        'retry_count': 0
    }
    
    # Inference section
    model_name = entry.get('model_used', 'qwen/qwen2.5-vl-32b-instruct:free')
    classification = None
    confidence = 0.0
    
    if entry.get('gpt5'):
        # Extract classification from parsed result
        parsed = entry['gpt5']
        if isinstance(parsed, dict):
            interpretation = parsed.get('interpretation', {})
            confidence = interpretation.get('confidence', 0.0)
            
            # Determine classification from chart type or content
            chart_meta = parsed.get('chart_meta', {})
            chart_type = chart_meta.get('chart_type', 'chart')
            classification = chart_type
    
    # Normalize model name for fixtures
    if 'qwen' in model_name.lower():
        model_name = 'qwen2.5-vl-32b'
    elif 'gpt' in model_name.lower():
        model_name = 'gpt-4-vision'
    else:
        model_name = 'vision-model'
    
    sanitized['inference'] = {
        'task': 'image_analysis',
        'model': model_name,
        'version': 'v1.0',
        'classification': classification,
        'confidence': confidence
    }
    
    # Sanitize paths
    sanitized['image_path'] = sanitize_path(entry.get('path', ''))
    
    # Include other relevant fields
    sanitized['ocr_text'] = entry.get('ocr_text')
    sanitized['numerical_data'] = {}
    sanitized['sentiment'] = {}
    sanitized['errors'] = []
    
    if processing_error:
        sanitized['errors'].append({
            'type': 'processing_error',
            'message': processing_error
        })
    
    # Add request ID for tracing
    sanitized['request_id'] = f"req_{hash(entry.get('path', '')) % 1000000:06d}"
    
    return sanitized


def sanitize_json_file(input_path: Path, output_path: Path, sample_only: bool = False):
    """Sanitize a JSON results file."""
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Create sanitized structure
    sanitized_data = {
        'metadata': {
            'unknown_directory': 'data/images/unknown',
            'output_directory': 'test_results/',
            'timestamp': '1970-01-01T00:00:00Z',  # Static timestamp
            'analyzer_version': 'BulletproofImageAnalyzer_v1.0'
        },
        'results': []
    }
    
    # Process image entries
    images = data.get('images', [])
    if sample_only:
        # Take just first few entries for sample
        images = images[:3]
    
    for entry in images:
        try:
            sanitized_entry = sanitize_result_entry(entry)
            sanitized_data['results'].append(sanitized_entry)
        except Exception as e:
            print(f"Warning: Failed to sanitize entry: {e}")
            continue
    
    # Add summary stats
    sanitized_data['summary'] = {
        'total_processed': len(sanitized_data['results']),
        'successful': sum(1 for r in sanitized_data['results'] if r['processing']['success']),
        'errors': sum(1 for r in sanitized_data['results'] if not r['processing']['success'])
    }
    
    # Write sanitized data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sanitized_data, f, indent=2)
    
    print(f"Sanitized {len(images)} entries -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sanitize test results JSON")
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('--output', help='Output JSON file path', default=None)
    parser.add_argument('--sample', action='store_true', help='Create sample fixture only')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    elif args.sample:
        output_path = Path('tests/fixtures/image_analysis.sample.json')
    else:
        output_path = input_path.with_stem(input_path.stem + '_sanitized')
    
    try:
        sanitize_json_file(input_path, output_path, sample_only=args.sample)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
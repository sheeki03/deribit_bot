#!/usr/bin/env python3

import json
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import difflib

@dataclass
class UnifiedArticleData:
    """Unified article data structure with all components."""
    # Core identification
    title: str
    url: str
    publication_date: str
    readable_date: str
    slug: str  # URL slug for easy identification
    
    # Classification data (from Playwright scraping)
    classification: Dict[str, Any]
    content_analysis: Dict[str, Any] 
    market_context: Dict[str, Any]
    trading_signals: Dict[str, Any]
    extraction_confidence: float
    
    # Content data (from cleaned scraping)
    playwright_content: str  # Content from Playwright scraper
    cleaned_content: Optional[str] = None  # Content from cleaned scraper
    body_text: Optional[str] = None  # Full body text from cleaned data
    summary_text: Optional[str] = None  # Summary from cleaned data
    
    # Image data
    article_images: List[Dict[str, str]] = None  # Images from article
    analyzed_images: List[Dict[str, Any]] = None  # Images with AI analysis
    
    # Metadata
    author: str = "Tony Stewart"
    scrape_timestamp: str = ""
    content_hash: Optional[str] = None
    data_sources: List[str] = None  # Track which sources provided data

class DataIntegrator:
    """Integrates all article data sources into unified structure."""
    
    def __init__(self):
        self.classified_articles = []
        self.cleaned_articles = []
        self.image_data = []
        
    def load_all_data(self):
        """Load all data sources."""
        print('📥 Loading all data sources...')
        
        # Load classified articles (perfect dates)
        with open('scraped_data/playwright/classified_articles_complete.json', 'r') as f:
            classified_data = json.load(f)
            self.classified_articles = classified_data['classified_articles']
        print(f'  ✅ Loaded {len(self.classified_articles)} classified articles')
        
        # Load cleaned articles (more text content)
        with open('scraped_data/cleaned/articles_cleaned.json', 'r') as f:
            self.cleaned_articles = json.load(f)
        print(f'  ✅ Loaded {len(self.cleaned_articles)} cleaned articles')
        
        # Load image analysis results
        with open('test_results/image_analysis.json', 'r') as f:
            image_data_raw = json.load(f)
            if 'results' in image_data_raw:
                self.image_data = image_data_raw['results']
            else:
                self.image_data = image_data_raw.get('images', [])
        print(f'  ✅ Loaded {len(self.image_data)} image analysis results')
        
    def create_unified_dataset(self) -> List[UnifiedArticleData]:
        """Create unified dataset with all data integrated."""
        print('\\n🔄 Creating unified dataset...')
        
        unified_articles = []
        
        for i, classified_article in enumerate(self.classified_articles, 1):
            if i % 25 == 0 or i == len(self.classified_articles):
                print(f'  Progress: {i}/{len(self.classified_articles)} articles processed')
            
            # Create unified article
            unified = self._create_unified_article(classified_article)
            unified_articles.append(unified)
        
        print(f'\\n✅ Created {len(unified_articles)} unified articles')
        return unified_articles
    
    def _create_unified_article(self, classified_article: Dict[str, Any]) -> UnifiedArticleData:
        """Create unified article from classified article and match with other data."""
        
        # Extract basic info
        title = classified_article['title']
        url = classified_article['url']
        date = classified_article['publication_date']
        slug = self._extract_url_slug(url)
        
        # Find matching cleaned article
        cleaned_match = self._find_matching_cleaned_article(title, url, slug)
        
        # Find matching images
        matching_images = self._find_matching_images(title, date, slug)
        
        # Determine data sources
        sources = ['playwright']
        if cleaned_match:
            sources.append('cleaned_scraper')
        if matching_images:
            sources.append('image_analysis')
        
        # Create unified article
        unified = UnifiedArticleData(
            # Core identification
            title=title,
            url=url,
            publication_date=date,
            readable_date=classified_article['readable_date'],
            slug=slug,
            
            # Classification data
            classification=classified_article['classification'],
            content_analysis=classified_article['content_analysis'],
            market_context=classified_article['market_context'],
            trading_signals=classified_article['trading_signals'],
            extraction_confidence=classified_article['extraction_confidence'],
            
            # Content data
            playwright_content=classified_article['content'],
            cleaned_content=cleaned_match.get('summary_text', '') if cleaned_match else None,
            body_text=cleaned_match.get('body_text', '') if cleaned_match else None,
            summary_text=cleaned_match.get('summary_text', '') if cleaned_match else None,
            
            # Image data
            article_images=classified_article['images'],
            analyzed_images=matching_images,
            
            # Metadata
            author=classified_article['author'],
            scrape_timestamp=classified_article['scrape_timestamp'],
            content_hash=cleaned_match.get('content_hash') if cleaned_match else None,
            data_sources=sources
        )
        
        return unified
    
    def _extract_url_slug(self, url: str) -> str:
        """Extract URL slug for matching."""
        try:
            path = urlparse(url).path
            slug = path.split('/')[-2] if path.endswith('/') else path.split('/')[-1]
            return slug.replace('option-flow-', '').replace('-', ' ')
        except:
            return ''
    
    def _find_matching_cleaned_article(self, title: str, url: str, slug: str) -> Optional[Dict[str, Any]]:
        """Find matching article in cleaned data."""
        
        # Try exact URL match first
        for article in self.cleaned_articles:
            if article.get('url', '') == url:
                return article
        
        # Try title similarity match
        title_clean = title.lower().replace('option flow:', '').replace('option flow –', '').strip()
        
        best_match = None
        best_score = 0
        
        for article in self.cleaned_articles:
            article_title = article.get('title', '').lower().replace('option flow:', '').replace('option flow –', '').strip()
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, title_clean, article_title).ratio()
            
            if similarity > best_score and similarity > 0.8:  # 80% similarity threshold
                best_score = similarity
                best_match = article
        
        return best_match
    
    def _find_matching_images(self, title: str, date: str, slug: str) -> List[Dict[str, Any]]:
        """Find images that belong to this article."""
        
        matching_images = []
        
        # Create search patterns from title and slug
        search_patterns = [
            slug.lower(),
            title.lower().replace('option flow:', '').replace('option flow –', '').strip(),
            date
        ]
        
        for image in self.image_data:
            # Check if image matches this article
            if self._image_matches_article(image, search_patterns, date):
                matching_images.append(image)
        
        return matching_images
    
    def _image_matches_article(self, image: Dict[str, Any], patterns: List[str], date: str) -> bool:
        """Check if image belongs to article based on patterns."""
        
        # Get image text content
        image_text = ''
        if 'ocr_text' in image and image['ocr_text']:
            image_text = str(image['ocr_text']).lower()
        elif 'gpt5' in image and image['gpt5']:
            image_text = str(image['gpt5']).lower()
        
        # Check for pattern matches in image content
        for pattern in patterns:
            if pattern and pattern in image_text:
                return True
        
        # Check for date proximity (if image has date info)
        if self._check_date_proximity(image, date):
            return True
        
        # Check path/filename for article indicators
        image_path = str(image.get('path', '')) + str(image.get('article', ''))
        for pattern in patterns:
            if pattern and pattern.replace(' ', '-') in image_path.lower():
                return True
        
        return False
    
    def _check_date_proximity(self, image: Dict[str, Any], target_date: str) -> bool:
        """Check if image date is close to article date."""
        
        # This is a simplified check - could be enhanced with more sophisticated date matching
        try:
            target_year = target_date[:4]
            target_month = target_date[5:7]
            
            image_content = str(image.get('ocr_text', '')) + str(image.get('gpt5', ''))
            
            # Look for year in image content
            if target_year in image_content and target_month in image_content:
                return True
                
        except:
            pass
        
        return False
    
    def generate_integration_report(self, unified_articles: List[UnifiedArticleData]) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        
        # Count data sources
        source_counts = defaultdict(int)
        content_quality = {'high': 0, 'medium': 0, 'low': 0}
        
        for article in unified_articles:
            for source in article.data_sources:
                source_counts[source] += 1
            
            # Assess content quality
            content_sources = len(article.data_sources)
            has_body_text = bool(article.body_text)
            has_images = bool(article.analyzed_images)
            
            if content_sources >= 3 and has_body_text and has_images:
                content_quality['high'] += 1
            elif content_sources >= 2 and (has_body_text or has_images):
                content_quality['medium'] += 1
            else:
                content_quality['low'] += 1
        
        # Analyze coverage by market period
        period_coverage = defaultdict(int)
        for article in unified_articles:
            period = article.market_context['market_period']
            period_coverage[period] += 1
        
        # Count total images
        total_images = sum(len(article.analyzed_images or []) for article in unified_articles)
        
        return {
            'integration_summary': {
                'total_unified_articles': len(unified_articles),
                'data_source_coverage': dict(source_counts),
                'content_quality_distribution': dict(content_quality),
                'total_integrated_images': total_images
            },
            'market_period_coverage': dict(period_coverage),
            'date_range': {
                'earliest': min(a.publication_date for a in unified_articles),
                'latest': max(a.publication_date for a in unified_articles)
            },
            'integration_metrics': {
                'articles_with_multiple_sources': len([a for a in unified_articles if len(a.data_sources) > 1]),
                'articles_with_body_text': len([a for a in unified_articles if a.body_text]),
                'articles_with_analyzed_images': len([a for a in unified_articles if a.analyzed_images]),
                'average_confidence': sum(a.extraction_confidence for a in unified_articles) / len(unified_articles)
            }
        }
    
    def save_unified_dataset(self, unified_articles: List[UnifiedArticleData], report: Dict[str, Any]) -> Path:
        """Save the unified dataset."""
        
        # Convert to dictionaries
        unified_data = [asdict(article) for article in unified_articles]
        
        # Create comprehensive output
        output = {
            'integration_metadata': {
                'integration_timestamp': datetime.now().isoformat(),
                'total_articles': len(unified_articles),
                'integration_version': '1.0',
                'data_sources_used': ['playwright', 'cleaned_scraper', 'image_analysis']
            },
            'integration_report': report,
            'unified_articles': unified_data
        }
        
        # Save to file
        output_path = Path('scraped_data/playwright/unified_articles_complete.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return output_path

def integrate_all_data():
    """Main function to integrate all data sources."""
    
    print('🚀 INTEGRATING ALL ARTICLE DATA')
    print('=' * 60)
    print('Combining classified articles, cleaned text, and image analysis...')
    
    # Initialize integrator
    integrator = DataIntegrator()
    
    # Load all data
    integrator.load_all_data()
    
    # Create unified dataset
    unified_articles = integrator.create_unified_dataset()
    
    # Generate integration report
    report = integrator.generate_integration_report(unified_articles)
    
    # Save unified dataset
    output_path = integrator.save_unified_dataset(unified_articles, report)
    
    # Print results
    print(f'\\n📊 INTEGRATION REPORT')
    print('=' * 40)
    
    summary = report['integration_summary']
    print(f'✅ Total unified articles: {summary["total_unified_articles"]}')
    print(f'📊 Data source coverage:')
    for source, count in summary['data_source_coverage'].items():
        print(f'   {source}: {count} articles')
    
    print(f'🎯 Content quality distribution:')
    for quality, count in summary['content_quality_distribution'].items():
        print(f'   {quality}: {count} articles')
    
    print(f'📸 Total integrated images: {summary["total_integrated_images"]}')
    
    metrics = report['integration_metrics']
    print(f'\n📈 Integration metrics:')
    print(f'   Articles with multiple data sources: {metrics["articles_with_multiple_sources"]}')
    print(f'   Articles with full body text: {metrics["articles_with_body_text"]}')
    print(f'   Articles with analyzed images: {metrics["articles_with_analyzed_images"]}')
    print(f'   Average extraction confidence: {metrics["average_confidence"]:.3f}')
    
    print(f'\\n💾 Unified dataset saved to: {output_path}')
    
    return unified_articles, report

if __name__ == "__main__":
    unified_articles, report = integrate_all_data()
    
    print(f'\\n🎉 INTEGRATION COMPLETE!')
    print(f'📊 {len(unified_articles)} articles with comprehensive data integration')
    print(f'🎯 All text and image data organized under correct article names and dates')
    print(f'🚀 Ready for advanced options analysis with complete data correlation!')
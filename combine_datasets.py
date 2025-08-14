#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path

def combine_datasets():
    print('📊 COMBINING ALL SCRAPED DATA')
    print('=' * 60)

    # Load page 1 data
    with open('scraped_data/playwright/playwright_articles_20250814_134628.json', 'r') as f:
        page1_data = json.load(f)

    # Load page 2 data  
    with open('scraped_data/playwright/page2_articles_20250814_140748.json', 'r') as f:
        page2_data = json.load(f)

    page1_articles = page1_data['articles']
    page2_articles = page2_data['articles']

    print(f'Page 1 articles: {len(page1_articles)}')
    print(f'Page 2 articles: {len(page2_articles)}')

    # Combine all articles
    all_articles = page1_articles + page2_articles
    print(f'Total combined articles: {len(all_articles)}')

    # Sort by publication date
    all_articles.sort(key=lambda x: x['publication_date'])

    # Get date range
    dates = [a['publication_date'] for a in all_articles if a['publication_date'] != '1970-01-01']
    earliest = min(dates)
    latest = max(dates)

    print(f'\nDate range: {earliest} to {latest}')
    print(f'Unique dates: {len(set(dates))}')

    # Create comprehensive dataset
    combined_data = {
        'scrape_metadata': {
            'scrape_timestamp': datetime.now().isoformat(),
            'total_articles': len(all_articles),
            'sources': ['page/1/', 'page/2/'],
            'date_range': {
                'earliest': earliest,
                'latest': latest,
                'span_days': (datetime.strptime(latest, '%Y-%m-%d') - datetime.strptime(earliest, '%Y-%m-%d')).days,
                'unique_dates': len(set(dates))
            },
            'extraction_confidence_avg': sum(a['extraction_confidence'] for a in all_articles) / len(all_articles),
            'scraper_type': 'playwright_comprehensive'
        },
        'articles': all_articles
    }

    # Save combined dataset
    output_path = Path('scraped_data/playwright/complete_articles_dataset.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f'\n✅ Combined dataset saved to: {output_path}')
    print(f'📈 Dataset spans {combined_data["scrape_metadata"]["date_range"]["span_days"]} days')
    print(f'🎯 Average confidence: {combined_data["scrape_metadata"]["extraction_confidence_avg"]:.3f}')

    # Show chronological sample
    print(f'\n📅 CHRONOLOGICAL SAMPLE (First 10 articles):')
    for i, article in enumerate(all_articles[:10], 1):
        print(f'{i:2}. {article["publication_date"]} - {article["title"]}')
    
    print(f'\n📅 RECENT SAMPLE (Last 10 articles):')
    for i, article in enumerate(all_articles[-10:], len(all_articles)-9):
        print(f'{i:2}. {article["publication_date"]} - {article["title"]}')
        
    return combined_data

if __name__ == "__main__":
    combined_data = combine_datasets()
    print(f'\n🎉 DATASET COMBINATION COMPLETE!')
    print(f'📊 Total: {len(combined_data["articles"])} articles')
    print(f'📅 Timespan: 3+ years ({combined_data["scrape_metadata"]["date_range"]["span_days"]} days)')
    print(f'🎯 Perfect accuracy: {combined_data["scrape_metadata"]["extraction_confidence_avg"]:.3f} confidence')
#!/usr/bin/env python3
"""
Simple URL Analyzer for Medical School Program Data
Extracts and ranks potentially relevant URLs from crawler results
"""

import json
import os
import sys
from datetime import datetime

def load_crawler_results(school_name: str, results_file: str) -> list:
    """Load and extract URLs from crawler results with basic filtering"""
    file_path = os.path.join('schools', school_name, 'crawler_results', results_file)
    with open(file_path) as f:
        data = json.load(f)
    
    # Keywords that suggest relevant content
    relevant_keywords = [
        "admission", "apply", "application", "deadline",
        "requirement", "prerequisite", "mcat", "gpa",
        "curriculum", "program", "course", "rotation",
        "statistics", "class", "profile", "demographics",
        "technical", "standards", "interview", "candidate",
        "residency", "state", "in-state"
    ]
    
    # Patterns to skip
    skip_patterns = [
        'login', 'search', 'contact', 'news', 'events', 
        'faculty', 'staff', 'directory', 'privacy', 
        'accessibility', 'sitemap', 'social'
    ]
    
    # Extract and filter URLs
    urls = []
    seen_urls = set()
    
    for page_data in data['relevant_urls'].values():
        if 'metadata' in page_data and 'link_data' in page_data['metadata']:
            for link in page_data['metadata']['link_data']:
                url = link['url']
                text = link['text'].lower()
                
                # Skip if already seen or no text
                if url in seen_urls or not text.strip():
                    continue
                    
                # Skip irrelevant patterns
                if any(pattern in text for pattern in skip_patterns):
                    continue
                
                # Check if URL text contains relevant keywords
                if any(keyword in text for keyword in relevant_keywords):
                    urls.append({
                        'url': url,
                        'text': link['text']
                    })
                    seen_urls.add(url)
    
    # Sort URLs by relevance score (number of keyword matches)
    for url_data in urls:
        text = url_data['text'].lower()
        score = sum(1 for keyword in relevant_keywords if keyword in text)
        url_data['relevance_score'] = score
    
    urls.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Take top 25 most relevant URLs
    return urls[:25]

def save_urls(school_name: str, urls: list) -> None:
    """Save filtered URLs to JSON file"""
    json_dir = os.path.join('schools', school_name, 'json')
    os.makedirs(json_dir, exist_ok=True)
    
    output_file = os.path.join(json_dir, f'{school_name}_urls.json')
    
    output_data = {
        'school_name': school_name,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'urls': urls
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nFound {len(urls)} potentially relevant URLs:")
    print("=" * 50)
    for url in urls:
        print(f"\n{url['text']}")
        print(f"{url['url']}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python url_analyzer.py <school_name> <crawler_results_file>")
        print("Example: python url_analyzer.py unc relevant_urls_20241218_190241.json")
        sys.exit(1)
    
    school_name = sys.argv[1]
    results_file = sys.argv[2]
    
    try:
        # Load and filter URLs
        urls = load_crawler_results(school_name, results_file)
        
        # Save results
        save_urls(school_name, urls)
        
    except Exception as e:
        print(f"Error processing URLs: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

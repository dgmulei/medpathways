#!/usr/bin/env python3
"""
Intelligent Medical School Website Crawler
Uses GPT-4 to analyze content and guide crawling through relevant pages
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Set, List, Optional, Tuple
import json
import logging
from datetime import datetime, timedelta
import os
import sys
from dataclasses import dataclass
import openai
from time import sleep
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please check your .env file.")

SYSTEM_PROMPT = """Medical school website crawler. Analyze page content to:
1. Match content to these taxonomy sections: {taxonomy_sections}
2. Identify relevant links to follow

Return JSON with:
{{
  "page_analysis": {{
    "relevant_sections": ["matching_section_names"],
    "section_confidence": {{"section": confidence_score}}
  }},
  "link_analysis": [{{
    "url": "link",
    "priority": "high/medium/low",
    "expected_content": ["likely_sections"]
  }}]
}}"""

@dataclass
class PageData:
    """Data structure for page content and metadata"""
    url: str
    title: Optional[str]
    content: str
    links: List[str]
    metadata: Dict[str, str]

class IntelligentCrawler:
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = os.path.join('schools', self.school_name, 'crawler_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'crawl_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def fetch_page(self, url: str) -> Optional[PageData]:
        """Fetch and parse a page, returning structured data"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else None
            
            # Extract main content more selectively
            main_content = ''
            
            # Try to find the most relevant content section
            content_selectors = [
                # Primary content areas
                'main#content', 'article', 'div.content', 'div.main-content',
                
                # Admissions-specific sections
                'div[id*="admission"]', 'div[id*="requirements"]',
                'div[class*="admission"]', 'div[class*="requirements"]',
                'div[id*="apply"]', 'div[class*="apply"]',
                
                # Program details
                'div[id*="curriculum"]', 'div[id*="program"]',
                'div[class*="curriculum"]', 'div[class*="program"]',
                
                # Requirements and standards
                'div[id*="prerequisite"]', 'div[id*="technical"]',
                'div[class*="prerequisite"]', 'div[class*="technical"]',
                
                # Student info
                'div[id*="student"]', 'div[id*="diversity"]',
                'div[class*="student"]', 'div[class*="diversity"]'
            ]
            
            for selector in content_selectors:
                content_tag = soup.select_one(selector)
                if content_tag:
                    # Extract only relevant content elements
                    for tag in content_tag.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td']):
                        # Skip navigation and footer elements
                        if not any(cls in (tag.get('class', []) + tag.parent.get('class', [])) 
                                 for cls in ['nav', 'menu', 'footer', 'copyright']):
                            main_content += tag.get_text(' ', strip=True) + ' '
                    break
            
            if not main_content:
                # Fallback: Get text from paragraphs and headers in the whole document
                main_content = ' '.join(
                    tag.get_text(' ', strip=True)
                    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                )
            
            # Extract key metadata
            metadata = {
                'h1': [h1.get_text(strip=True) for h1 in soup.find_all('h1')],
                'h2': [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:2]],  # First two h2s often indicate main topics
                'description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else None
            }
            
            # Extract and pre-filter relevant links
            links = []
            relevant_patterns = [
                # Application process
                'admission', 'apply', 'application', 'deadline',
                'requirement', 'prerequisite', 'checklist',
                
                # Academic requirements
                'mcat', 'gpa', 'grade', 'academic', 'course',
                'prerequisite', 'preparation', 'standard',
                
                # Program details
                'curriculum', 'program', 'rotation', 'clinical',
                'research', 'volunteer', 'experience',
                
                # Supporting materials
                'recommendation', 'letter', 'interview', 'statement',
                
                # Student info
                'student', 'diversity', 'inclusion', 'class',
                'profile', 'statistic', 'demographic',
                
                # Technical standards
                'technical', 'physical', 'cognitive', 'standard'
            ]
            
            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                text = a.get_text(strip=True).lower()
                
                # Only include links that:
                # 1. Stay within the same domain
                # 2. Have relevant text
                # 3. Are not file downloads
                if (link.startswith('http') and 
                    urlparse(link).netloc == self.base_domain and
                    not any(ext in link.lower() for ext in ['.pdf', '.doc', '.docx']) and
                    any(pattern in text or pattern in link.lower() for pattern in relevant_patterns)):
                    
                    links.append({
                        'url': link,
                        'text': text,
                        'title': a.get('title', '')
                    })
            
            return PageData(
                url=url,
                title=title,
                content=main_content[:500],  # First 500 chars is usually enough for context
                links=[l['url'] for l in links],
                metadata={
                    'title': title,
                    'h1': metadata['h1'],
                    'h2': metadata['h2'],
                    'description': metadata['description'],
                    'link_data': links
                }
            )
            
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None
            
    def check_api_quota(self) -> bool:
        """Check OpenAI API quota status and account details"""
        try:
            print("\n=== OpenAI Account Diagnostics ===")
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Check API Key
            print("\n1. API Key Check:")
            try:
                models = client.models.list()
                print("✓ API key is valid")
                print("✓ Authentication successful")
                
                # List available models
                print("\nAvailable Models:")
                for model in models.data:
                    if model.id.startswith(("gpt-3.5", "gpt-4")):
                        print(f"- {model.id}")
                
                # Check GPT-4 specifically
                gpt4_available = any(model.id.startswith("gpt-4") for model in models.data)
                if gpt4_available:
                    print("\n✓ GPT-4 model is available")
                else:
                    print("\n✗ GPT-4 model not available - account may need additional access")
                    return False
                    
            except openai.AuthenticationError:
                print("✗ API key is invalid")
                return False
                
            # Check API Access
            print("\n2. API Access Check:")
            try:
                print("Testing GPT-4 access...")
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1,
                    temperature=0
                )
                print("✓ GPT-4 access working")
                print("✓ Account is operational")
                return True
                
            except openai.RateLimitError as e:
                error_msg = str(e).lower()
                print("\n3. Error Analysis:")
                print("✗ API access failed")
                
                if 'quota' in error_msg:
                    print("\nQuota Issue Details:")
                    if 'exceeded' in error_msg:
                        print("- Monthly quota exceeded")
                    if 'billing' in error_msg:
                        print("- Billing verification needed")
                    if 'plan' in error_msg:
                        print("- Plan upgrade required")
                        
                    print("\nTroubleshooting Steps:")
                    print("1. Visit https://platform.openai.com/account/billing")
                    print("2. Check if payment method is verified")
                    print("3. Verify organization settings")
                    print("4. Check usage limits at https://platform.openai.com/account/limits")
                    
                else:
                    print("\nRate Limit Issue:")
                    print("- Temporary rate limit (not a quota issue)")
                    print("- Try again in a few minutes")
                    
                print(f"\nFull Error Message: {str(e)}")
                return False
        except openai.RateLimitError as e:
            if 'quota' in str(e).lower():
                print("\nERROR: OpenAI API quota exceeded")
                print("Please check your account billing status and limits")
                print(f"Error details: {str(e)}")
                return False
            else:
                print("\nRate limit hit, but not a quota issue")
                print(f"Error details: {str(e)}")
                return True
        except openai.AuthenticationError as e:
            print("\nERROR: Authentication failed")
            print("Please check your API key in the .env file")
            print(f"Error details: {str(e)}")
            return False
        except Exception as e:
            print("\nERROR: Unexpected error checking API quota")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            return False

    def __init__(self, school_name: str, base_url: str, simulation: bool = False):
        self.school_name = school_name
        self.base_url = base_url
        self.base_domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.relevant_urls: Dict[str, Dict] = {}
        
        # Token tracking
        self.total_tokens_used = 0
        self.minute_tokens = 0
        self.minute_requests = 0
        self.last_request_time = datetime.now()
        
        # API limits
        self.TPM_LIMIT = 30000  # tokens per minute
        self.RPM_LIMIT = 500    # requests per minute
        self.TPD_LIMIT = 90000  # tokens per day
        
        self.setup_logging()
        
        # Load taxonomy for context
        with open('taxonomy.json') as f:
            self.taxonomy = json.load(f)
            
        self.simulation = simulation
        
        # Check API quota before starting if not in simulation mode
        if not simulation and not self.check_api_quota():
            raise Exception("Cannot start crawler: OpenAI API quota exceeded")
            
    def check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Reset counters if it's been more than a minute
        if self.last_request_time < minute_ago:
            self.minute_tokens = 0
            self.minute_requests = 0
            
        # Check limits
        if self.minute_tokens >= self.TPM_LIMIT:
            logging.warning(f"Token per minute limit reached: {self.minute_tokens}/{self.TPM_LIMIT}")
            return False
            
        if self.minute_requests >= self.RPM_LIMIT:
            logging.warning(f"Requests per minute limit reached: {self.minute_requests}/{self.RPM_LIMIT}")
            return False
            
        if self.total_tokens_used >= self.TPD_LIMIT:
            logging.error(f"Daily token limit reached: {self.total_tokens_used}/{self.TPD_LIMIT}")
            return False
            
        return True

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """Roughly estimate tokens in messages"""
        # GPT-4 uses ~4 chars per token on average
        chars = sum(len(str(m.get('content', ''))) for m in messages)
        return chars // 4

    def simulate_gpt4_response(self, url: str, content: str, links: List[Dict]) -> Dict:
        """Simulate GPT-4 analysis for testing"""
        # Basic pattern matching to identify likely sections
        sections = []
        confidences = {}
        
        patterns = {
            'application_requirements': ['requirement', 'prerequisite', 'mcat', 'gpa'],
            'application_deadlines': ['deadline', 'date', 'timeline'],
            'program_curriculum': ['curriculum', 'course', 'rotation'],
            'interview_process': ['interview', 'mmi', 'visit'],
            'technical_standards': ['technical', 'standard', 'physical', 'cognitive'],
            'program_statistics': ['statistic', 'class', 'profile', 'demographic'],
            'ideal_candidate_profile': ['candidate', 'quality', 'trait', 'seek'],
            'program_type': ['md program', 'medical doctor', 'degree'],
            'program_name': ['school of medicine', 'medical school'],
            'program_website': ['http', 'www', '.edu'],
            'residency_preferences': ['resident', 'match', 'placement']
        }
        
        content_lower = content.lower()
        for section, keywords in patterns.items():
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            if matches > 0:
                confidence = min(0.9, matches * 0.3)  # Scale confidence with matches
                sections.append(section)
                confidences[section] = confidence
        
        # Analyze links
        link_analysis = []
        for link in links:
            link_text = link['text'].lower()
            priority = 'low'
            expected = []
            
            # Check if link text matches any patterns
            for section, keywords in patterns.items():
                if any(keyword in link_text for keyword in keywords):
                    priority = 'high'
                    expected.append(section)
            
            if priority == 'high':
                link_analysis.append({
                    "url": link['url'],
                    "relevance_score": 0.8,
                    "priority": priority,
                    "expected_content": expected
                })
        
        return {
            "page_analysis": {
                "relevant_sections": sections,
                "section_confidence": confidences
            },
            "link_analysis": link_analysis
        }

    def call_gpt4(self, messages: List[Dict]) -> Tuple[Dict, bool]:
        """Make API call to GPT-4 with retry logic and usage tracking"""
        if self.simulation:
            # Extract data from messages
            data = json.loads(messages[1]['content'])
            url = data['url']
            content = data['content']
            links = data.get('links', [])
            
            # Log simulation info
            print(f"\nSimulating GPT-4 analysis for: {url}")
            print(f"Content length: {len(content)} chars")
            print(f"Number of links: {len(links)}")
            
            return self.simulate_gpt4_response(url, content, links), True
            
        max_retries = 3
        client = openai.OpenAI(api_key=openai.api_key)
        
        # Estimate tokens before making the call
        estimated_tokens = self.estimate_tokens(messages)
        logging.info(f"Estimated tokens for request: {estimated_tokens}")
        
        if estimated_tokens + self.minute_tokens > self.TPM_LIMIT:
            sleep_time = 60 - (datetime.now() - self.last_request_time).seconds
            if sleep_time > 0:
                logging.info(f"Sleeping {sleep_time}s to reset rate limits...")
                sleep(sleep_time)
        
        for attempt in range(max_retries):
            if not self.check_rate_limits():
                sleep_time = 60 - (datetime.now() - self.last_request_time).seconds
                if sleep_time > 0:
                    logging.info(f"Rate limit reached. Sleeping {sleep_time}s...")
                    sleep(sleep_time)
                continue
                
            try:
                self.minute_requests += 1
                self.last_request_time = datetime.now()
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # Track token usage
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                self.total_tokens_used += total_tokens
                self.minute_tokens += total_tokens
                
                # Log usage statistics
                logging.info(f"API Usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                logging.info(f"Total tokens used so far: {self.total_tokens_used}")
                
                # Estimate cost ($0.03/1K prompt tokens, $0.06/1K completion tokens for GPT-4)
                prompt_cost = (prompt_tokens / 1000) * 0.03
                completion_cost = (completion_tokens / 1000) * 0.06
                total_cost = prompt_cost + completion_cost
                logging.info(f"Request cost: ${total_cost:.4f}")
                
                # Parse response
                try:
                    result = json.loads(response.choices[0].message.content)
                    return result, True
                except json.JSONDecodeError:
                    logging.error("Failed to parse GPT-4 response as JSON")
                    return {}, False
                    
            except openai.RateLimitError as e:
                if 'quota' in str(e).lower():
                    logging.error(f"Quota exceeded after using {self.total_tokens_used} tokens")
                    self.save_results()  # Save progress before exiting
                    raise Exception(f"API quota exceeded. Total tokens used: {self.total_tokens_used}")
                else:
                    logging.error(f"Rate limit error: {str(e)}")
                    if attempt < max_retries - 1:
                        sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                logging.error(f"GPT-4 API error (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)
                continue
                
        return {}, False
        
    def analyze_page(self, page_data: PageData) -> Dict:
        """
        Send page data to GPT-4 for analysis
        Returns dict with analysis results
        """
        # Prepare minimal data for GPT-4
        user_prompt = {
            "url": page_data.url,
            "headings": {
                "h1": page_data.metadata['h1'],
                "h2": page_data.metadata.get('h2', [])[:2]  # First two H2s only
            },
            "content": page_data.content[:500] if page_data.content else "",
            "links": [
                {"text": link['text'], "url": link['url']}  # Omit empty titles
                for link in page_data.metadata['link_data']
                if link['text'].strip()  # Only include links with text
            ]
        }

        # Prepare messages for GPT-4
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    taxonomy_sections=json.dumps(list(self.taxonomy["institutions"][0]["programs"][0].keys()), indent=2)
                )
            },
            {
                "role": "user",
                "content": json.dumps(user_prompt, indent=2)
            }
        ]

        # Log what we're sending to GPT-4
        logging.info(f"Analyzing {page_data.url}")
        logging.debug(f"Sending to GPT-4: {json.dumps(messages, indent=2)}")

        # Make API call
        analysis, success = self.call_gpt4(messages)
        
        if not success:
            logging.warning(f"Failed to get GPT-4 analysis for {page_data.url}")
            return {
                'url': page_data.url,
                'relevant_sections': [],
                'relevant_links': [],
                'metadata': page_data.metadata
            }
            
        # Process GPT-4's analysis
        relevant_links = [
            link['url'] 
            for link in analysis.get('link_analysis', [])
            if link['priority'] in ['high', 'medium']  # Only follow high/medium priority links
        ]
        
        return {
            'url': page_data.url,
            'relevant_sections': analysis.get('page_analysis', {}).get('relevant_sections', []),
            'relevant_links': relevant_links,
            'metadata': page_data.metadata,
            'gpt4_analysis': analysis  # Store full analysis for reference
        }
        
    def crawl(self):
        """
        Main crawling function that:
        1. Fetches pages
        2. Gets GPT-4 analysis
        3. Updates relevant URL mapping
        4. Continues crawling relevant links
        """
        to_crawl = [(self.base_url, 'high')]  # (url, priority)
        required_sections = set(self.taxonomy["institutions"][0]["programs"][0].keys())
        found_sections = set()
        start_time = datetime.now()
        max_duration = timedelta(minutes=5)  # Stop after 5 minutes
        
        while to_crawl and (datetime.now() - start_time) < max_duration:
            # Sort by priority and take highest priority URL
            to_crawl.sort(key=lambda x: 0 if x[1] == 'high' else 1 if x[1] == 'medium' else 2)
            url, priority = to_crawl.pop(0)
            
            if url in self.visited_urls:
                continue
                
            self.visited_urls.add(url)
            # Print to console and log
            print(f"\n{'='*80}")
            print(f"Crawling: {url}")
            print(f"Progress: {len(found_sections)}/{len(required_sections)} sections found")
            print(f"Found: {sorted(list(found_sections))}")
            print(f"Missing: {sorted(list(required_sections - found_sections))}")
            print(f"{'='*80}\n")
            
            logging.info(f"Crawling: {url}")
            logging.info(f"Progress: Found {len(found_sections)}/{len(required_sections)} required sections")
            logging.info(f"Found sections: {sorted(list(found_sections))}")
            logging.info(f"Missing sections: {sorted(list(required_sections - found_sections))}")
            
            # Fetch page
            page_data = self.fetch_page(url)
            if not page_data:
                continue
                
            # Get GPT-4 analysis
            analysis = self.analyze_page(page_data)
            
            # Save results if page is relevant
            if analysis['relevant_sections']:
                self.relevant_urls[url] = analysis
                found_sections.update(analysis['relevant_sections'])
                
                # If we've found all required sections, we can stop
                if found_sections.issuperset(required_sections):
                    logging.info("Found all required sections! Crawl complete.")
                    break
                
            # Add relevant links to crawl queue with their priorities
            for link_data in analysis.get('gpt4_analysis', {}).get('link_analysis', []):
                if link_data['url'] not in self.visited_urls:
                    to_crawl.append((link_data['url'], link_data['priority']))
                    
            # Save progress
            self.save_results()
            
    def save_results(self):
        """Save crawling results to file"""
        results_dir = os.path.join('schools', self.school_name, 'crawler_results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'relevant_urls_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump({
                'school_name': self.school_name,
                'base_url': self.base_url,
                'relevant_urls': self.relevant_urls,
                'crawl_stats': {
                    'total_pages_visited': len(self.visited_urls),
                    'relevant_pages_found': len(self.relevant_urls),
                    'timestamp': timestamp
                }
            }, f, indent=2)
            
def main():
    if len(sys.argv) < 3:
        print("Usage: python intelligent_crawler.py <school_name> <base_url> [--simulate]")
        print("Example: python intelligent_crawler.py unc https://www.med.unc.edu/admit/")
        sys.exit(1)
        
    school_name = sys.argv[1]
    base_url = sys.argv[2]
    simulation = "--simulate" in sys.argv
    
    if simulation:
        print("\nRunning in simulation mode (no API calls)")
        print("This will test content extraction and link filtering")
        print("Token usage will be estimated but no actual API calls made\n")
    
    crawler = IntelligentCrawler(school_name, base_url, simulation)
    crawler.crawl()

if __name__ == "__main__":
    main()

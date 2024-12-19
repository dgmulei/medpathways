#!/usr/bin/env python3
"""
Medical School Content Processor
Uses AI to intelligently process and extract medical school program information
"""

import json
import os
import sys
from datetime import datetime
import logging
import asyncio
from pyppeteer import launch
import time
from urllib.parse import urljoin, urlparse
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def load_taxonomy():
    """Load and validate taxonomy schema"""
    try:
        # First try current directory
        if os.path.exists('taxonomy.json'):
            with open('taxonomy.json', 'r') as f:
                taxonomy = json.load(f)
        # Then try project root
        else:
            root_taxonomy = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'taxonomy.json')
            with open(root_taxonomy, 'r') as f:
                taxonomy = json.load(f)
        
        # Validate schema
        if not taxonomy.get('institutions'):
            raise ValueError("Invalid taxonomy: missing 'institutions' key")
        if not taxonomy['institutions'][0].get('programs'):
            raise ValueError("Invalid taxonomy: missing 'programs' key")
        
        return taxonomy['institutions'][0]['programs'][0]
    except Exception as e:
        raise Exception(f"Failed to load taxonomy schema: {str(e)}")

# Load program schema
PROGRAM_SCHEMA = load_taxonomy()

class MedSchoolProcessor:
    def __init__(self, school_name: str):
        self.school_name = school_name
        self.setup_logging()
        self.program_data = {k: {} for k in PROGRAM_SCHEMA.keys()}
        
    def setup_logging(self):
        """Configure logging"""
        log_dir = os.path.join('schools', self.school_name, 'processor_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'processor_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def get_page_content(self, url: str, browser) -> str:
        """Get cleaned page content using Puppeteer"""
        try:
            # Skip non-web content
            if any(ext in url.lower() for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']):
                logging.info(f"Skipping non-web content: {url}")
                return None
                
            page = await browser.newPage()
            response = await page.goto(url, {'waitUntil': 'networkidle0'})
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('text/html'):
                logging.info(f"Skipping non-HTML content ({content_type}): {url}")
                await page.close()
                return None
            
            # Extract main content
            content = await page.evaluate('''() => {
                // Helper to clean text
                const cleanText = text => text.replace(/\\s+/g, ' ').trim();
                
                // Get main content area
                const mainSelectors = [
                    'main',
                    'article',
                    '[role="main"]',
                    '#content',
                    '.content',
                    '.main-content'
                ];
                
                let mainContent;
                for (const selector of mainSelectors) {
                    mainContent = document.querySelector(selector);
                    if (mainContent) break;
                }
                
                if (!mainContent) mainContent = document.body;
                
                // Remove unwanted elements
                const removeSelectors = [
                    'nav',
                    'header',
                    'footer',
                    '.nav',
                    '.menu',
                    '.sidebar',
                    '.widget',
                    'script',
                    'style',
                    'noscript'
                ];
                
                removeSelectors.forEach(selector => {
                    mainContent.querySelectorAll(selector).forEach(el => el.remove());
                });
                
                return {
                    title: document.title,
                    content: cleanText(mainContent.innerText),
                    url: window.location.href
                };
            }''')
            
            await page.close()
            return content
            
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def analyze_content(self, content: dict) -> dict:
        """Use GPT to analyze content and extract program information"""
        try:
            # Create OpenAI client with API key
            client = openai.OpenAI(api_key=openai.api_key)
            
            # Use GPT-4 for structured extraction
            response = client.chat.completions.create(
                model="gpt-4",
                temperature=0,  # Maximum precision
                messages=[
                    {"role": "system", "content": """
                        You are a medical education expert who extracts and structures 
                        information from medical school websites. Focus on extracting 
                        factual information and ignore marketing language. Format all
                        output as clean, structured JSON matching the provided schema.
                        
                        Important rules:
                        1. Only include information explicitly stated in the content
                        2. Use exact data types specified in schema
                        3. Leave fields empty (null) if information is not found
                        4. Do not make assumptions or inferences
                        5. Extract specific numbers and dates when available
                    """},
                    {"role": "user", "content": f"""
                        Analyze this medical school webpage content and extract program information
                        matching this exact schema:
                        
                        {json.dumps(PROGRAM_SCHEMA, indent=2)}
                        
                        URL: {content['url']}
                        Title: {content['title']}
                        Content: {content['content']}
                        
                        Return only factual information found in the content, formatted
                        exactly according to the schema. Leave fields empty (null) if
                        information is not found in the content.
                    """}
                ]
            )
            
            try:
                # Get and parse response
                if not response.choices or not response.choices[0].message:
                    logging.error("No response from GPT")
                    return {}
                    
                response_text = response.choices[0].message.content
                logging.debug(f"Raw GPT response: {response_text}")
                
                # Parse with error context
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON Error at position {e.pos}: {e.msg}")
                    logging.error(f"Context: {response_text[max(0, e.pos-50):min(len(response_text), e.pos+50)]}")
                    return {}
                
                # Validate against schema
                if not isinstance(result, dict):
                    raise ValueError("Response is not a dictionary")
                    
                # Ensure required top-level keys exist
                for key in PROGRAM_SCHEMA.keys():
                    if key not in result:
                        result[key] = {}
                
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Failed content: {response_text}")
                return {}
            except Exception as e:
                logging.error(f"Validation error: {str(e)}")
                return {}
            
        except Exception as e:
            logging.error(f"Error analyzing content: {str(e)}")
            return {}
    
    async def process_url_batch(self, urls: list, browser) -> list:
        """Process a batch of URLs concurrently"""
        # Get content for all URLs in parallel
        tasks = []
        for url_data in urls:
            url = url_data['url']
            logging.info(f"Fetching {url}")
            tasks.append(self.get_page_content(url, browser))
        
        contents = await asyncio.gather(*tasks)
        
        # Analyze content (rate limited to 500 RPM)
        results = []
        for content in contents:
            if content:
                result = self.analyze_content(content)
                if result:  # Only include non-empty results
                    results.append(result)
                # Small delay to stay under rate limit
                await asyncio.sleep(0.12)  # ~500 requests per minute
        
        return results

    async def process_urls(self, urls: list):
        """Process list of URLs and extract program information"""
        # Launch browser with more robust settings
        browser = await launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--no-zygote',
                '--single-process'
            ],
            handleSIGINT=False,
            handleSIGTERM=False,
            handleSIGHUP=False
        )
        
        try:
            # Process URLs in smaller batches
            program_data = {}
            batch_size = 3  # Process 3 pages at once to avoid connection issues
            
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}")
                
                # Process batch
                results = await self.process_url_batch(batch, browser)
                
                # Merge results
                for result in results:
                    self._deep_merge(program_data, result)
            
            self.program_data = program_data
        
        finally:
            await browser.close()
    
    def _deep_merge(self, target: dict, source: dict) -> None:
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if key in target:
                if isinstance(target[key], dict) and isinstance(value, dict):
                    self._deep_merge(target[key], value)
                elif isinstance(target[key], list) and isinstance(value, list):
                    # Deduplicate list items
                    target[key].extend(x for x in value if x not in target[key])
                else:
                    # Prefer non-empty values
                    if value and not target[key]:
                        target[key] = value
            else:
                target[key] = value

    def save_results(self):
        """Save extracted program information in taxonomy format"""
        output_dir = os.path.join('schools', self.school_name, 'processed_data')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f'program_data_{timestamp}.json')
        
        # Format according to taxonomy
        result = {
            "institutions": [
                {
                    "institution_name": self.school_name,
                    "programs": [self.program_data]
                }
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logging.info(f"Saved program data to {output_file}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python med_school_processor.py <school_name>")
        sys.exit(1)
        
    school_name = sys.argv[1]
    
    # Load URLs
    urls_file = os.path.join('schools', school_name, 'json', f'{school_name}_urls.json')
    with open(urls_file) as f:
        urls = json.load(f)['urls']
    
    # Process school
    processor = MedSchoolProcessor(school_name)
    await processor.process_urls(urls)
    processor.save_results()

if __name__ == "__main__":
    asyncio.run(main())

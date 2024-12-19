#!/usr/bin/env python3
"""
Demo of med_school_processor.py focusing on a single page
"""

import json
import asyncio
from pyppeteer import launch
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Load taxonomy schema
with open('taxonomy.json', 'r') as f:
    TAXONOMY = json.load(f)
PROGRAM_SCHEMA = TAXONOMY['institutions'][0]['programs'][0]

async def get_page_content(url: str) -> dict:
    """Get cleaned page content using Puppeteer"""
    browser = await launch(headless=True, args=['--no-sandbox'])
    try:
        page = await browser.newPage()
        await page.goto(url, {'waitUntil': 'networkidle0'})
        
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
        
        return content
        
    finally:
        await browser.close()

def analyze_content(content: dict) -> dict:
    """Use GPT to analyze content and extract program information"""
    try:
        # Focus on application requirements section of schema
        requirements_schema = PROGRAM_SCHEMA['application_requirements']
        
        # Create OpenAI client
        client = openai.Client()
        
        # Call GPT API with focused prompt
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                    You are a medical education expert who extracts and structures 
                    information from medical school websites. Focus on extracting 
                    factual information and ignore marketing language.
                    
                    Important rules:
                    1. Only include information explicitly stated in the content
                    2. Use exact data types specified in schema
                    3. Leave fields null if information is not found
                    4. Do not make assumptions or inferences
                    5. Extract specific numbers and dates when available
                    6. Pay special attention to:
                       - Required vs recommended courses
                       - Credit hour requirements
                       - Lab requirements
                       - Course format restrictions
                       - GPA requirements
                       - MCAT requirements
                """},
                {"role": "user", "content": f"""
                    Analyze this medical school requirements page and extract information
                    matching this schema:
                    
                    {json.dumps(requirements_schema, indent=2)}
                    
                    URL: {content['url']}
                    Title: {content['title']}
                    Content: {content['content']}
                    
                    Return only factual information found in the content, formatted
                    exactly according to the schema. Leave fields empty (null) if
                    information is not found in the content.
                """}
            ]
        )
        
        # Parse and validate response
        result = json.loads(response.choices[0].message.content)
        
        # Show the mapping process
        print("\nExtracted Information:")
        print("=====================")
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        print(f"Error analyzing content: {str(e)}")
        return {}

async def main():
    # URL to process
    url = "https://www.med.unc.edu/admit/requirements/academic-requirements/"
    
    print(f"\nFetching content from: {url}")
    content = await get_page_content(url)
    
    print("\nRaw Content Sample:")
    print("==================")
    print(content['content'][:500] + "...")
    
    print("\nAnalyzing content with GPT-4...")
    result = analyze_content(content)
    
    # Save result
    with open('demo_output.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nFull results saved to demo_output.json")

if __name__ == "__main__":
    asyncio.run(main())

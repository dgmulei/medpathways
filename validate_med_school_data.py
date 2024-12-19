#!/usr/bin/env python3
"""
Medical School Data Validation Script
Validates structured JSON data against source website content
"""

import json
import requests
from bs4 import BeautifulSoup
import datetime
import difflib
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
from urllib.parse import urljoin
import hashlib
import os
import sys
import time

@dataclass
class ValidationResult:
    field_path: str
    is_valid: bool
    source_url: str
    source_text: str
    validation_date: datetime.datetime
    notes: str

class MedSchoolDataValidator:
    def __init__(self, json_path: str, urls_path: str, taxonomy_path: str, school_name: str):
        """Initialize validator with paths to required files"""
        self.json_path = json_path
        self.urls_path = urls_path
        self.taxonomy_path = taxonomy_path
        self.school_name = school_name
        self.validation_results: List[ValidationResult] = []
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        school_validation_dir = f"schools/{self.school_name}/validation"
        os.makedirs(school_validation_dir, exist_ok=True)
        
        log_path = os.path.join(
            school_validation_dir,
            f'validation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_json_data(self) -> Tuple[Dict, Dict, List[str]]:
        """Load JSON data and URLs"""
        try:
            with open(self.json_path, 'r') as f:
                program_data = json.load(f)
            with open(self.taxonomy_path, 'r') as f:
                taxonomy = json.load(f)
            with open(self.urls_path, 'r') as f:
                urls = [line.strip() for line in f.readlines()]
            return program_data, taxonomy, urls
        except Exception as e:
            self.logger.error(f"Error loading data files: {str(e)}")
            raise

    def validate_urls(self, urls: List[str]) -> Dict[str, bool]:
        """Validate URLs are accessible with retries and longer timeout"""
        url_status = {}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        
        max_retries = 3
        timeout = 30
        
        for url in urls:
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                    url_status[url] = response.status_code == 200
                    if url_status[url]:
                        break
                    else:
                        self.logger.warning(f"URL returned status {response.status_code}: {url}")
                        time.sleep(2)  # Wait before retry
                except Exception as e:
                    self.logger.error(f"Error accessing URL {url} (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        url_status[url] = False
                    time.sleep(2)  # Wait before retry
        
        return url_status

    def fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and parse content from URL with improved cleaning"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            # Normalize whitespace
            text = ' '.join(text.split())
            # Remove special characters
            text = re.sub(r'[^\w\s-]', ' ', text)
            # Normalize spaces
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {str(e)}")
            return None

    def validate_field(self, field_path: str, field_value: any, content: str, url: str) -> ValidationResult:
        """Validate a single field against source content with improved matching"""
        if isinstance(field_value, str) and field_value != "Not published":
            # Clean and normalize the field value and content
            clean_value = ' '.join(field_value.lower().split())
            clean_content = ' '.join(content.lower().split())
            
            # Try exact match first
            search_pattern = re.escape(clean_value)
            match = re.search(search_pattern, clean_content)
            
            if not match:
                # Try fuzzy matching if exact match fails
                words = clean_value.split()
                # Consider it a match if 75% of words are found
                matches = sum(1 for word in words if word in clean_content)
                is_valid = matches >= len(words) * 0.75
                source_text = clean_value if is_valid else ""
                notes = f"Fuzzy match ({matches}/{len(words)} words)" if is_valid else "No match found"
            else:
                is_valid = True
                source_text = match.group(0)
                notes = "Exact match found"
        else:
            is_valid = True  # Can't validate non-string or "Not published" values
            source_text = ""
            notes = "Non-string or 'Not published' value"

        result = ValidationResult(
            field_path=field_path,
            is_valid=is_valid,
            source_url=url,
            source_text=source_text,
            validation_date=datetime.datetime.now(),
            notes=notes
        )
        self.validation_results.append(result)
        return result

    def validate_dict(self, data: Dict, content: str, url: str, path: str = "") -> None:
        """Recursively validate dictionary fields"""
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                self.validate_dict(value, content, url, current_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self.validate_dict(item, content, url, f"{current_path}[{i}]")
                    else:
                        self.validate_field(f"{current_path}[{i}]", item, content, url)
            else:
                self.validate_field(current_path, value, content, url)

    def validate_against_taxonomy(self, program_data: Dict, taxonomy: Dict) -> List[str]:
        """Validate program data structure against taxonomy"""
        def check_structure(data: Dict, schema: Dict, path: str = "") -> List[str]:
            issues = []
            
            for key, value in schema.items():
                new_path = f"{path}.{key}" if path else key
                
                if key not in data:
                    issues.append(f"Missing required field: {new_path}")
                    continue
                
                if isinstance(value, dict):
                    if not isinstance(data[key], dict):
                        issues.append(f"Type mismatch at {new_path}: expected dict")
                        continue
                    issues.extend(check_structure(data[key], value, new_path))
                
                # Add additional type checking as needed
                
            return issues
        
        return check_structure(program_data, taxonomy)

    def generate_validation_report(self) -> str:
        """Generate detailed validation report"""
        report = []
        report.append(f"Medical School Data Validation Report - {self.school_name}")
        report.append(f"Generated: {datetime.datetime.now().isoformat()}")
        report.append("-" * 80)
        
        # Summary statistics
        total_fields = len(self.validation_results)
        if total_fields > 0:
            valid_fields = sum(1 for r in self.validation_results if r.is_valid)
            validation_rate = (valid_fields/total_fields)*100
            
            report.append(f"Total fields validated: {total_fields}")
            report.append(f"Valid fields: {valid_fields}")
            report.append(f"Invalid fields: {total_fields - valid_fields}")
            report.append(f"Validation rate: {validation_rate:.2f}%")
            
            # Add match type breakdown
            exact_matches = sum(1 for r in self.validation_results if "Exact match" in r.notes)
            fuzzy_matches = sum(1 for r in self.validation_results if "Fuzzy match" in r.notes)
            not_published = sum(1 for r in self.validation_results if "Not published" in r.notes)
            
            report.append(f"\nMatch Type Breakdown:")
            report.append(f"Exact matches: {exact_matches}")
            report.append(f"Fuzzy matches: {fuzzy_matches}")
            report.append(f"Not published fields: {not_published}")
        else:
            report.append("No fields were validated")
        report.append("-" * 80)
        
        # Detailed results
        report.append("Detailed Results:")
        for result in self.validation_results:
            report.append(f"\nField: {result.field_path}")
            report.append(f"Valid: {result.is_valid}")
            report.append(f"Source URL: {result.source_url}")
            if result.source_text:
                report.append(f"Source text: {result.source_text[:100]}...")
            report.append(f"Notes: {result.notes}")
        
        return "\n".join(report)

    def run_validation(self):
        """Run complete validation process"""
        try:
            # Load data
            print("Loading data files...")
            self.logger.info("Loading data files...")
            program_data, taxonomy, urls = self.load_json_data()
            print(f"Loaded {len(urls)} URLs to validate")
            
            # Validate URLs
            print("Validating URLs...")
            self.logger.info("Validating URLs...")
            url_status = self.validate_urls(urls)
            accessible_urls = sum(1 for status in url_status.values() if status)
            print(f"Found {accessible_urls} accessible URLs out of {len(urls)}")
            
            # Validate against taxonomy
            print("Validating against taxonomy...")
            self.logger.info("Validating against taxonomy...")
            taxonomy_issues = self.validate_against_taxonomy(program_data, taxonomy)
            if taxonomy_issues:
                print("Found taxonomy issues:")
                for issue in taxonomy_issues:
                    print(f"- {issue}")
            else:
                print("No taxonomy issues found")
            
            # Content validation
            print("Validating content...")
            self.logger.info("Validating content...")
            for url in urls:
                if url_status[url]:
                    print(f"Processing URL: {url}")
                    content = self.fetch_page_content(url)
                    if content:
                        print(f"Validating fields against content from {url}")
                        self.validate_dict(program_data, content, url)
            
            # Generate report
            print("Generating validation report...")
            self.logger.info("Generating validation report...")
            report = self.generate_validation_report()
            
            # Save report
            validation_dir = f"schools/{self.school_name}/validation"
            os.makedirs(validation_dir, exist_ok=True)
            report_filename = os.path.join(
                validation_dir,
                f'validation_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
            )
            
            with open(report_filename, 'w') as f:
                f.write(report)
            
            print(f"Validation complete. Report saved to {report_filename}")
            self.logger.info(f"Validation complete. Report saved to {report_filename}")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_med_school_data.py <school_name>")
        print("Example: python validate_med_school_data.py mt_sinai")
        sys.exit(1)
        
    school_name = sys.argv[1]
    school_dir = f"schools/{school_name}"
    
    if not os.path.exists(school_dir):
        print(f"Error: School directory {school_dir} not found")
        sys.exit(1)
    
    json_path = os.path.join(school_dir, "json", f"{school_name}_md_program.json")
    urls_path = os.path.join(school_dir, "json", f"{school_name}_urls")
    taxonomy_path = "taxonomy.json"  # Shared schema file stays in root
    
    if not all(os.path.exists(p) for p in [json_path, urls_path, taxonomy_path]):
        print("Error: Required files not found")
        print(f"Checking for:")
        print(f"- {json_path}")
        print(f"- {urls_path}")
        print(f"- {taxonomy_path}")
        sys.exit(1)
    
    validator = MedSchoolDataValidator(
        json_path=json_path,
        urls_path=urls_path,
        taxonomy_path=taxonomy_path,
        school_name=school_name
    )
    validator.run_validation()

if __name__ == "__main__":
    main()

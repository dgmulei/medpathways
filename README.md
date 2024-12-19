# MedPathways

An intelligent system for extracting and structuring medical school program information using AI.

## Project Overview

MedPathways uses a three-stage pipeline to collect, analyze, and structure medical school program information:

1. **Intelligent Crawler** (`1. intelligent_crawler.py`)
   - Smart web crawler that identifies relevant pages on medical school websites
   - Uses heuristics to find important content (admissions, curriculum, etc.)
   - Handles pagination and navigation patterns
   - Outputs: `schools/{school}/crawler_results/relevant_urls_{timestamp}.json`

2. **URL Analyzer** (`2. url_analyzer.py`)
   - Analyzes discovered URLs to identify content categories
   - Maps URLs to specific information types (requirements, curriculum, etc.)
   - Filters out non-relevant pages
   - Outputs: `schools/{school}/json/{school}_urls.json`

3. **Content Processor** (`3. med_school_processor.py`)
   - Uses GPT-4o-mini for high-throughput content extraction
   - Maps content to standardized taxonomy schema
   - Handles parallel processing and rate limits
   - Outputs: `schools/{school}/processed_data/program_data_{timestamp}.json`

## Pipeline Flow

The system uses a staged approach where each component builds on the previous:

1. **Intelligent Crawler → URL Discovery**
   - Starts with school's main MD program page
   - Uses intelligent navigation to find relevant content
   - Identifies important pages through link analysis
   - Saves discovered URLs for further processing
   - Example output:
     ```json
     {
       "urls": [
         {"url": "https://med.school.edu/admissions", "depth": 1},
         {"url": "https://med.school.edu/curriculum", "depth": 2}
       ]
     }
     ```

2. **URL Analyzer → Content Categorization**
   - Takes crawler output as input
   - Analyzes URL patterns and page content
   - Maps pages to taxonomy categories
   - Filters irrelevant or duplicate content
   - Example output:
     ```json
     {
       "urls": [
         {
           "url": "https://med.school.edu/admissions",
           "category": "admissions",
           "relevance_score": 0.95
         }
       ]
     }
     ```

3. **Content Processor → Structured Data**
   - Uses analyzed URLs as input
   - Extracts content using GPT-4o-mini
   - Maps to standardized taxonomy
   - Merges related information
   - Example output:
     ```json
     {
       "institutions": [{
         "programs": [{
           "admissions": {
             "requirements": {...},
             "deadlines": {...}
           }
         }]
       }]
     }
     ```

Each stage preserves data in a structured format that serves as input for the next stage, creating a robust pipeline for processing any medical school's program information.

## Usage

1. Run the crawler for a school:
```bash
python "1. intelligent_crawler.py" <school_name> <start_url>
```

2. Analyze discovered URLs:
```bash
python "2. url_analyzer.py" <school_name>
```

3. Process content into structured data:
```bash
python "3. med_school_processor.py" <school_name>
```

## Output Structure

The system follows a standardized taxonomy (defined in `taxonomy.json`) for consistent data structure across all schools:

```json
{
  "institutions": [{
    "institution_name": "string",
    "programs": [{
      "program_name": "string",
      "application_requirements": {...},
      "program_curriculum": {...},
      ...
    }]
  }]
}
```

## Getting Started

1. **Setup Environment**
   ```bash
   # Clone repository
   git clone https://github.com/yourusername/MedPathways.git
   cd MedPathways

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Configure OpenAI**
   - Get API key from OpenAI dashboard
   - Create `.env` file:
     ```
     OPENAI_API_KEY=your-key-here
     ```

3. **Initial Test**
   ```bash
   # Test with UNC School of Medicine
   python "1. intelligent_crawler.py" unc "https://www.med.unc.edu/md/"
   python "2. url_analyzer.py" unc
   python "3. med_school_processor.py" unc
   ```

## Dependencies

- Python 3.8+
- OpenAI API key (for GPT-4o-mini)
- Chrome/Chromium for browser automation
- See `requirements.txt` for full list

## Project Structure

```
MedPathways/
├── 1. intelligent_crawler.py   # Smart web crawler
├── 2. url_analyzer.py         # URL categorization
├── 3. med_school_processor.py # Content extraction
├── taxonomy.json             # Data schema
├── requirements.txt          # Dependencies
└── schools/                  # School-specific data
    └── {school_name}/
        ├── crawler_results/  # Raw crawler output
        ├── json/            # Analyzed URLs
        └── processed_data/  # Final structured data
```

## Performance

- Crawler: Intelligent discovery of relevant pages
- Analyzer: Fast URL categorization
- Processor: 
  - Uses GPT-4o-mini (200K TPM, 500 RPM)
  - Parallel processing with rate limiting
  - Handles 3 pages concurrently

## AI Components

The system uses increasingly sophisticated AI at each stage:

1. **Crawler Intelligence**
   - Link relevance scoring
   - Navigation path optimization
   - Content importance evaluation
   - Duplicate detection

2. **URL Analysis**
   - Pattern recognition
   - Content categorization
   - Relevance scoring
   - Semantic grouping

3. **Content Processing**
   - GPT-4o-mini for extraction (200K TPM)
   - Schema-guided parsing
   - Information merging
   - Validation against taxonomy

## Error Handling

- Robust browser management
- Content type validation
- JSON response verification
- Detailed logging
- Graceful failure recovery

## Pipeline Validation

Each stage includes validation to ensure data quality:

1. **Crawler → Analyzer**
   - Validates URL formats and accessibility
   - Checks for required page types (admissions, curriculum)
   - Ensures minimum content coverage
   - Logs missing or problematic sections

2. **Analyzer → Processor**
   - Validates URL categorization
   - Checks relevance scores meet thresholds
   - Ensures complete category coverage
   - Identifies content gaps

3. **Processor → Output**
   - Validates against taxonomy schema
   - Checks data completeness
   - Ensures type consistency
   - Verifies required fields

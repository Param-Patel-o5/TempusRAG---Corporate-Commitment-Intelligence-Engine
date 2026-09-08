"""SEC EDGAR fetch, format detection, and filing parsing - REWRITTEN."""

import logging
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Only import fitz if we actually need it for PDFs
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "./data/filings")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "TempusRAG/1.0 contact@example.com")
RATE_LIMIT_DELAY = 0.11  # SEC allows ~10 req/sec, stay under

# URLs
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# Global cache
_ticker_cik_cache = None
_last_request_time = 0.0


class EdgarFetchError(Exception):
    """Raised when SEC EDGAR returns a non-200 response or is unreachable."""
    pass


def _make_request(url: str) -> requests.Response:
    """Make a rate-limited request to SEC EDGAR with proper error handling."""
    global _last_request_time
    
    # Rate limiting
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html, */*"
    }
    
    try:
        logger.debug(f"Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        _last_request_time = time.time()
        
        if response.status_code != 200:
            logger.error(f"HTTP {response.status_code} for {url}")
            raise EdgarFetchError(f"HTTP {response.status_code}: {url}")
            
        return response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {e}")
        raise EdgarFetchError(f"Request failed: {e}") from e


def load_ticker_cik_map() -> dict[str, str]:
    """Load ticker->CIK mapping from SEC, with caching."""
    global _ticker_cik_cache
    
    if _ticker_cik_cache is not None:
        return _ticker_cik_cache
    
    logger.info("Fetching SEC ticker-to-CIK mapping")
    response = _make_request(COMPANY_TICKERS_URL)
    data = response.json()
    
    # Convert to ticker -> zero-padded CIK mapping
    ticker_map = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        ticker_map[ticker] = cik
    
    logger.info(f"Loaded {len(ticker_map)} ticker-to-CIK mappings")
    _ticker_cik_cache = ticker_map
    return ticker_map


def get_recent_10k_filings(ticker: str, cik_map: dict[str, str], count: int = 5) -> list[dict]:
    """Get recent 10-K filings for a ticker."""
    ticker = ticker.upper()
    cik = cik_map.get(ticker)
    if not cik:
        raise ValueError(f"Ticker not found: {ticker}")
    
    logger.info(f"Fetching recent 10-K filings for {ticker} (CIK: {cik})")
    
    # Get submissions
    url = SUBMISSIONS_URL_TEMPLATE.format(cik=cik)
    response = _make_request(url)
    submissions = response.json()
    
    # Filter for 10-K forms
    recent = submissions["filings"]["recent"]
    filings = []
    
    for i, form in enumerate(recent["form"]):
        if form != "10-K":
            continue
            
        accession = recent["accessionNumber"][i]
        filing_date = recent["filingDate"][i]
        primary_document = recent["primaryDocument"][i]
        
        # Build document URL
        cik_no_leading_zeros = str(int(cik))
        accession_no_dashes = accession.replace("-", "")
        document_url = ARCHIVES_URL_TEMPLATE.format(
            cik=cik_no_leading_zeros,
            accession=accession_no_dashes,
            document=primary_document
        )
        
        filings.append({
            "accessionNumber": accession,
            "filingDate": filing_date,
            "primaryDocument": primary_document,
            "document_url": document_url
        })
    
    # Sort by date and return most recent
    filings.sort(key=lambda x: x["filingDate"], reverse=True)
    result = filings[:count]
    
    logger.info(f"Found {len(filings)} total 10-K filings, returning {len(result)} most recent")
    return result


def download_filing(filing: dict) -> bytes:
    """Download raw filing content."""
    url = filing["document_url"]
    date = filing.get("filingDate", "unknown")
    doc = filing.get("primaryDocument", "unknown")
    
    logger.info(f"Downloading 10-K filing: {date} ({doc})")
    
    response = _make_request(url)
    content = response.content
    
    logger.debug(f"Downloaded {len(content)} bytes")
    return content


def detect_format(content: bytes, filename: str = "") -> str:
    """Detect if content is HTML, PDF, or plain text."""
    # Check filename extension first
    filename_lower = filename.lower()
    if filename_lower.endswith((".pdf",)):
        return "pdf"
    elif filename_lower.endswith((".htm", ".html")):
        return "html"
    
    # Check content
    if content.startswith(b"%PDF"):
        return "pdf"
    
    # Check for HTML markers in first 1KB
    sample = content[:1024].lower()
    if b"<html" in sample or b"<!doctype" in sample or b"<div" in sample:
        return "html"
    
    return "text"


def parse_html_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Parse HTML filing content into sections."""
    soup = BeautifulSoup(content, "html.parser")
    
    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    # Try to find the main content area
    # Many SEC filings wrap content in specific divs or tables
    main_content = None
    
    # Look for common SEC filing containers
    for selector in [
        "div[style*='margin-left']",  # Common in SEC filings
        "div.DocumentBody",
        "div#document",
        "table[width='100%']",
        "body"
    ]:
        elements = soup.select(selector)
        if elements:
            main_content = elements[0]
            logger.debug(f"Found main content using selector: {selector}")
            break
    
    if main_content is None:
        main_content = soup
    
    # Extract all text in a simple way first
    full_text = main_content.get_text(separator="\n", strip=True)
    
    if not full_text or len(full_text) < 100:
        logger.warning("Very little text extracted from HTML filing")
        return {"Document": {"full_text": full_text}}
    
    # Try to find sections using text patterns
    sections = _extract_sections_from_text(full_text)
    
    if not sections or len(sections) == 1:
        # Fallback: return all text as one document
        return {"Document": {"full_text": full_text}}
    
    return sections


def parse_pdf_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Parse PDF filing content."""
    if not FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF not available for PDF parsing")
    
    doc = fitz.open(stream=content, filetype="pdf")
    
    # Extract text from all pages
    text_parts = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(text)
    
    doc.close()
    
    full_text = "\n\n".join(text_parts)
    logger.debug(f"Extracted {len(full_text)} chars from {len(text_parts)} PDF pages")
    
    # Try to extract sections
    sections = _extract_sections_from_text(full_text)
    
    if not sections:
        return {"Document": {"full_text": full_text}}
    
    return sections


def parse_text_filing(content: bytes) -> dict[str, dict[str, str]]:
    """Parse plain text filing."""
    try:
        text = content.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        text = content.decode("latin1", errors="replace")
    
    sections = _extract_sections_from_text(text)
    
    if not sections:
        return {"Document": {"full_text": text}}
    
    return sections


def _extract_sections_from_text(text: str) -> dict[str, dict[str, str]]:
    """Extract sections from text using pattern matching."""
    
    # Clean up text first
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    
    # Patterns to look for section headings
    section_patterns = [
        # Standard 10-K sections
        r"^\s*ITEM\s+(\d+[A-Z]?)\.\s*(.+?)$",
        r"^\s*Item\s+(\d+[A-Z]?)\.\s*(.+?)$", 
        r"^\s*PART\s+([IVX]+)\s*(.*)$",
        r"^\s*Part\s+([IVX]+)\s*(.*)$",
        # All caps headings
        r"^([A-Z][A-Z\s,&'\-\.\(\)\/]{10,})$",
        # Title case headings
        r"^([A-Z][A-Za-z\s,&'\-\.\(\)\/]{10,})$"
    ]
    
    sections = {}
    current_section = None
    current_text = []
    
    lines = text.split("\n")
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            current_text.append("")
            continue
        
        # Check if this line looks like a section header
        is_header = False
        section_name = None
        
        for pattern in section_patterns:
            match = re.match(pattern, line_stripped, re.MULTILINE)
            if match:
                if len(match.groups()) == 2:
                    # Item pattern with number and title
                    section_name = f"Item {match.group(1)} - {match.group(2).strip()}"
                elif len(match.groups()) == 1:
                    section_name = match.group(1).strip()
                else:
                    section_name = line_stripped
                
                is_header = True
                break
        
        # Additional heuristics for headers
        if not is_header and len(line_stripped) > 10 and len(line_stripped) < 100:
            # Check if it's likely a header based on context
            if (line_stripped.isupper() or 
                line_stripped.count(" ") < 10 and any(word in line_stripped.upper() for word in 
                ["MANAGEMENT", "DISCUSSION", "ANALYSIS", "BUSINESS", "RISK", "FINANCIAL", "OPERATIONS"])):
                is_header = True
                section_name = line_stripped
        
        if is_header and section_name:
            # Save previous section
            if current_section and current_text:
                content = "\n".join(current_text).strip()
                if content:
                    sections[current_section] = {"full_text": content}
            
            # Start new section
            current_section = section_name
            current_text = []
        else:
            # Add to current section
            current_text.append(line)
    
    # Save last section
    if current_section and current_text:
        content = "\n".join(current_text).strip()
        if content:
            sections[current_section] = {"full_text": content}
    
    logger.debug(f"Extracted {len(sections)} sections from text")
    return sections


def _has_mda_section(sections: dict) -> bool:
    """Check if MD&A section exists."""
    for section_name in sections.keys():
        name_upper = section_name.upper()
        if ("MD&A" in name_upper or 
            "MDA" in name_upper or
            ("MANAGEMENT" in name_upper and "DISCUSSION" in name_upper)):
            return True
    return False


def ingest_company(ticker: str, cik_map: dict[str, str], data_dir: str) -> dict[int, dict]:
    """Ingest filings for a company."""
    ticker = ticker.upper()
    
    # Get recent filings
    filings = get_recent_10k_filings(ticker, cik_map, count=5)
    
    # Create output directory
    output_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for filing in filings:
        filing_date = filing["filingDate"]
        year = int(filing_date[:4])
        
        try:
            # Download content
            content = download_filing(filing)
            
            # Detect format
            fmt = detect_format(content, filing["primaryDocument"])
            logger.info(f"Detected format '{fmt}' for {ticker} {year}")
            
            # Parse based on format
            if fmt == "html":
                sections = parse_html_filing(content)
            elif fmt == "pdf":
                sections = parse_pdf_filing(content)
            else:
                sections = parse_text_filing(content)
            
            # Check for MD&A
            if not _has_mda_section(sections):
                logger.warning(f"No MD&A section found for {ticker} {year}")
            
            # Save to cache file
            plain_text_parts = []
            for section_name, subsections in sections.items():
                plain_text_parts.append(f"=== {section_name} ===")
                for sub_name, text in subsections.items():
                    if sub_name != "full_text":
                        plain_text_parts.append(f"--- {sub_name} ---")
                    plain_text_parts.append(text)
            
            cache_file = output_dir / f"{ticker}_{year}.txt"
            cache_file.write_text("\n\n".join(plain_text_parts), encoding="utf-8")
            
            logger.info(f"Successfully processed {ticker} {year} -> {cache_file}")
            results[year] = sections
            
        except Exception as e:
            logger.error(f"Failed to process {ticker} {year}: {e}")
            continue
        
        # Rate limiting between filings
        time.sleep(RATE_LIMIT_DELAY)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        # Load ticker mapping
        cik_map = load_ticker_cik_map()
        
        # Process HIG
        results = ingest_company("HIG", cik_map, "./data/filings")
        
        # Print analysis
        print(f"\n=== Analysis for HIG ===")
        print(f"Processed {len(results)} years")
        
        for year, sections in results.items():
            print(f"\nYear {year}:")
            print(f"  Sections: {len(sections)}")
            
            for section_name, subsections in sections.items():
                subsection_count = len(subsections)
                print(f"    '{section_name}': {subsection_count} subsection(s)")
                
                # Show preview of first subsection
                first_subsection = next(iter(subsections.values()))
                preview = first_subsection[:200].replace('\n', ' ')
                print(f"      Preview: {preview}...")
                
                # Check for MD&A-like content
                if any(keyword in section_name.upper() for keyword in ["MD&A", "MDA", "MANAGEMENT", "DISCUSSION"]):
                    print(f"      *** POTENTIAL MD&A SECTION ***")
    
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise
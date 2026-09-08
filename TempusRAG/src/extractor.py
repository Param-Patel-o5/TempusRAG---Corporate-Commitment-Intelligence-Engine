"""Promise extraction from MD&A and Future Outlook sections using Gemini 1.5 Flash.

This is LLM Call 1 in the TempusRAG architecture. Takes filtered chunks from promise-extraction
sections (MD&A, Future Outlook, Growth Strategy) and extracts structured Promise objects.

API Key handling: This module reads GEMINI_API_KEY from config.py at call time, not import time.
The Streamlit UI may set API keys at runtime, so extract_promises_for_year() accepts an optional
api_key parameter that overrides the config value if provided.
"""

import json
import logging

import google.generativeai as genai

from src.config import GEMINI_API_KEY, EXTRACTOR_MODEL, PROMISE_EXTRACTION_SECTIONS
from src.models import Promise

logger = logging.getLogger(__name__)

# Configure Gemini client at module level if API key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini client configured with API key from config")
else:
    logger.warning(
        "No GEMINI_API_KEY found in config - API key must be provided at runtime "
        "via extract_promises_for_year(api_key=...) parameter"
    )


def filter_promise_chunks(chunks_by_year: dict[int, list], sections: list[str] = PROMISE_EXTRACTION_SECTIONS) -> dict[int, list]:
    """
    Filter chunks to only include promise-extraction-relevant sections.
    
    Args:
        chunks_by_year: Dictionary mapping year to list of Chunk objects.
        sections: List of section names to include (defaults to PROMISE_EXTRACTION_SECTIONS).
        
    Returns:
        dict[int, list]: Same structure as input, but with chunks filtered to only relevant sections.
    """
    filtered_chunks = {}
    
    for year, chunks in chunks_by_year.items():
        filtered_chunks[year] = [
            chunk for chunk in chunks 
            if chunk.section in sections
        ]
        
        logger.debug(
            f"Year {year}: filtered {len(chunks)} chunks down to {len(filtered_chunks[year])} "
            f"promise-extraction chunks (sections: {sections})"
        )
    
    return filtered_chunks


def build_extraction_prompt(year: int, chunks: list) -> str:
    """
    Build prompt for extracting forward-looking commitments from chunks.
    
    Args:
        year: Fiscal year context for the extraction.
        chunks: List of Chunk objects to extract promises from.
        
    Returns:
        str: Complete prompt for the LLM.
    """
    # Concatenate all chunk texts (they already have contextual prefixes)
    chunk_texts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_texts.append(f"--- Chunk {i} ---\n{chunk.text}")
    
    combined_text = "\n\n".join(chunk_texts)
    
    prompt = f"""You are analyzing SEC filing content from fiscal year {year} to extract forward-looking management commitments and promises.

TASK: Extract every distinct forward-looking commitment, promise, or target mentioned by management in the following text. Focus on specific, measurable commitments about future performance, initiatives, or goals.

INSTRUCTIONS:
1. Extract each forward-looking commitment as a separate promise
2. For each promise, determine the domain (must be exactly one of: Technology, Claims, Growth, Finance, Operations)
3. If a specific deadline is mentioned in the text, capture it verbatim; otherwise use null
4. The year_made is {year} (the fiscal year of this filing)
5. Set page_number to 0 (page numbers are not available from chunk text)

RESPONSE FORMAT: Respond with ONLY a JSON array matching this exact structure, no markdown fences, no preamble:
[{{"promise_text": "exact text of the commitment", "domain": "Technology|Claims|Growth|Finance|Operations", "deadline_mentioned": "exact deadline text or null", "year_made": {year}, "page_number": 0}}]

If no forward-looking commitments are found, respond with an empty array: []

CONTENT TO ANALYZE:
{combined_text}"""

    return prompt


def _call_gemini(prompt: str, api_key: str) -> str:
    """
    Low-level Gemini API call.
    
    Args:
        prompt: The prompt to send to Gemini.
        api_key: API key for authentication.
        
    Returns:
        str: Raw response text from Gemini.
        
    Raises:
        Any network or API exceptions (propagated to caller for retry logic).
    """
    # Reconfigure client with the provided API key (supports per-user keys from Streamlit)
    genai.configure(api_key=api_key)
    
    # Create model and generate content
    model = genai.GenerativeModel(EXTRACTOR_MODEL)
    response = model.generate_content(prompt)
    
    return response.text


def _parse_promise_json(raw_text: str) -> list[dict]:
    """
    Parse JSON response from Gemini, handling markdown code fences.
    
    Args:
        raw_text: Raw response text from Gemini.
        
    Returns:
        list[dict]: Parsed JSON data.
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails (propagated to caller).
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    
    # Remove ```json ... ``` or ``` ... ``` fences
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]   # Remove ```
        
    if text.endswith("```"):
        text = text[:-3]  # Remove closing ```
        
    text = text.strip()
    
    # Parse JSON
    return json.loads(text)


def extract_promises_for_year(year: int, chunks: list, api_key: str | None = None) -> list[Promise]:
    """
    Extract promises from chunks for a single year using Gemini.
    
    Args:
        year: Fiscal year for context.
        chunks: List of Chunk objects to extract from.
        api_key: Optional API key override (uses config if None).
        
    Returns:
        list[Promise]: Extracted and validated Promise objects.
        
    Raises:
        ValueError: If no API key is available.
    """
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Gemini API key available - set GEMINI_API_KEY in config or pass one explicitly"
        )
    
    # Skip if no chunks for this year
    if not chunks:
        logger.warning(f"No chunks provided for year {year}, skipping promise extraction")
        return []
    
    logger.info(f"Extracting promises from {len(chunks)} chunks for year {year}")
    
    # Build prompt
    prompt = build_extraction_prompt(year, chunks)
    
    # First attempt
    try:
        raw_response = _call_gemini(prompt, resolved_api_key)
        promise_dicts = _parse_promise_json(raw_response)
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed for year {year} on first attempt: {e}")
        
        # Retry with stricter prompt
        retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Respond with ONLY the JSON array, nothing else."
        
        try:
            raw_response = _call_gemini(retry_prompt, resolved_api_key)
            promise_dicts = _parse_promise_json(raw_response)
            logger.info(f"Retry successful for year {year}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed for year {year} after retry: {e}")
            logger.error(f"Raw response: {raw_response[:500]}...")
            return []  # Fail gracefully
    
    # Convert dicts to Promise objects
    promises = []
    for i, item in enumerate(promise_dicts):
        try:
            # Force year_made to the known year parameter (ground truth)
            item["year_made"] = year
            
            # Create Promise object
            promise = Promise(**item)
            promises.append(promise)
            
        except Exception as e:
            logger.warning(f"Failed to create Promise object {i} for year {year}: {e}")
            logger.warning(f"Offending dict: {item}")
            # Skip this malformed entry, continue with others
    
    logger.info(f"Successfully extracted {len(promises)} promises for year {year}")
    return promises


def extract_all_promises(chunks_by_year: dict[int, list], api_key: str | None = None) -> dict[int, list[Promise]]:
    """
    Extract promises from all years of chunks.
    
    Args:
        chunks_by_year: Dictionary mapping year to list of Chunk objects.
        api_key: Optional API key override (uses config if None).
        
    Returns:
        dict[int, list[Promise]]: Dictionary mapping year to extracted Promise objects.
    """
    logger.info(f"Starting promise extraction for {len(chunks_by_year)} years")
    
    # Filter chunks to promise-extraction sections only
    filtered_chunks = filter_promise_chunks(chunks_by_year)
    
    # Extract promises for each year
    results = {}
    total_promises = 0
    
    for year in sorted(filtered_chunks.keys()):
        chunks = filtered_chunks[year]
        
        try:
            promises = extract_promises_for_year(year, chunks, api_key)
            results[year] = promises
            total_promises += len(promises)
            
        except Exception as e:
            logger.error(f"Promise extraction failed for year {year}: {e}")
            results[year] = []  # Fail gracefully, continue with other years
    
    logger.info(f"Promise extraction complete: {total_promises} total promises across all years")
    return results


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    from src.models import Chunk

    # Sample chunks for testing
    sample_chunks = {
        2021: [
            Chunk(
                text="[Hartford | 2021 | MD&A > Technology] We are committed to reducing claims processing time by 40% by the end of 2023 through our new digital platform.",
                company="HIG", 
                year=2021, 
                section="MD&A", 
                subsection="Technology",
                chunk_id="HIG_2021_MDA_TECH_00", 
                word_count=25
            ),
            Chunk(
                text="[Hartford | 2021 | Future Outlook] We plan to expand our market share in the small business segment by 15% over the next two years.",
                company="HIG",
                year=2021,
                section="Future Outlook",
                subsection="full_text",
                chunk_id="HIG_2021_FUTURE_00",
                word_count=20
            )
        ]
    }
    
    print("Testing promise extraction...")
    print(f"Sample chunks: {sum(len(chunks) for chunks in sample_chunks.values())} total")
    
    # Test filtering
    filtered = filter_promise_chunks(sample_chunks)
    print(f"Filtered chunks: {sum(len(chunks) for chunks in filtered.values())} in promise-extraction sections")
    
    # Test prompt building
    if filtered[2021]:
        prompt = build_extraction_prompt(2021, filtered[2021])
        print(f"\nSample prompt length: {len(prompt)} characters")
        print(f"Prompt preview: {prompt[:300]}...")
    
    # Test full extraction (requires GEMINI_API_KEY to be set in .env or passed explicitly)
    try:
        results = extract_all_promises(sample_chunks, api_key=None)  # Will use config value
        
        print(f"\nExtraction results:")
        for year, promises in results.items():
            print(f"{year}: {len(promises)} promises")
            for p in promises:
                print(f"  - {p.promise_text[:80]}... | {p.domain} | {p.deadline_mentioned}")
                
    except ValueError as e:
        print(f"\nExtraction test skipped: {e}")
        print("To test extraction, set GEMINI_API_KEY in .env file or pass api_key parameter")
    except Exception as e:
        print(f"\nExtraction test failed: {e}")

    print("\nExtractor testing complete!")
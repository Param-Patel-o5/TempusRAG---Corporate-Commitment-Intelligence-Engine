"""Section detection, recursive chunking, and chunk metadata attachment.

Takes structured sections dict from ingestion.py and produces Chunk objects for embedder.py.
Handles paragraph → sentence → chunk splitting with contextual prefixes and overlap.
"""

import logging
import re

from src.config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
from src.models import Chunk

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex patterns.
    
    Note: Uses simple regex splitting, not a full NLP tokenizer.
    Handles common abbreviations but may not be perfect for all edge cases.
    
    Args:
        text: Input text to split into sentences.
        
    Returns:
        List of sentence strings, stripped of leading/trailing whitespace.
    """
    # Handle common abbreviations that shouldn't trigger sentence breaks
    text = re.sub(r'\b(U\.S\.|Inc\.|Corp\.|Ltd\.|Co\.)\s+', r'\1~ABBREV~', text)
    
    # Split on sentence ending punctuation followed by whitespace and capital letter or end of string
    # Pattern: [.!?] + whitespace + (capital letter OR end of string)
    sentence_pattern = r'[.!?]+\s+(?=[A-Z]|$)'
    
    sentences = re.split(sentence_pattern, text)
    
    # Restore abbreviation spaces and clean up
    result = []
    for sentence in sentences:
        sentence = sentence.replace('~ABBREV~', ' ').strip()
        if sentence:  # Skip empty sentences
            result.append(sentence)
    
    return result


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs on double newlines.
    
    Args:
        text: Input text to split.
        
    Returns:
        List of paragraph strings, empty paragraphs filtered out.
    """
    # Split on one or more sets of double newlines
    paragraphs = re.split(r'\n\s*\n+', text)
    
    # Filter out empty paragraphs and strip whitespace
    result = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            result.append(paragraph)
    
    return result


def chunk_subsection_text(
    text: str, 
    max_tokens: int = CHUNK_MAX_TOKENS, 
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """
    Chunk subsection text by greedily packing sentences into chunks with overlap.
    
    Token counting approximation: len(text.split()) is treated as token count.
    This is intentionally approximate to avoid adding tokenizer dependencies.
    
    Args:
        text: Raw subsection text to chunk.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Approximate tokens to overlap between chunks.
        
    Returns:
        List of plain text chunk strings (no metadata, no contextual prefix).
    """
    if not text or not text.strip():
        return []
    
    # Start with paragraphs, then break into sentences
    paragraphs = split_into_paragraphs(text)
    all_sentences = []
    
    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        all_sentences.extend(sentences)
    
    if not all_sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    
    i = 0
    while i < len(all_sentences):
        sentence = all_sentences[i]
        sentence_tokens = len(sentence.split())
        
        # Check if adding this sentence would exceed max_tokens
        if current_chunk_sentences and current_token_count + sentence_tokens > max_tokens:
            # Close current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap from the end of previous chunk
            overlap_sentences = []
            overlap_count = 0
            
            # Walk backward from end of current chunk to build overlap
            for j in range(len(current_chunk_sentences) - 1, -1, -1):
                overlap_sentence = current_chunk_sentences[j]
                overlap_sentence_tokens = len(overlap_sentence.split())
                
                if overlap_count + overlap_sentence_tokens <= overlap_tokens:
                    overlap_sentences.insert(0, overlap_sentence)
                    overlap_count += overlap_sentence_tokens
                else:
                    break
            
            # Reset for new chunk with overlap
            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_count
            
            # INFINITE LOOP FIX: Check if overlap + next sentence still exceeds max_tokens
            # If so, force-append the sentence anyway to guarantee forward progress
            if overlap_count + sentence_tokens > max_tokens:
                logger.debug(
                    f"Overlap ({overlap_count} tokens) + sentence ({sentence_tokens} tokens) "
                    f"exceeds max_tokens ({max_tokens}), force-appending to prevent infinite loop"
                )
                # Force-append the sentence and advance (accept temporary overflow)
                current_chunk_sentences.append(sentence)
                current_token_count += sentence_tokens
                i += 1
                continue
            
            # Don't increment i - try to add the same sentence to the new chunk
            continue
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens
        
        # Handle oversized single sentence
        if len(current_chunk_sentences) == 1 and sentence_tokens > max_tokens:
            logger.warning(
                f"Single sentence exceeds max_tokens ({sentence_tokens} > {max_tokens}), "
                f"keeping as standalone chunk: '{sentence[:100]}...'"
            )
            chunk_text = sentence
            chunks.append(chunk_text)
            
            # Reset for next chunk
            current_chunk_sentences = []
            current_token_count = 0
        
        i += 1
    
    # Add final chunk if any sentences remain
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(chunk_text)
    
    return chunks


def build_contextual_prefix(company: str, year: int, section: str, subsection: str) -> str:
    """
    Build contextual prefix for chunk text.
    
    Args:
        company: Company display name (e.g. "Hartford").
        year: Filing year (e.g. 2021).
        section: Section name (e.g. "MD&A").
        subsection: Subsection name (e.g. "Technology Initiatives") or "full_text".
        
    Returns:
        Formatted prefix string like "[Hartford | 2021 | MD&A > Technology Initiatives]"
        or "[Hartford | 2021 | Risk Factors]" if subsection is "full_text".
    """
    if subsection == "full_text":
        return f"[{company} | {year} | {section}]"
    else:
        return f"[{company} | {year} | {section} > {subsection}]"


def make_chunk_id(company_ticker: str, year: int, section: str, subsection: str, index: int) -> str:
    """
    Generate deterministic chunk ID.
    
    Args:
        company_ticker: Company ticker symbol (e.g. "HIG").
        year: Filing year.
        section: Section name.
        subsection: Subsection name.
        index: Chunk index within this section/subsection (0-based).
        
    Returns:
        Chunk ID like "HIG_2021_MDA_TECH_07".
    """
    def abbreviate_name(name: str) -> str:
        """Convert name to short uppercase alphanumeric token."""
        # Remove non-alphanumeric characters and split into words
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        words = clean_name.split()
        
        if not words:
            return "UNK"
        
        # Take first 4-6 characters of each significant word
        abbreviated = []
        for word in words:
            if len(word) >= 3:  # Only abbreviate meaningful words
                abbreviated.append(word[:4].upper())
            elif len(word) > 0:
                abbreviated.append(word.upper())
        
        # Join with underscore, limit total length
        result = '_'.join(abbreviated)
        if len(result) > 12:  # Reasonable limit
            result = result[:12]
        
        return result if result else "UNK"
    
    section_abbrev = abbreviate_name(section)
    subsection_abbrev = abbreviate_name(subsection) if subsection != "full_text" else ""
    
    # Build ID components
    components = [company_ticker.upper(), str(year), section_abbrev]
    if subsection_abbrev:
        components.append(subsection_abbrev)
    
    # Add zero-padded index
    components.append(f"{index:02d}")
    
    # Create base ID
    base_id = "_".join(components)
    
    # Always add a unique hash suffix to ensure absolute uniqueness
    # This is critical when processing multiple years of SEC filings with similar section names
    import hashlib
    unique_string = f"{company_ticker}_{year}_{section}_{subsection}_{index}"
    hash_suffix = hashlib.md5(unique_string.encode()).hexdigest()[:6].upper()
    
    return f"{base_id}_{hash_suffix}"


def chunk_year_sections(
    sections: dict, 
    company_ticker: str, 
    company_display_name: str, 
    year: int
) -> list[Chunk]:
    """
    Chunk all sections for one year into Chunk objects.
    
    Args:
        sections: Structured sections dict from ingestion.py.
        company_ticker: Ticker symbol for chunk IDs (e.g. "HIG").
        company_display_name: Display name for prefixes (e.g. "Hartford").
        year: Filing year.
        
    Returns:
        List of Chunk objects ready for embedder.py.
    """
    if not sections:
        logger.warning(f"No sections provided for {company_ticker} {year}")
        return []
    
    all_chunks = []
    total_chunks_for_year = 0
    
    for section_name, subsections in sections.items():
        if not isinstance(subsections, dict):
            logger.warning(f"Invalid subsections format for {section_name} in {company_ticker} {year}")
            continue
        
        if not subsections:
            logger.warning(f"Empty subsections dict for {section_name} in {company_ticker} {year}")
            continue
        
        section_chunk_count = 0
        
        for subsection_name, text in subsections.items():
            if not text or not text.strip():
                logger.debug(f"Empty subsection text: {section_name} > {subsection_name}")
                continue
            
            # Chunk the subsection text
            text_chunks = chunk_subsection_text(text)
            
            # Convert text chunks to Chunk objects
            for chunk_index, chunk_text in enumerate(text_chunks):
                # Build contextual prefix
                prefix = build_contextual_prefix(
                    company_display_name, year, section_name, subsection_name
                )
                
                # Prepend prefix to chunk text
                prefixed_text = f"{prefix} {chunk_text}"
                
                # Count words in final prefixed text
                word_count = len(prefixed_text.split())
                
                # Generate chunk ID
                chunk_id = make_chunk_id(
                    company_ticker, year, section_name, subsection_name, chunk_index
                )
                
                # Create Chunk object
                chunk = Chunk(
                    text=prefixed_text,
                    company=company_ticker,
                    year=year,
                    section=section_name,
                    subsection=subsection_name,
                    chunk_id=chunk_id,
                    word_count=word_count
                )
                
                all_chunks.append(chunk)
                section_chunk_count += 1
        
        if section_chunk_count > 0:
            logger.debug(f"Section '{section_name}': {section_chunk_count} chunks")
            total_chunks_for_year += section_chunk_count
    
    logger.info(f"Chunked {company_ticker} {year}: {total_chunks_for_year} total chunks across {len(sections)} sections")
    
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Sample data for testing
    sample_sections = {
        "MD&A": {
            "Technology Initiatives": "We are committed to modernizing our claims platform. " * 80,
            "Claims Operations": "Our claims processing has improved significantly. " * 60,
        },
        "Risk Factors": {
            "full_text": "Our business faces various risks. " * 40,
        },
        "Future Outlook": {
            "Growth Strategy": "We expect continued growth in digital channels. " * 50,
        }
    }
    
    # Test chunking
    chunks = chunk_year_sections(
        sample_sections, 
        company_ticker="HIG", 
        company_display_name="Hartford", 
        year=2021
    )
    
    print(f"\nProduced {len(chunks)} chunks")
    print("\nFirst few chunks:")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section} > {chunk.subsection}")
        print(f"Word Count: {chunk.word_count}")
        print(f"Text Preview: {chunk.text[:150]}...")
    
    # REGRESSION TEST: Infinite loop fix test
    print("\n--- REGRESSION TEST: Infinite Loop Fix ---")
    
    # Create a more targeted test case that would specifically trigger the original bug
    # The bug occurred when: overlap_count + next_sentence_tokens > max_tokens
    # This creates a scenario where overlap is exactly 50 tokens, next sentence is 470 tokens
    overlap_sentence = " ".join(["overlap"] * 50)  # exactly 50 tokens
    large_sentence = " ".join(["large"] * 470)      # exactly 470 tokens
    
    # Pattern that would cause the original infinite loop:
    # 1. First chunk fills up and creates 50-token overlap
    # 2. 50 + 470 = 520 > 500, so it would retry without progress
    targeted_test = f"{overlap_sentence} {large_sentence} {large_sentence} {large_sentence}"
    
    print(f"Testing targeted infinite loop scenario...")
    print(f"Overlap sentence: 50 tokens, Large sentence: 470 tokens each")
    
    try:
        import time
        start_time = time.time()
        
        targeted_chunks = chunk_subsection_text(targeted_test, max_tokens=500, overlap_tokens=50)
        
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.2f}s - no infinite loop")
        print(f"✓ Produced {len(targeted_chunks)} chunks")
        
        # Show chunk sizes to verify behavior
        for i, chunk in enumerate(targeted_chunks):
            token_count = len(chunk.split())
            print(f"  Chunk {i}: {token_count} tokens")
            
    except Exception as e:
        print(f"✗ Targeted test failed: {e}")
        
    # Also test the original problematic pattern
    print(f"\nTesting original problematic text pattern...")
    # Construct text with short sentence followed by long sentence pattern
    short_sentence = "This is a short sentence with about twenty tokens here for testing purposes."
    long_sentence = " ".join(["This", "is", "a", "very", "long", "sentence"] * 80)  # ~490 tokens
    
    # Create pattern: short + long, repeated 3 times (reduced from 5 to minimize output)
    problematic_text = ""
    for i in range(3):
        problematic_text += short_sentence + " " + long_sentence + " "
    
    print(f"Text length: {len(problematic_text.split())} tokens")
    
    # This should not hang and should produce reasonable chunks
    try:
        problem_chunks = chunk_subsection_text(problematic_text, max_tokens=500, overlap_tokens=50)
        print(f"✓ No infinite loop: produced {len(problem_chunks)} chunks")
        
        # Check for reasonable chunk count (not one per sentence pair, not absurd count)
        if 3 <= len(problem_chunks) <= 10:
            print(f"✓ Reasonable chunk count: {len(problem_chunks)}")
        else:
            print(f"⚠ Unusual chunk count: {len(problem_chunks)} (expected 3-10)")
            
    except Exception as e:
        print(f"✗ Error during regression test: {e}")
    
    # Test with real HIG data if available
    print("\n--- REAL DATA TEST ---")
    try:
        from pathlib import Path
        hig_files = list(Path("data/filings").glob("HIG_*.txt"))
        
        if hig_files:
            # Load first available HIG file
            test_file = hig_files[0]
            print(f"Testing with real HIG data: {test_file.name}")
            
            with open(test_file, 'r', encoding='utf-8') as f:
                real_text = f.read()
            
            # Take a large section for testing (first 50,000 characters)
            test_section_text = real_text[:50000]
            
            # Wrap as single subsection
            real_sections = {"Test Section": {"full_text": test_section_text}}
            
            print(f"Real text sample length: {len(test_section_text.split())} tokens")
            
            # Test chunking on real data
            real_chunks = chunk_year_sections(
                real_sections,
                company_ticker="HIG",
                company_display_name="Hartford", 
                year=2021
            )
            
            print(f"✓ Real data test completed: {len(real_chunks)} chunks produced")
            
            # Check for duplicates in real data chunks
            real_duplicates = False
            for i in range(len(real_chunks) - 1):
                if real_chunks[i].text[:100] == real_chunks[i + 1].text[:100]:
                    real_duplicates = True
                    break
            
            if not real_duplicates:
                print("✓ No duplicate chunks in real data test")
            else:
                print("⚠ Duplicate chunks found in real data test")
                
        else:
            print("No HIG files found in data/filings/, skipping real data test")
            
    except Exception as e:
        print(f"Real data test error: {e}")
    
    # Test sentence splitting
    print("\n--- Testing Sentence Splitting ---")
    test_text = "The U.S. market is strong. Inc. profits are up! What will happen next? We don't know."
    sentences = split_into_sentences(test_text)
    print(f"Input: {test_text}")
    print(f"Sentences: {sentences}")
    
    # Test paragraph splitting  
    print("\n--- Testing Paragraph Splitting ---")
    test_paragraphs = "First paragraph here.\n\nSecond paragraph here.\n\n\nThird paragraph after extra newlines."
    paragraphs = split_into_paragraphs(test_paragraphs)
    print(f"Paragraphs: {paragraphs}")
"""Environment configuration and shared constants — single source of truth for all tunable parameters.

Every other module in src/ imports from this file to ensure consistency across the pipeline.
No logic, no I/O beyond loading .env — pure constants and a single utility function.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ============================================================================
# API KEYS AND EXTERNAL SERVICE CONFIG
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
"""Google Generative AI (Gemini 1.5 Flash) API key for promise extraction."""

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
"""Groq API key for Llama 3.1 70B judge model."""

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
"""User-Agent header for SEC EDGAR requests. Required by SEC policy."""

# ============================================================================
# DIRECTORY PATHS (as Path objects for pathlib operations)
# ============================================================================

CHROMADB_PATH = Path(os.getenv("CHROMADB_PATH", "./chroma_store"))
"""Persistent ChromaDB storage directory (one collection per ticker)."""

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/filings"))
"""Cached SEC 10-K filings stored as plaintext (ticker_year.txt format)."""

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
"""Application log files directory."""

# Ensure all directories exist
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CACHE AND STALENESS CONFIG
# ============================================================================

CACHE_STALENESS_DAYS = int(os.getenv("CACHE_STALENESS_DAYS", "180"))
"""Filing cache is considered stale if older than this many days; triggers re-download."""

# ============================================================================
# CHUNKING PARAMETERS (ingestion + chunking pipeline)
# ============================================================================

CHUNK_MAX_TOKENS = 500
"""Maximum tokens per chunk after splitting; enforced by chunker.py."""

CHUNK_OVERLAP_TOKENS = 50
"""Overlap between consecutive chunks to preserve context bridges."""

# ============================================================================
# EMBEDDING PARAMETERS
# ============================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
"""Sentence-Transformers model for dense embeddings (384-dimensional)."""

EMBEDDING_DIMENSIONS = 384
"""Output dimensionality of the embedding model."""

# ============================================================================
# RETRIEVAL PARAMETERS (hybrid search + RRF fusion)
# ============================================================================

DENSE_TOP_K = 20
"""Top-K results from dense (semantic) search before RRF fusion."""

SPARSE_TOP_K = 20
"""Top-K results from sparse (BM25) search before RRF fusion."""

RRF_K = 60
"""RRF parameter; controls rank weighting in reciprocal rank fusion."""

FINAL_TOP_N = 5
"""Final number of ranked results returned to extraction/reasoning pipeline."""

# ============================================================================
# SECTION BOOST WEIGHTS (retrieval result re-ranking)
# ============================================================================

SECTION_BOOST_WEIGHTS = {
    "Future Outlook": 1.3,
    "MD&A": 1.2,
    "Financial Results": 1.0,
    "Risk Factors": 0.8,
}
"""
Multipliers applied to relevance scores based on section source.
Sections in Future Outlook are more likely to contain promises.
Sections outside this dict fall back to DEFAULT_SECTION_BOOST.
"""

DEFAULT_SECTION_BOOST = 1.0
"""Default boost weight for sections not explicitly listed above."""

# ============================================================================
# SIMILARITY THRESHOLDS (evidence classification)
# ============================================================================

SIMILARITY_STRONG_EVIDENCE = 0.7
"""Similarity ≥ this threshold → strong evidence for delivery."""

SIMILARITY_WEAK_EVIDENCE = 0.5
"""Similarity in range [WEAK, STRONG) → weak or partial evidence.
Similarity < this threshold → absent, candidate for Silently Abandoned.
"""

# ============================================================================
# SECTION ROLES (where promises are extracted vs where evidence is searched)
# ============================================================================

PROMISE_EXTRACTION_SECTIONS = [
    "MD&A", 
    "management", 
    "discussion", 
    "outlook", 
    "future", 
    "growth", 
    "strategy"
]
"""Keyword patterns for substring matching in section names (case-insensitive)."""

EVIDENCE_SEARCH_SECTIONS = ["MD&A", "Financial Results", "Claims Operations"]
"""Sections searched for evidence of promise fulfillment (LLM Call 2, Groq)."""

# ============================================================================
# RECENCY WEIGHTS FOR CREDIBILITY SCORING
# ============================================================================


def recency_weight(year: int, most_recent_year: int) -> float:
    """
    Returns recency weight for a given fiscal year relative to the most recent year analyzed.

    Most recent year = 1.0, each year back decreases by 0.1, floor at 0.6.
    Matches documented mapping:
        - most_recent = 1.0
        - -1 year = 0.9
        - -2 years = 0.8
        - -3 years = 0.7
        - -4+ years = 0.6 (clamped)

    Args:
        year: Fiscal year of the promise or evidence.
        most_recent_year: The most recent year in the analysis set.

    Returns:
        float: Recency weight in range [0.6, 1.0].
    """
    offset = most_recent_year - year
    weight = 1.0 - (offset * 0.1)
    return max(weight, 0.6)


# ============================================================================
# CONFIDENCE SCORE SIGNAL WEIGHTS (rule-based promise extraction)
# ============================================================================

CONFIDENCE_SIGNAL_WEIGHTS = {
    "has_deadline": 0.3,
    "has_metric": 0.3,
    "has_commitment_verb": 0.2,
    "domain_detectable": 0.2,
}
"""
Weights for signals used to compute confidence_score on extracted promises.
Sum of all weights should equal 1.0.
Used by scorer.py to assign confidence_score (0.0-1.0) before LLM judge.
"""

# ============================================================================
# JUDGE PROMPT ANCHORS (LLM Call 2 scoring scale)
# ============================================================================

JUDGE_ANCHORS = """90-100: Explicit delivery with matching metrics
70-89:  Confirmed progress, metrics incomplete
50-69:  Topic mentioned, no clear confirmation
30-49:  Contradicted or shortfall shown
0-29:   No evidence or explicit failure""".strip()
"""
Scoring rubric for Groq Llama judge (reasoner.py, LLM Call 2).
Embedded directly into judge prompts to ground delivery_score (0-100).
"""

# ============================================================================
# LLM MODEL IDENTIFIERS
# ============================================================================

EXTRACTOR_MODEL = "gemini-3.6-flash"
"""Google Generative AI model for LLM Call 1 (promise extraction)."""

JUDGE_MODEL = "groq/compound"
"""Groq model identifier for LLM Call 2 (cross-year reasoning + scoring).
Updated to use groq/compound (Groq's router model) instead of decommissioned llama-3.1-70b-versatile."""

# ============================================================================
# RATE LIMITING AND RETRY CONFIG
# ============================================================================

SEC_MIN_REQUEST_INTERVAL = 0.11
"""Minimum seconds between SEC EDGAR requests (stay well under 10 req/sec limit)."""

# Gemini API rate limiting (for promise extraction)
GEMINI_REQUESTS_PER_MINUTE = 15
"""Maximum requests per minute for Gemini API to avoid 429 errors."""

GEMINI_REQUEST_INTERVAL = 60.0 / GEMINI_REQUESTS_PER_MINUTE
"""Minimum seconds between Gemini API calls."""

# Groq API rate limiting (for promise judging) - more conservative after testing
GROQ_REQUESTS_PER_MINUTE = 10
"""Maximum requests per minute for Groq API to avoid 429 errors (reduced after testing)."""

GROQ_REQUEST_INTERVAL = 60.0 / GROQ_REQUESTS_PER_MINUTE
"""Minimum seconds between Groq API calls (6 seconds)."""

# Retry configuration for both APIs
MAX_RETRIES = 3
"""Maximum number of retries for failed API calls."""

RETRY_BASE_DELAY = 1.0
"""Base delay in seconds for exponential backoff retry."""

RETRY_MAX_DELAY = 60.0
"""Maximum delay in seconds for exponential backoff retry."""

# ============================================================================
# VALIDATION AT IMPORT TIME
# ============================================================================

# Note: We do NOT raise errors for missing API keys or SEC_USER_AGENT here.
# - LLM keys may be provided directly via Streamlit UI at runtime (app.py handles it).
# - SEC_USER_AGENT fallback is logged in ingestion.py if empty.
# - This keeps config.py pure and prevents app startup failures for missing credentials.


if __name__ == "__main__":
    print("=== TempusRAG Configuration ===\n")

    print("Directories:")
    print(f"  CHROMADB_PATH: {CHROMADB_PATH}")
    print(f"  DATA_DIR: {DATA_DIR}")
    print(f"  LOG_DIR: {LOG_DIR}")

    print("\nChunking:")
    print(f"  CHUNK_MAX_TOKENS: {CHUNK_MAX_TOKENS}")
    print(f"  CHUNK_OVERLAP_TOKENS: {CHUNK_OVERLAP_TOKENS}")

    print("\nEmbeddings:")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    print(f"  EMBEDDING_DIMENSIONS: {EMBEDDING_DIMENSIONS}")

    print("\nRetrieval:")
    print(f"  DENSE_TOP_K: {DENSE_TOP_K}")
    print(f"  SPARSE_TOP_K: {SPARSE_TOP_K}")
    print(f"  RRF_K: {RRF_K}")
    print(f"  FINAL_TOP_N: {FINAL_TOP_N}")

    print("\nSection Boost Weights:")
    for section, weight in SECTION_BOOST_WEIGHTS.items():
        print(f"  {section}: {weight}")
    print(f"  DEFAULT: {DEFAULT_SECTION_BOOST}")

    print("\nSimilarity Thresholds:")
    print(f"  STRONG_EVIDENCE: {SIMILARITY_STRONG_EVIDENCE}")
    print(f"  WEAK_EVIDENCE: {SIMILARITY_WEAK_EVIDENCE}")

    print("\nSections:")
    print(f"  PROMISE_EXTRACTION: {PROMISE_EXTRACTION_SECTIONS}")
    print(f"  EVIDENCE_SEARCH: {EVIDENCE_SEARCH_SECTIONS}")

    print("\nRecency Weights (example):")
    print(f"  recency_weight(2024, 2024): {recency_weight(2024, 2024):.1f}")
    print(f"  recency_weight(2023, 2024): {recency_weight(2023, 2024):.1f}")
    print(f"  recency_weight(2022, 2024): {recency_weight(2022, 2024):.1f}")
    print(f"  recency_weight(2021, 2024): {recency_weight(2021, 2024):.1f}")
    print(f"  recency_weight(2020, 2024): {recency_weight(2020, 2024):.1f}")

    print("\nConfidence Weights:")
    for signal, weight in CONFIDENCE_SIGNAL_WEIGHTS.items():
        print(f"  {signal}: {weight}")

    print("\nLLM Models:")
    print(f"  EXTRACTOR_MODEL: {EXTRACTOR_MODEL}")
    print(f"  JUDGE_MODEL: {JUDGE_MODEL}")

    print("\nRate Limiting:")
    print(f"  SEC_MIN_REQUEST_INTERVAL: {SEC_MIN_REQUEST_INTERVAL}s")

    print("\nConfig loaded successfully!")

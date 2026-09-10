"""
TempusRAG Pipeline Orchestrator

End-to-end pipeline that chains all components together:
ingestion → chunking → embedding → extraction → reasoning → scoring

Given a ticker, produces a CompanyCredibilityReport from SEC filings.
"""

import logging
import time
from typing import Callable, Dict, List, Optional

from src.ingestion import load_ticker_cik_map, ingest_company
from src.chunker import chunk_year_sections
from src.embedder import embed_and_store_chunks
from src.extractor import extract_all_promises
from src.reasoner import reason_across_years
from src.scorer import generate_credibility_report
from src.config import DATA_DIR, GEMINI_API_KEY, GROQ_API_KEY
from src.models import CompanyCredibilityReport, Chunk, Promise, DeliveryEvidence

logger = logging.getLogger(__name__)

# Last-run artifacts for the dashboard (not part of the report schema).
_LAST_ARTIFACTS: Dict[str, dict] = {}


def get_last_artifacts(ticker: str) -> Optional[dict]:
    """Return delivery evidence and promises from the most recent pipeline run."""
    return _LAST_ARTIFACTS.get(ticker.upper())


def validate_ticker(ticker: str, cik_map: dict) -> str:
    """
    Validate ticker exists in CIK map and return normalized ticker.
    
    Args:
        ticker: Company ticker symbol
        cik_map: Mapping from ticker to CIK
        
    Returns:
        Normalized ticker (uppercase)
        
    Raises:
        ValueError: If ticker not found in CIK map
    """
    ticker_upper = ticker.upper()
    if ticker_upper not in cik_map:
        available_tickers = sorted(list(cik_map.keys()))[:10]  # Show first 10
        raise ValueError(
            f"Ticker '{ticker}' not found in CIK map. "
            f"Available tickers include: {available_tickers}..."
        )
    return ticker_upper


def run_tempusrag_pipeline(
    ticker: str,
    company_display_name: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    force_reingest: bool = False,
    on_stage: Optional[Callable[[str], None]] = None,
) -> CompanyCredibilityReport:
    """
    Run complete TempusRAG pipeline for a company ticker.
    
    Args:
        ticker: Company ticker (e.g. "HIG")
        company_display_name: Human-readable name (defaults to ticker if None)
        gemini_api_key: Optional override for Gemini key (uses config if None)
        groq_api_key: Optional override for Groq key (uses config if None)
        force_reingest: If True, re-download and re-embed filings even if cached
        on_stage: Optional callback invoked with the current stage label
        
    Returns:
        CompanyCredibilityReport with analysis results
        
    Raises:
        ValueError: If ticker not found or validation fails
        Exception: If critical pipeline steps fail
    """
    start_time = time.time()
    
    # Use provided display name or default to ticker
    if company_display_name is None:
        company_display_name = ticker.upper()
    
    # Use provided API keys or fall back to config
    gemini_key = gemini_api_key or GEMINI_API_KEY
    groq_key = groq_api_key or GROQ_API_KEY
    
    logger.info(f"Starting TempusRAG pipeline for {ticker} ({company_display_name})")

    def stage(label: str) -> None:
        logger.info(label)
        if on_stage:
            on_stage(label)

    try:
        stage("Validate")
        cik_map = load_ticker_cik_map()
        validated_ticker = validate_ticker(ticker, cik_map)
        logger.info(f"Validated ticker: {validated_ticker}")

        stage("Ingest")
        try:
            year_sections = ingest_company(validated_ticker, cik_map, DATA_DIR)
            if not year_sections:
                raise ValueError(f"No filings found for ticker {validated_ticker}")
            logger.info(f"Ingested {len(year_sections)} years of filings")
        except Exception as e:
            logger.error(f"Ingestion failed for {validated_ticker}: {e}")
            raise Exception(f"Failed to ingest filings for {validated_ticker}: {e}")
        
        stage("Chunk")
        all_chunks = {}
        total_chunks = 0
        try:
            for year, sections_dict in year_sections.items():
                chunks = chunk_year_sections(
                    sections_dict,
                    company_ticker=validated_ticker,
                    company_display_name=company_display_name,
                    year=year
                )
                all_chunks[year] = chunks
                total_chunks += len(chunks)
                logger.info(f"Chunked {year}: {len(chunks)} chunks")
            logger.info(f"Total chunks across all years: {total_chunks}")
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            # Continue with empty chunks - may have cached embeddings
            all_chunks = {}
        
        stage("Embed")
        try:
            for year, chunks in all_chunks.items():
                if chunks:  # Only embed if we have chunks
                    years_covered = [year]  # Single year per call
                    embed_and_store_chunks(chunks, validated_ticker, years_covered)
                    logger.info(f"Embedded and stored {len(chunks)} chunks for {year}")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            logger.warning("Continuing with cached embeddings if available")
        
        stage("Extract")
        all_promises = {}
        total_promises = 0
        try:
            all_promises = extract_all_promises(
                all_chunks, 
                api_key=gemini_key
            )
            for year, promises in all_promises.items():
                total_promises += len(promises)
                logger.info(f"Extracted {len(promises)} promises from {year}")
            logger.info(f"Total promises extracted: {total_promises}")
        except Exception as e:
            logger.error(f"Promise extraction failed: {e}")
            logger.warning("Continuing with empty promises - will generate report without promise analysis")
            all_promises = {}
        
        stage("Judge")
        delivery_evidence = []
        try:
            if all_promises:
                delivery_evidence = reason_across_years(
                    all_promises, 
                    validated_ticker, 
                    api_key=groq_key
                )
                logger.info(f"Generated {len(delivery_evidence)} delivery judgments")
            else:
                logger.info("No promises to judge - skipping reasoning step")
        except Exception as e:
            logger.error(f"Cross-year reasoning failed: {e}")
            logger.warning("Continuing without delivery evidence - will use confidence-only scoring")
            delivery_evidence = []
        
        stage("Score")
        try:
            report = generate_credibility_report(
                company_ticker=validated_ticker,
                company_display_name=company_display_name,
                promises_by_year=all_promises,
                delivery_evidence=delivery_evidence
            )
            logger.info("Successfully generated credibility report")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise Exception(f"Failed to generate credibility report: {e}")
        
        stage("Done")
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Final report: {report.total_promises} promises, "
                   f"overall score {report.overall_score:.1f}/100.0")

        _LAST_ARTIFACTS[validated_ticker] = {
            "delivery_evidence": delivery_evidence,
            "promises_by_year": all_promises,
            "elapsed_seconds": elapsed_time,
        }

        return report
        
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Pipeline failed after {elapsed_time:.2f} seconds: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    # Run full pipeline on Hartford
    try:
        report = run_tempusrag_pipeline(
            ticker="HIG",
            company_display_name="The Hartford",
            gemini_api_key=None,  # Uses config
            groq_api_key=None,    # Uses config
            force_reingest=False
        )
        
        print("\n" + "="*60)
        print(f"CREDIBILITY REPORT: {report.company}")
        print("="*60)
        print(f"Years Analyzed: {report.years_analyzed}")
        print(f"Total Promises: {report.total_promises}")
        print(f"Delivered: {report.delivered} | Partial: {report.partial} | Abandoned: {report.abandoned}")
        print(f"Overall Score: {report.overall_score:.1f}/100.0")
        print(f"\nDomain Scores:")
        for domain, score in report.domain_scores.items():
            print(f"  {domain}: {score:.1f}")
        print(f"\nRed Flags: {len(report.red_flags)}")
        for flag in report.red_flags:
            print(f"  - {flag}")
        print("="*60 + "\n")
        
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
"""Cross-year reasoning and judging using Groq Llama 3.1 70B.

This is LLM Call 2 in the TempusRAG architecture. Takes extracted promises from extractor.py
and searches for evidence of delivery across subsequent years, then judges each promise as 
Delivered/Partial/Silently Abandoned/Pending using async batched calls to Groq.

Rate limiting: Implements exponential backoff retry logic to handle 429 rate limit errors.
"""

import asyncio
import json
import logging
import time
import random

import groq

from src.config import (
    GROQ_API_KEY, JUDGE_MODEL, JUDGE_ANCHORS,
    GROQ_REQUEST_INTERVAL, MAX_RETRIES, RETRY_BASE_DELAY, RETRY_MAX_DELAY
)
from src.models import Promise, DeliveryEvidence, Chunk
from src.retriever import hybrid_retrieve

logger = logging.getLogger(__name__)

# Track last request time for rate limiting (thread-safe with asyncio)
_last_groq_request_time = 0.0
_groq_lock = asyncio.Lock()

async def _enforce_rate_limit_async():
    """Enforce rate limiting between Groq API calls (async-safe)."""
    global _last_groq_request_time
    async with _groq_lock:
        current_time = time.time()
        time_since_last = current_time - _last_groq_request_time
        
        if time_since_last < GROQ_REQUEST_INTERVAL:
            sleep_time = GROQ_REQUEST_INTERVAL - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        _last_groq_request_time = time.time()

def _exponential_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
    # Add jitter to prevent thundering herd
    jitter = random.uniform(0.1, 0.3) * delay
    return delay + jitter


def search_evidence_for_promise(promise: Promise, ticker: str, evidence_search_years: list[int]) -> list[Chunk]:
    """
    Search for evidence of promise delivery across subsequent years.
    
    Args:
        promise: Promise object to search evidence for
        ticker: Company ticker symbol  
        evidence_search_years: List of years to search in (typically N+1, N+2, N+3)
        
    Returns:
        list[Chunk]: De-duplicated chunks that may contain delivery evidence
    """
    all_chunks = []
    
    for year in evidence_search_years:
        try:
            year_chunks = hybrid_retrieve(promise.promise_text, ticker, top_n=5)
            # Filter to chunks from this specific year
            year_filtered = [chunk for chunk in year_chunks if chunk.year == year]
            all_chunks.extend(year_filtered)
            
        except Exception as e:
            logger.warning(f"Failed to retrieve evidence for promise {promise.promise_text[:50]}... in year {year}: {e}")
    
    # De-duplicate by chunk_id
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(chunk.chunk_id)
    
    logger.info(f"Found {len(unique_chunks)} evidence chunks for promise: {promise.promise_text[:50]}...")
    return unique_chunks


def build_judge_prompt(promise: Promise, evidence_chunks: list[Chunk]) -> str:
    """
    Build prompt for judging promise delivery based on evidence.
    
    Args:
        promise: Promise object to judge
        evidence_chunks: Evidence chunks found from subsequent years
        
    Returns:
        str: Complete judge prompt for Groq Llama
    """
    # Format evidence or indicate none found
    if evidence_chunks:
        evidence_text = "\n\n".join([
            f"--- Evidence {i+1} (Year {chunk.year}) ---\n{chunk.text}"
            for i, chunk in enumerate(evidence_chunks)
        ])
    else:
        evidence_text = "No evidence found in subsequent years."
    
    prompt = f"""You are judging whether a management promise was delivered based on evidence from subsequent SEC filings.

PROMISE TO JUDGE:
"{promise.promise_text}"
- Domain: {promise.domain}
- Made in: {promise.year_made}
- Deadline mentioned: {promise.deadline_mentioned or "None specified"}

EVIDENCE FROM SUBSEQUENT YEARS:
{evidence_text}

SCORING SCALE (use these anchors):
{JUDGE_ANCHORS}

INSTRUCTIONS:
1. Analyze the evidence to determine if the promise was delivered, partially delivered, or abandoned
2. Score from 0-100 based on the anchors above
3. Choose status: Delivered (70-100), Partial (50-69), Silently Abandoned (0-49), or Pending (if deadline not yet reached)
4. Provide reasoning explaining your assessment

{"If no evidence is found, this typically indicates the promise was silently abandoned (score 0-29)." if not evidence_chunks else ""}

RESPONSE FORMAT: Respond with ONLY a JSON object, no markdown fences:
{{"delivery_score": <0-100>, "status": "<Delivered|Partial|Silently Abandoned|Pending>", "judge_reasoning": "<your reasoning>"}}"""

    return prompt


async def _call_groq_judge(prompt: str, api_key: str) -> str:
    """
    Low-level call to Groq Llama model with rate limiting and retry logic.
    
    Args:
        prompt: Judge prompt to send
        api_key: Groq API key for authentication
        
    Returns:
        str: Raw response text from Groq
        
    Raises:
        Exception: If all retries are exhausted or non-recoverable error occurs.
    """
    client = groq.Groq(api_key=api_key)
    
    for attempt in range(MAX_RETRIES + 1):  # 0-based attempts, so +1 for total tries
        try:
            # Enforce rate limiting before making request
            await _enforce_rate_limit_async()
            
            # Make the API call
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Low temperature for consistent judging
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if this is a rate limit error (429 or quota exceeded)
            is_rate_limit = ('429' in error_str or 
                           'rate limit' in error_str or 
                           'quota exceeded' in error_str or
                           'too many requests' in error_str)
            
            if is_rate_limit and attempt < MAX_RETRIES:
                # Exponential backoff for rate limit errors
                delay = _exponential_backoff_delay(attempt)
                logger.warning(f"Rate limit hit on attempt {attempt + 1}/{MAX_RETRIES + 1}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
                continue
            elif attempt < MAX_RETRIES:
                # For other errors, shorter delay
                delay = RETRY_BASE_DELAY * (attempt + 1)
                logger.warning(f"API error on attempt {attempt + 1}/{MAX_RETRIES + 1}, retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
                continue
            else:
                # All retries exhausted
                logger.error(f"All {MAX_RETRIES + 1} attempts failed for Groq API call: {e}")
                raise


def _parse_judge_json(raw_text: str) -> dict:
    """
    Parse JSON response from Groq judge, handling markdown fences.
    
    Args:
        raw_text: Raw response text from Groq
        
    Returns:
        dict: Parsed JSON with delivery_score, status, judge_reasoning
        
    Raises:
        json.JSONDecodeError: If JSON parsing fails (propagated to caller)
        ValueError: If required keys are missing
    """
    # Strip markdown code fences if present
    text = raw_text.strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
        
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    # Parse JSON
    result = json.loads(text)
    
    # Validate required keys
    required_keys = {"delivery_score", "status", "judge_reasoning"}
    if not all(key in result for key in required_keys):
        missing = required_keys - set(result.keys())
        raise ValueError(f"Missing required keys in judge response: {missing}")
    
    return result


async def judge_single_promise_async(promise: Promise, evidence_chunks: list[Chunk], api_key: str | None = None) -> DeliveryEvidence:
    """
    Judge a single promise using Groq Llama model (async version).
    
    Args:
        promise: Promise to judge
        evidence_chunks: Evidence chunks from subsequent years
        api_key: Optional API key override (uses config if None)
        
    Returns:
        DeliveryEvidence: Judgment result with score, status, and reasoning
    """
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GROQ_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Groq API key available - set GROQ_API_KEY in config or pass one explicitly"
        )
    
    # Build prompt
    prompt = build_judge_prompt(promise, evidence_chunks)
    
    # First attempt
    try:
        raw_response = await _call_groq_judge(prompt, resolved_api_key)
        judge_result = _parse_judge_json(raw_response)
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Judge parse failed for promise {promise.promise_text[:50]}... on first attempt: {e}")
        
        # Retry with stricter prompt
        retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Respond with ONLY the JSON object, nothing else."
        
        try:
            raw_response = await _call_groq_judge(retry_prompt, resolved_api_key)
            judge_result = _parse_judge_json(raw_response)
            logger.info(f"Judge retry successful for promise: {promise.promise_text[:50]}...")
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Judge parse failed after retry for promise {promise.promise_text[:50]}...: {e}")
            logger.error(f"Raw response: {raw_response[:500]}...")
            
            # Fail gracefully
            return DeliveryEvidence(
                promise=promise,
                evidence_text="\n".join([chunk.text for chunk in evidence_chunks]) if evidence_chunks else None,
                year_found=max([chunk.year for chunk in evidence_chunks]) if evidence_chunks else None,
                delivery_score=0,
                status="Silently Abandoned",
                judge_reasoning="Judge call failed - could not parse response"
            )
    
    # Construct DeliveryEvidence object
    try:
        evidence = DeliveryEvidence(
            promise=promise,
            evidence_text="\n".join([chunk.text for chunk in evidence_chunks]) if evidence_chunks else None,
            year_found=max([chunk.year for chunk in evidence_chunks]) if evidence_chunks else None,
            delivery_score=int(judge_result["delivery_score"]),
            status=judge_result["status"],
            judge_reasoning=judge_result["judge_reasoning"]
        )
        
        logger.info(f"Successfully judged promise (score: {evidence.delivery_score}, status: {evidence.status}): {promise.promise_text[:50]}...")
        return evidence
        
    except Exception as e:
        logger.error(f"Failed to construct DeliveryEvidence for promise {promise.promise_text[:50]}...: {e}")
        
        # Fail gracefully
        return DeliveryEvidence(
            promise=promise,
            evidence_text="\n".join([chunk.text for chunk in evidence_chunks]) if evidence_chunks else None,
            year_found=max([chunk.year for chunk in evidence_chunks]) if evidence_chunks else None,
            delivery_score=0,
            status="Silently Abandoned", 
            judge_reasoning=f"Failed to construct result: {e}"
        )

def judge_single_promise(promise: Promise, evidence_chunks: list[Chunk], api_key: str | None = None) -> DeliveryEvidence:
    """
    Judge a single promise using Groq Llama model (sync wrapper).
    
    Args:
        promise: Promise to judge
        evidence_chunks: Evidence chunks from subsequent years
        api_key: Optional API key override (uses config if None)
        
    Returns:
        DeliveryEvidence: Judgment result with score, status, and reasoning
    """
    return asyncio.run(judge_single_promise_async(promise, evidence_chunks, api_key))


async def _judge_batch(batch_with_evidence: list[tuple[Promise, list[Chunk]]], api_key: str) -> list[DeliveryEvidence]:
    """
    Helper to judge a batch of promises concurrently with rate limiting.
    
    Args:
        batch_with_evidence: List of (promise, evidence_chunks) tuples
        api_key: Groq API key
        
    Returns:
        list[DeliveryEvidence]: Results for the batch (may include exceptions)
    """
    tasks = [
        judge_single_promise_async(promise, evidence, api_key)
        for promise, evidence in batch_with_evidence
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            promise, evidence = batch_with_evidence[i]
            logger.error(f"Exception judging promise {promise.promise_text[:50]}...: {result}")
            
            # Create fallback DeliveryEvidence
            fallback = DeliveryEvidence(
                promise=promise,
                evidence_text="\n".join([chunk.text for chunk in evidence]) if evidence else None,
                year_found=max([chunk.year for chunk in evidence]) if evidence else None,
                delivery_score=0,
                status="Silently Abandoned",
                judge_reasoning=f"Exception during judging: {result}"
            )
            final_results.append(fallback)
        else:
            final_results.append(result)
    
    return final_results


def batch_judge_promises(promises: list[Promise], ticker: str, evidence_search_years: list[int], 
                        api_key: str | None = None, batch_size: int = 1) -> list[DeliveryEvidence]:
    """
    Judge multiple promises in batches using async calls to Groq with conservative rate limiting.
    
    Args:
        promises: List of promises to judge
        ticker: Company ticker for evidence search
        evidence_search_years: Years to search for evidence
        api_key: Optional API key override
        batch_size: Number of promises per concurrent batch (1 = sequential for max safety)
        
    Returns:
        list[DeliveryEvidence]: Delivery evidence for all promises in input order
    """
    start_time = time.time()
    
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GROQ_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Groq API key available - set GROQ_API_KEY in config or pass one explicitly"
        )
    
    logger.info(f"Starting batch judgment of {len(promises)} promises with batch_size={batch_size}")
    
    # Search evidence for all promises first
    promises_with_evidence = []
    for promise in promises:
        # Calculate evidence search years for this promise
        max_year = max(evidence_search_years) if evidence_search_years else promise.year_made + 3
        search_years = [y for y in evidence_search_years if y > promise.year_made][:3]  # Limit to 3 years ahead
        
        evidence_chunks = search_evidence_for_promise(promise, ticker, search_years)
        promises_with_evidence.append((promise, evidence_chunks))
    
    # Create batches
    batches = []
    for i in range(0, len(promises_with_evidence), batch_size):
        batch = promises_with_evidence[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches for concurrent processing")
    
    # Run all batches concurrently
    async def _run_all_batches():
        batch_tasks = []
        for i, batch in enumerate(batches):
            # Include batch index for safer error handling
            batch_tasks.append((i, _judge_batch(batch, resolved_api_key)))
        
        # Execute all batches
        indexed_results = await asyncio.gather(
            *[task for _, task in batch_tasks], 
            return_exceptions=True
        )
        
        # Flatten results while handling failures robustly
        all_results = []
        for i, batch_result in enumerate(indexed_results):
            if isinstance(batch_result, Exception):
                logger.error(f"Batch {i} processing failed: {batch_result}")
                # Create fallback results for this specific failed batch
                failed_batch = batches[i]
                for promise, evidence in failed_batch:
                    fallback = DeliveryEvidence(
                        promise=promise,
                        evidence_text="\n".join([chunk.text for chunk in evidence]) if evidence else None,
                        year_found=max([chunk.year for chunk in evidence]) if evidence else None,
                        delivery_score=0,
                        status="Silently Abandoned",
                        judge_reasoning=f"Batch {i} processing failed: {batch_result}"
                    )
                    all_results.append(fallback)
            else:
                all_results.extend(batch_result)
        
        return all_results
    
    # Execute async batch processing
    results = asyncio.run(_run_all_batches())
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Count successes/failures
    success_count = sum(1 for r in results if "failed" not in r.judge_reasoning.lower())
    failure_count = len(results) - success_count
    
    logger.info(f"Batch judgment complete: {total_time:.2f}s, {len(batches)} batches, "
               f"{success_count} success, {failure_count} failures")
    
    return results


def reason_across_years(promises_by_year: dict[int, list[Promise]], ticker: str, api_key: str | None = None) -> list[DeliveryEvidence]:
    """
    Top-level orchestrator for cross-year reasoning and judging.
    
    Args:
        promises_by_year: Dictionary mapping year to promises extracted in that year
        ticker: Company ticker symbol
        api_key: Optional API key override
        
    Returns:
        list[DeliveryEvidence]: All delivery evidence across all years
    """
    start_time = time.time()
    
    all_promises = []
    available_years = sorted(promises_by_year.keys())
    
    # Collect all promises with their evidence search years
    for year in available_years:
        year_promises = promises_by_year[year]
        
        # For promises made in year N, search in years N+1, N+2, N+3 (where available)
        evidence_search_years = [y for y in available_years if y > year][:3]
        
        logger.info(f"Year {year}: {len(year_promises)} promises, evidence search in years {evidence_search_years}")
        
        for promise in year_promises:
            all_promises.append(promise)
    
    if not all_promises:
        logger.info("No promises to judge")
        return []
    
    # Use evidence search years from all available years
    evidence_years = [y for y in available_years if y > min(available_years)]
    
    # Batch judge all promises
    all_evidence = batch_judge_promises(all_promises, ticker, evidence_years, api_key)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Cross-year reasoning complete: {len(all_promises)} promises judged in {total_time:.2f}s")
    
    return all_evidence


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    from src.models import Promise, Chunk
    
    # Sample promise and evidence
    sample_promise = Promise(
        promise_text="We will reduce claims processing time by 40%",
        domain="Claims",
        deadline_mentioned="2023", 
        year_made=2021,
        page_number=0,
        confidence_score=0.8
    )
    
    sample_evidence = [
        Chunk(
            text="[Hartford | 2023 | MD&A] Claims processing time improved by 35% year-over-year.",
            company="HIG", 
            year=2023, 
            section="MD&A", 
            subsection="full_text",
            chunk_id="HIG_2023_MDA_00", 
            word_count=15
        )
    ]
    
    print("Testing reasoner.py...")
    print(f"Sample promise: {sample_promise.promise_text}")
    print(f"Sample evidence: {len(sample_evidence)} chunks")
    
    # Test judge prompt building
    prompt = build_judge_prompt(sample_promise, sample_evidence)
    print(f"\nJudge prompt length: {len(prompt)} characters")
    print(f"Judge prompt preview: {prompt[:300]}...")
    
    # Test single promise judging (requires GROQ_API_KEY in .env)
    try:
        result = judge_single_promise(sample_promise, sample_evidence, api_key=None)
        print(f"\nJudgment result:")
        print(f"  Delivery score: {result.delivery_score}")
        print(f"  Status: {result.status}")
        print(f"  Reasoning: {result.judge_reasoning}")
        
    except ValueError as e:
        print(f"\nJudgment test skipped: {e}")
        print("To test judgment, set GROQ_API_KEY in .env file")
    except Exception as e:
        print(f"\nJudgment test failed: {e}")
    
    print("\nReasoner testing complete!")
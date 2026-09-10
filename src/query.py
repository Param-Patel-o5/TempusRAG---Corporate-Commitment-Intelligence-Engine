"""
Natural Language Query Processing for TempusRAG

Flow 2: Query rewriting, HyDE (hypothetical document expansion), retrieval,
and answer generation with citations. Optional feature on top of core pipeline.
"""

import logging
import re
from typing import Optional, List, Dict, Any

import google.generativeai as genai

from src.retriever import hybrid_retrieve
from src.models import Chunk
from src.config import EXTRACTOR_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Configure Gemini client at module level if API key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini client configured for query processing")
else:
    logger.warning(
        "No GEMINI_API_KEY found in config - API key must be provided at runtime "
        "for query rewriting and answer generation"
    )


def detect_followup_query(query: str, conversation_history: Optional[List[str]] = None) -> bool:
    """
    Detect if query is a follow-up that requires context from previous conversation.
    
    Args:
        query: User query text
        conversation_history: Previous queries/responses in conversation
        
    Returns:
        bool: True if likely a follow-up, False if standalone
    """
    query_lower = query.lower().strip()
    
    # Pronouns that indicate reference to prior context
    followup_indicators = [
        "it", "that", "this", "they", "them", "those", "these",
        "what about", "how about", "and that", "regarding that",
        "about it", "for that", "with that", "from that"
    ]
    
    # Check for pronoun patterns
    has_pronouns = any(indicator in query_lower for indicator in followup_indicators)
    
    # Additional heuristics
    starts_with_question_word_only = re.match(r'^(what|how|when|where|why|which)\s*\?*$', query_lower)
    very_short = len(query.split()) <= 3
    lacks_specifics = not re.search(r'\b(technology|claims|growth|finance|operations|revenue|profit|investment)\b', query_lower, re.IGNORECASE)
    
    is_followup = has_pronouns or (very_short and lacks_specifics) or starts_with_question_word_only
    
    logger.debug(f"Follow-up detection: {is_followup} (pronouns: {has_pronouns}, "
                f"very_short: {very_short}, lacks_specifics: {lacks_specifics}) - '{query}'")
    
    return is_followup


def rewrite_query(query: str, conversation_history: Optional[List[str]] = None, api_key: Optional[str] = None) -> str:
    """
    Rewrite follow-up queries to be standalone by expanding pronouns and implicit references.
    
    Args:
        query: Original query that may be a follow-up
        conversation_history: Previous queries/responses for context
        api_key: Optional Gemini API key override
        
    Returns:
        str: Rewritten standalone query, or original if not a follow-up
    """
    if not detect_followup_query(query, conversation_history):
        logger.info(f"Query is standalone, no rewriting needed: '{query}'")
        return query
    
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key:
        logger.warning("No Gemini API key available for query rewriting, returning original query")
        return query
    
    # Build context from conversation history
    context = ""
    if conversation_history:
        recent_context = conversation_history[-3:]  # Use last 3 exchanges
        context = "\n".join(f"Previous: {item}" for item in recent_context)
    
    # Build rewriting prompt
    prompt = f"""You are helping rewrite a follow-up question to be standalone and complete.

{"CONVERSATION CONTEXT:" if context else ""}
{context}

FOLLOW-UP QUESTION: "{query}"

TASK: Rewrite this follow-up question to be a complete, standalone question that doesn't rely on pronouns or implicit references. Include any implied context from the conversation.

Examples:
- "What about that?" → "What are the financial targets mentioned previously?"
- "How about the timeline?" → "What is the timeline for the technology investments discussed?"
- "And the results?" → "What were the results of the initiatives mentioned earlier?"

RESPONSE: Provide only the rewritten standalone question, nothing else."""

    try:
        # Reconfigure with provided API key
        genai.configure(api_key=resolved_api_key)
        
        model = genai.GenerativeModel(EXTRACTOR_MODEL)
        response = model.generate_content(prompt)
        
        rewritten = response.text.strip().strip('"').strip("'")
        
        logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
        
    except Exception as e:
        logger.warning(f"Query rewriting failed: {e}, using original query")
        return query


def generate_hypothetical_answer(query: str, api_key: Optional[str] = None) -> str:
    """
    Generate a hypothetical answer for HyDE (Hypothetical Document Expansion).
    
    Args:
        query: User query to generate hypothetical answer for
        api_key: Optional Gemini API key override
        
    Returns:
        str: Hypothetical answer text for improved retrieval
    """
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key:
        logger.warning("No Gemini API key available for HyDE, returning empty hypothetical")
        return ""
    
    prompt = f"""Generate a hypothetical answer to this question in 2-3 sentences, as if it came from a corporate SEC filing:

Question: {query}

Write a plausible answer mentioning specific business areas, initiatives, or commitments that a company might discuss in their annual report. Be concrete and specific."""

    try:
        # Reconfigure with provided API key
        genai.configure(api_key=resolved_api_key)
        
        model = genai.GenerativeModel(EXTRACTOR_MODEL)
        response = model.generate_content(prompt)
        
        hypothetical = response.text.strip()
        
        logger.debug(f"Generated hypothetical answer: '{hypothetical}'")
        return hypothetical
        
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}, returning empty hypothetical")
        return ""


def should_use_hyde(query: str) -> bool:
    """
    Determine if HyDE would improve retrieval for this query.
    
    Args:
        query: User query text
        
    Returns:
        bool: True if HyDE would help, False if query is already specific
    """
    query_lower = query.lower()
    
    # Signs that query is already specific (don't need HyDE)
    has_specific_terms = any(term in query_lower for term in [
        "technology", "claims", "growth", "finance", "operations",
        "revenue", "profit", "investment", "digital", "platform",
        "strategy", "market", "customer", "employee", "regulatory"
    ])
    
    has_numbers = re.search(r'\d+', query)
    has_years = re.search(r'\b(20\d{2}|202[0-9])\b', query)
    has_percentages = re.search(r'\d+%', query)
    
    is_specific = has_specific_terms or has_numbers or has_years or has_percentages
    
    # Signs that query is vague (would benefit from HyDE)
    is_vague = len(query.split()) <= 5
    has_general_words = any(word in query_lower for word in [
        "about", "regarding", "concerning", "what", "how", "tell me"
    ])
    
    use_hyde = (is_vague or has_general_words) and not is_specific
    
    logger.debug(f"HyDE decision: {use_hyde} (specific: {is_specific}, vague: {is_vague}) - '{query}'")
    
    return use_hyde


def generate_answer_with_citations(chunks: List[Chunk], query: str, api_key: Optional[str] = None) -> str:
    """
    Generate an answer to the query based on retrieved chunks with citations.
    
    Args:
        chunks: Retrieved chunks that may contain relevant information
        query: User query to answer
        api_key: Optional Gemini API key override
        
    Returns:
        str: Generated answer with embedded citations
    """
    if not chunks:
        return "No relevant information found in the available SEC filings."
    
    # Resolve API key
    resolved_api_key = api_key if api_key is not None else GEMINI_API_KEY
    if not resolved_api_key:
        raise ValueError(
            "No Gemini API key available for answer generation - "
            "set GEMINI_API_KEY in config or pass one explicitly"
        )
    
    # Format chunks for the prompt
    chunk_texts = []
    for i, chunk in enumerate(chunks, 1):
        chunk_texts.append(f"[Chunk {i}: {chunk.text}]")
    
    chunks_context = "\n\n".join(chunk_texts)
    
    # Build answer generation prompt
    prompt = f"""Based on these SEC filing excerpts, answer the user's question with specific citations.

SEC FILING EXCERPTS:
{chunks_context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer the question based only on the information in the excerpts
2. Include citations in this format: (Source: {chunks[0].chunk_id if chunks else 'CHUNK_ID'}, Year {chunks[0].year if chunks else 'YEAR'})
3. Keep citations brief and natural within the text
4. If the excerpts don't contain enough information to fully answer the question, say so
5. Be specific and quote relevant details from the filings when possible

ANSWER:"""

    try:
        # Reconfigure with provided API key
        genai.configure(api_key=resolved_api_key)
        
        model = genai.GenerativeModel(EXTRACTOR_MODEL)
        response = model.generate_content(prompt)
        
        answer = response.text.strip()
        
        logger.info(f"Generated answer with citations: {answer[:100]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise Exception(f"Failed to generate answer: {e}")


def process_query(
    query: str, 
    ticker: str, 
    conversation_history: Optional[List[str]] = None, 
    use_hyde: Optional[bool] = None, 
    api_key: Optional[str] = None, 
    top_n: int = 5
) -> Dict[str, Any]:
    """
    Main orchestrator for Flow 2 - natural language query processing.
    
    Args:
        query: User query text
        ticker: Company ticker symbol for retrieval
        conversation_history: Previous queries/responses for context
        use_hyde: Force HyDE on/off (None = auto-detect)
        api_key: Optional Gemini API key override
        top_n: Number of chunks to retrieve
        
    Returns:
        dict: Query processing results with keys:
            - query: Original query
            - rewritten_query: Rewritten query (if was follow-up)
            - used_hyde: Whether HyDE was used
            - hypothetical_answer: Hypothetical answer (if HyDE used)
            - answer: Final generated answer with citations
            - source_chunks: Retrieved chunks used for answer
    """
    logger.info(f"Processing query: '{query}' for ticker {ticker}")
    
    # Step 1: Detect and rewrite follow-up queries
    is_followup = detect_followup_query(query, conversation_history)
    rewritten_query = query
    
    if is_followup:
        logger.info("Detected follow-up query, attempting rewrite")
        rewritten_query = rewrite_query(query, conversation_history, api_key)
    else:
        logger.info("Standalone query, no rewriting needed")
    
    # Step 2: Decide on HyDE usage
    if use_hyde is None:
        use_hyde = should_use_hyde(rewritten_query)
    
    # Step 3: Generate hypothetical answer if using HyDE
    hypothetical_answer = ""
    retrieval_query = rewritten_query
    
    if use_hyde:
        logger.info("Using HyDE for query expansion")
        hypothetical_answer = generate_hypothetical_answer(rewritten_query, api_key)
        if hypothetical_answer:
            retrieval_query = f"{rewritten_query} {hypothetical_answer}"
    else:
        logger.info("Not using HyDE, proceeding with direct retrieval")
    
    # Step 4: Retrieve relevant chunks
    logger.info(f"Retrieving {top_n} chunks for: '{retrieval_query[:100]}...'")
    try:
        chunks = hybrid_retrieve(retrieval_query, ticker, top_n=top_n)
        logger.info(f"Retrieved {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        chunks = []
    
    # Step 5: Generate answer with citations
    logger.info("Generating answer with citations")
    try:
        answer = generate_answer_with_citations(chunks, rewritten_query, api_key)
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        answer = f"Failed to generate answer: {e}"
    
    # Step 6: Compile results
    results = {
        "query": query,
        "rewritten_query": rewritten_query if is_followup else None,
        "used_hyde": use_hyde,
        "hypothetical_answer": hypothetical_answer if use_hyde else None,
        "answer": answer,
        "source_chunks": chunks
    }
    
    logger.info(f"Query processing complete: {len(answer)} char answer, {len(chunks)} sources")
    
    return results


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    # Simulate a conversation
    conversation = []
    
    # Query 1: Standalone
    q1 = "What technology investments did Hartford commit to?"
    result1 = process_query(q1, ticker="HIG", conversation_history=conversation)
    print(f"Q1: {q1}")
    print(f"Answer: {result1['answer']}\n")
    conversation.append(q1)
    
    # Query 2: Follow-up
    q2 = "What about the timeline for that?"
    result2 = process_query(q2, ticker="HIG", conversation_history=conversation)
    print(f"Q2 (original): {q2}")
    print(f"Q2 (rewritten): {result2['rewritten_query']}")
    print(f"Answer: {result2['answer']}\n")
    conversation.append(q2)
    
    # Query 3: Vague query that should use HyDE
    q3 = "Tell me about growth"
    result3 = process_query(q3, ticker="HIG", conversation_history=conversation)
    print(f"Q3: {q3}")
    print(f"Used HyDE: {result3['used_hyde']}")
    if result3['hypothetical_answer']:
        print(f"Hypothetical: {result3['hypothetical_answer']}")
    print(f"Answer: {result3['answer']}\n")
    
    print("Query processing demo complete!")
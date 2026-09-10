"""Hybrid search + RRF fusion + section boosting + lost-in-middle reordering.

Combines dense (ChromaDB) and sparse (BM25) retrieval into a single ranked list of chunks.
Applies section-based score boosting and lost-in-middle reordering for optimal LLM performance.
"""

import logging
import numpy as np

from src.config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
    FINAL_TOP_N,
    SECTION_BOOST_WEIGHTS,
    DEFAULT_SECTION_BOOST
)
from src.embedder import get_chroma_collection, get_bm25_index, tokenize_for_bm25, _get_embedding_model
from src.models import Chunk

logger = logging.getLogger(__name__)


def dense_search(query: str, ticker: str, top_k: int = DENSE_TOP_K) -> list[tuple[str, float]]:
    """
    Perform dense semantic search using ChromaDB.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_k: Number of results to return.
        
    Returns:
        list[tuple[str, float]]: List of (chunk_id, score) tuples, ranked best-first.
    """
    try:
        # Get ChromaDB collection
        collection = get_chroma_collection(ticker)
        
        # Embed the query
        model = _get_embedding_model()
        query_embedding = model.encode([query], show_progress_bar=False)[0]
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["distances", "metadatas", "documents"]
        )
        
        # Extract results - ChromaDB returns distances (smaller = more similar)
        # Convert to similarity scores (higher = better) for consistent ranking
        chunk_ids = results["ids"][0]
        distances = results["distances"][0]
        
        # Convert distances to similarity scores (1 / (1 + distance))
        # NOTE: This assumes distances are non-negative (true for cosine distance, ChromaDB's default).
        # If ChromaDB's distance metric is changed to something that can be negative, 
        # this conversion will break and needs to be updated accordingly.
        similarities = [1.0 / (1.0 + dist) for dist in distances]
        
        # Combine and sort by similarity (descending)
        ranked_results = list(zip(chunk_ids, similarities))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Dense search for '{query}' in {ticker}: {len(ranked_results)} results")
        return ranked_results
        
    except Exception as e:
        logger.error(f"Dense search failed for {ticker}: {e}")
        return []


def sparse_search(query: str, ticker: str, top_k: int = SPARSE_TOP_K) -> list[tuple[str, float]]:
    """
    Perform sparse keyword search using BM25.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_k: Number of results to return.
        
    Returns:
        list[tuple[str, float]]: List of (chunk_id, score) tuples, ranked best-first.
    """
    # Get BM25 index
    bm25_entry = get_bm25_index(ticker)
    if bm25_entry is None:
        logger.error(f"No BM25 index found for {ticker}, falling back to dense-only search")
        return []
    
    bm25_index = bm25_entry["bm25"]
    chunks = bm25_entry["chunks"]
    
    # Tokenize query
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        logger.warning(f"Query '{query}' tokenized to empty list")
        return []
    
    # Get BM25 scores
    scores = bm25_index.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Map indices to chunk_ids and scores
    results = []
    for idx in top_indices:
        if idx < len(chunks) and scores[idx] > 0:  # Only include positive scores
            chunk_id = chunks[idx].chunk_id
            score = float(scores[idx])
            results.append((chunk_id, score))
    
    logger.debug(f"Sparse search for '{query}' in {ticker}: {len(results)} results")
    return results


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]], 
    sparse_results: list[tuple[str, float]], 
    k: int = RRF_K
) -> list[tuple[str, float]]:
    """
    Fuse dense and sparse results using Reciprocal Rank Fusion.
    
    Args:
        dense_results: Results from dense search (chunk_id, score).
        sparse_results: Results from sparse search (chunk_id, score).
        k: RRF parameter (typically 60).
        
    Returns:
        list[tuple[str, float]]: Fused results (chunk_id, fused_score), ranked best-first.
    """
    # Build rank mappings (1-indexed)
    dense_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(dense_results)}
    sparse_ranks = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(sparse_results)}
    
    # Get all unique chunk_ids
    all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    
    # Compute fused scores
    fused_results = []
    for chunk_id in all_chunk_ids:
        dense_rank = dense_ranks.get(chunk_id, float('inf'))
        sparse_rank = sparse_ranks.get(chunk_id, float('inf'))
        
        # RRF formula: 1/(k + rank)
        dense_score = 1.0 / (k + dense_rank) if dense_rank != float('inf') else 0.0
        sparse_score = 1.0 / (k + sparse_rank) if sparse_rank != float('inf') else 0.0
        
        fused_score = dense_score + sparse_score
        fused_results.append((chunk_id, fused_score))
    
    # Sort by fused score (descending)
    fused_results.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse -> {len(fused_results)} fused")
    return fused_results


def apply_section_boost(
    ranked_chunk_ids: list[tuple[str, float]], 
    chunk_lookup: dict[str, Chunk]
) -> list[tuple[str, float]]:
    """
    Apply section-based score boosting.
    
    Args:
        ranked_chunk_ids: List of (chunk_id, score) tuples.
        chunk_lookup: Mapping from chunk_id to Chunk objects.
        
    Returns:
        list[tuple[str, float]]: Boosted results (chunk_id, boosted_score), ranked best-first.
    """
    boosted_results = []
    
    for chunk_id, score in ranked_chunk_ids:
        if chunk_id not in chunk_lookup:
            logger.warning(f"Chunk ID {chunk_id} not found in chunk lookup, skipping")
            continue
        
        chunk = chunk_lookup[chunk_id]
        section = chunk.section
        
        # Get boost weight
        boost_weight = SECTION_BOOST_WEIGHTS.get(section, DEFAULT_SECTION_BOOST)
        boosted_score = score * boost_weight
        
        boosted_results.append((chunk_id, boosted_score))
        logger.debug(f"Boosted {chunk_id} (section: {section}): {score:.4f} -> {boosted_score:.4f} (boost: {boost_weight})")
    
    # Re-sort by boosted score
    boosted_results.sort(key=lambda x: x[1], reverse=True)
    
    return boosted_results


def lost_in_middle_reorder(ranked_chunk_ids: list[str]) -> list[str]:
    """
    Reorder chunks using lost-in-middle strategy.
    
    Places most relevant chunks at beginning and end, with less relevant in the middle.
    Algorithm: rank 1 -> position 1, rank 2 -> last position, rank 3+ fill middle sequentially.
    
    Args:
        ranked_chunk_ids: List of chunk IDs in rank order.
        
    Returns:
        list[str]: Reordered chunk IDs optimized for LLM performance.
    """
    n = len(ranked_chunk_ids)
    
    if n == 0:
        return []
    elif n == 1:
        return ranked_chunk_ids.copy()
    elif n == 2:
        return ranked_chunk_ids.copy()  # rank 1 at pos 0, rank 2 at pos 1
    
    # For n >= 3: rank 1 -> pos 0, rank 2 -> pos n-1, ranks 3+ fill middle
    reordered = [None] * n
    
    # Place rank 1 (most relevant) at beginning
    reordered[0] = ranked_chunk_ids[0]
    
    # Place rank 2 (second most relevant) at end
    reordered[n - 1] = ranked_chunk_ids[1]
    
    # Fill middle positions with remaining ranks in order
    middle_start = 1
    for i in range(2, n):
        reordered[middle_start] = ranked_chunk_ids[i]
        middle_start += 1
    
    logger.debug(f"Lost-in-middle reordering: {n} chunks reordered")
    return reordered


def build_chunk_lookup(ticker: str) -> dict[str, Chunk]:
    """
    Build mapping from chunk_id to Chunk objects for a ticker.
    
    Args:
        ticker: Company ticker symbol.
        
    Returns:
        dict[str, Chunk]: Mapping from chunk_id to Chunk objects.
    """
    bm25_entry = get_bm25_index(ticker)
    if bm25_entry is None:
        logger.error(f"No BM25 index found for {ticker}, cannot build chunk lookup")
        return {}
    
    chunks = bm25_entry["chunks"]
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    
    logger.debug(f"Built chunk lookup for {ticker}: {len(chunk_lookup)} chunks")
    return chunk_lookup


def hybrid_retrieve(query: str, ticker: str, top_n: int = FINAL_TOP_N) -> list[Chunk]:
    """
    Perform hybrid retrieval with RRF fusion, boosting, and reordering.
    
    Main orchestration function that combines dense and sparse search results,
    applies RRF fusion, section boosting, and lost-in-middle reordering.
    
    Args:
        query: Search query string.
        ticker: Company ticker symbol.
        top_n: Number of final results to return.
        
    Returns:
        list[Chunk]: Final ranked and reordered chunks, length <= top_n.
    """
    logger.info(f"Hybrid retrieval for '{query}' in {ticker}, requesting top {top_n}")
    
    # Build chunk lookup first (needed for boosting and final mapping)
    chunk_lookup = build_chunk_lookup(ticker)
    if not chunk_lookup:
        logger.warning(f"No chunks available for {ticker}")
        return []
    
    # Parallel dense and sparse search
    dense_results = dense_search(query, ticker)
    sparse_results = sparse_search(query, ticker)
    
    if not dense_results and not sparse_results:
        logger.warning(f"Both dense and sparse search returned no results for '{query}' in {ticker}")
        return []
    
    # Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
    
    if not fused_results:
        logger.warning(f"RRF fusion resulted in no results for '{query}' in {ticker}")
        return []
    
    # Apply section boosting
    boosted_results = apply_section_boost(fused_results, chunk_lookup)
    
    # Take top N by boosted score
    top_boosted = boosted_results[:top_n]
    top_chunk_ids = [chunk_id for chunk_id, _ in top_boosted]
    
    # Apply lost-in-middle reordering
    reordered_chunk_ids = lost_in_middle_reorder(top_chunk_ids)
    
    # Map back to Chunk objects
    final_chunks = []
    for chunk_id in reordered_chunk_ids:
        if chunk_id in chunk_lookup:
            final_chunks.append(chunk_lookup[chunk_id])
        else:
            logger.warning(f"Chunk ID {chunk_id} missing from lookup during final mapping")
    
    final_chunk_ids = [chunk.chunk_id for chunk in final_chunks]
    logger.info(f"Hybrid retrieval complete for '{query}' in {ticker}: {final_chunk_ids}")
    
    return final_chunks


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Test hybrid retrieval (assumes HIG has been embedded via embedder.py)
    query = "technology investments and digital claims platform"
    ticker = "HIG"
    
    print(f"Testing hybrid retrieval...")
    print(f"Query: '{query}'")
    print(f"Ticker: {ticker}")
    print()
    
    try:
        results = hybrid_retrieve(query, ticker=ticker, top_n=5)
        
        print(f"Retrieved {len(results)} chunks:")
        print()
        
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk.chunk_id}")
            print(f"   Section: {chunk.section}")
            if chunk.subsection != "full_text":
                print(f"   Subsection: {chunk.subsection}")
            print(f"   Text: {chunk.text[:100]}...")
            print()
        
        # Test individual components
        print("--- Component Testing ---")
        
        # Test dense search
        dense_results = dense_search(query, ticker, top_k=3)
        print(f"Dense search: {len(dense_results)} results")
        for chunk_id, score in dense_results[:2]:
            print(f"  {chunk_id}: {score:.4f}")
        
        # Test sparse search
        sparse_results = sparse_search(query, ticker, top_k=3)
        print(f"Sparse search: {len(sparse_results)} results")
        for chunk_id, score in sparse_results[:2]:
            print(f"  {chunk_id}: {score:.4f}")
        
        # Test RRF fusion
        if dense_results or sparse_results:
            fused = reciprocal_rank_fusion(dense_results, sparse_results)
            print(f"RRF fusion: {len(fused)} results")
            for chunk_id, score in fused[:2]:
                print(f"  {chunk_id}: {score:.4f}")
        
        # Test lost-in-middle reordering
        test_ids = ['A', 'B', 'C', 'D', 'E']
        reordered = lost_in_middle_reorder(test_ids)
        print(f"Lost-in-middle test: {test_ids} -> {reordered}")
        
        print("\nHybrid retrieval test complete!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
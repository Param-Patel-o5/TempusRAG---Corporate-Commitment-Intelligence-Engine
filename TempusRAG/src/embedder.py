"""Dense + sparse embeddings + ChromaDB storage for TempusRAG chunks.

Handles sentence-transformer encoding to dense vectors stored in ChromaDB (persistent),
and in-memory BM25 index construction for sparse retrieval. Implements caching/staleness
checks to avoid re-embedding fresh collections.
"""

import logging
import re
import time
from datetime import datetime, timezone

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.config import CHROMADB_PATH, EMBEDDING_MODEL_NAME, CACHE_STALENESS_DAYS
from src.models import Chunk

logger = logging.getLogger(__name__)

# Module-level state
_embedding_model: SentenceTransformer | None = None
_chroma_client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
_bm25_indices: dict[str, dict] = {}


def _get_embedding_model() -> SentenceTransformer:
    """
    Get the sentence transformer model singleton.
    
    Loads the model on first call and caches for subsequent calls.
    Model loading takes a few seconds and should only happen once.
    
    Returns:
        SentenceTransformer: The cached embedding model instance.
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}")
        start_time = time.time()
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
    
    return _embedding_model


def collection_exists_and_fresh(ticker: str) -> bool:
    """
    Check if a ChromaDB collection exists and is fresh (within staleness threshold).
    
    Args:
        ticker: Company ticker symbol (e.g. "HIG").
        
    Returns:
        bool: True if collection exists and is fresh, False otherwise.
    """
    collection_name = ticker.upper()
    
    try:
        # Check if collection exists
        collection = _chroma_client.get_collection(name=collection_name)
        
        # Get metadata to check ingestion timestamp
        metadata = collection.metadata
        
        if not metadata or 'ingested_at' not in metadata:
            logger.info(f"Collection {collection_name} exists but has no ingestion timestamp, will re-ingest")
            return False
        
        # Parse the ingestion timestamp
        ingested_at_str = metadata['ingested_at']
        try:
            ingested_at = datetime.fromisoformat(ingested_at_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid ingested_at timestamp for {collection_name}, will re-ingest")
            return False
        
        # Check if it's within staleness threshold
        now = datetime.now(timezone.utc)
        age_days = (now - ingested_at).days
        
        if age_days <= CACHE_STALENESS_DAYS:
            logger.info(f"Collection {collection_name} is fresh (ingested {age_days} days ago)")
            return True
        else:
            logger.info(f"Collection {collection_name} is stale (ingested {age_days} days ago), will re-ingest")
            return False
            
    except Exception:
        logger.info(f"No collection found for {collection_name}")
        return False


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Simple tokenizer for BM25 indexing.
    
    Strips punctuation, converts to lowercase, and splits on whitespace.
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        list[str]: List of tokens.
    """
    # Strip basic punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    
    # Split on whitespace and filter empty strings
    tokens = [token for token in cleaned.split() if token.strip()]
    
    return tokens


def embed_and_store_chunks(chunks: list[Chunk], ticker: str, years_covered: list[int]) -> None:
    """
    Embed chunks and store in both ChromaDB and BM25 index.
    
    IMPORTANT: This function handles both fresh ingestion and BM25 rebuilding on cache hits.
    When a collection is fresh (cache hit), ChromaDB embedding is skipped but BM25 index
    is still rebuilt from the provided chunks list. This means callers must still run
    chunking and pass chunks even on cache hits.
    
    Args:
        chunks: List of Chunk objects to embed and store.
        ticker: Company ticker symbol.
        years_covered: List of years that the chunks cover.
    """
    if not chunks:
        logger.warning(f"Empty chunks list provided for {ticker}, skipping embedding")
        return
    
    ticker_upper = ticker.upper()
    collection_name = ticker_upper
    
    # Check if we need to rebuild BM25 index even if collection is fresh
    is_fresh = collection_exists_and_fresh(ticker)
    needs_bm25_rebuild = ticker_upper not in _bm25_indices
    
    if is_fresh and not needs_bm25_rebuild:
        logger.info(f"Collection {collection_name} is fresh and BM25 index exists, skipping all work")
        return
    
    if is_fresh and needs_bm25_rebuild:
        logger.info(f"Collection {collection_name} is fresh but BM25 index missing, rebuilding BM25 only")
        
        # Build BM25 index only
        tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        
        _bm25_indices[ticker_upper] = {
            "bm25": bm25_index,
            "chunks": chunks.copy()
        }
        
        logger.info(f"BM25 index rebuilt for {ticker_upper} with {len(chunks)} chunks")
        return
    
    # Full ingestion needed (collection is stale or doesn't exist)
    logger.info(f"Starting full embedding and storage for {ticker_upper} with {len(chunks)} chunks")
    start_time = time.time()
    
    # Get or create collection
    try:
        collection = _chroma_client.get_collection(name=collection_name)
        logger.info(f"Using existing collection {collection_name}")
    except Exception:
        collection = _chroma_client.create_collection(name=collection_name)
        logger.info(f"Created new collection {collection_name}")
    
    # Get embedding model
    model = _get_embedding_model()
    
    # Batch encode all chunk texts
    logger.info(f"Encoding {len(chunks)} chunks with {EMBEDDING_MODEL_NAME}")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=False)
    
    # Prepare data for ChromaDB upsert
    ids = [chunk.chunk_id for chunk in chunks]
    documents = chunk_texts
    metadatas = []
    
    for chunk in chunks:
        metadata = {
            "company": chunk.company,
            "year": chunk.year,
            "section": chunk.section,
            "subsection": chunk.subsection,
            "word_count": chunk.word_count
        }
        metadatas.append(metadata)
    
    # Upsert to ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )
    
    # Update collection metadata with ingestion timestamp and years covered
    now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    years_str = ','.join(map(str, sorted(years_covered)))
    
    collection.modify(metadata={
        "ingested_at": now_iso,
        "years_covered": years_str
    })
    
    # Build BM25 index
    tokenized_chunks = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    _bm25_indices[ticker_upper] = {
        "bm25": bm25_index,
        "chunks": chunks.copy()
    }
    
    elapsed_time = time.time() - start_time
    logger.info(
        f"Embedded and stored {len(chunks)} chunks for {ticker_upper} in {elapsed_time:.2f}s "
        f"(ChromaDB count: {collection.count()})"
    )


def get_bm25_index(ticker: str) -> dict | None:
    """
    Get the BM25 index for a ticker.
    
    Args:
        ticker: Company ticker symbol.
        
    Returns:
        dict: BM25 index entry with "bm25" and "chunks" keys, or None if not found.
    """
    ticker_upper = ticker.upper()
    entry = _bm25_indices.get(ticker_upper)
    
    if entry is None:
        logger.warning(
            f"No BM25 index found for {ticker_upper}. "
            f"This may happen if the collection was fresh and embed_and_store_chunks was not called. "
            f"Available indices: {list(_bm25_indices.keys())}"
        )
    
    return entry


def get_chroma_collection(ticker: str):
    """
    Get the ChromaDB collection for a ticker.
    
    Args:
        ticker: Company ticker symbol.
        
    Returns:
        chromadb.Collection: The ChromaDB collection.
        
    Raises:
        Exception: If collection doesn't exist (lets ChromaDB's native exception propagate).
    """
    collection_name = ticker.upper()
    
    try:
        collection = _chroma_client.get_collection(name=collection_name)
        return collection
    except Exception as e:
        logger.error(f"Failed to get ChromaDB collection {collection_name}: {e}")
        raise


if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # Create sample chunks for testing
    sample_chunks = [
        Chunk(
            text="[Hartford | 2021 | MD&A > Technology] We are investing heavily in our digital claims platform.",
            company="HIG",
            year=2021,
            section="MD&A",
            subsection="Technology",
            chunk_id="HIG_2021_MDA_TECH_00",
            word_count=15
        ),
        Chunk(
            text="[Hartford | 2021 | Risk Factors] Our business faces cybersecurity risks.",
            company="HIG",
            year=2021,
            section="Risk Factors",
            subsection="full_text",
            chunk_id="HIG_2021_RISK_00",
            word_count=10
        ),
        Chunk(
            text="[Hartford | 2021 | MD&A > Claims] Claims processing efficiency improved significantly this year.",
            company="HIG",
            year=2021,
            section="MD&A", 
            subsection="Claims",
            chunk_id="HIG_2021_MDA_CLAI_00",
            word_count=12
        )
    ]
    
    print("Testing embedder.py functionality...")
    print(f"Sample chunks: {len(sample_chunks)}")
    
    # Test embedding and storage
    embed_and_store_chunks(sample_chunks, ticker="HIG", years_covered=[2021])
    
    # Test ChromaDB retrieval
    try:
        coll = get_chroma_collection("HIG")
        print(f"ChromaDB collection count: {coll.count()}")
        
        # Show collection metadata
        metadata = coll.metadata
        if metadata:
            print(f"Collection metadata: {metadata}")
        
    except Exception as e:
        print(f"ChromaDB test failed: {e}")
    
    # Test BM25 index
    bm25_entry = get_bm25_index("HIG")
    if bm25_entry:
        print(f"BM25 chunks indexed: {len(bm25_entry['chunks'])}")
        
        # Test a simple BM25 search
        bm25_index = bm25_entry["bm25"]
        query = tokenize_for_bm25("digital claims platform")
        scores = bm25_index.get_scores(query)
        print(f"BM25 search scores for 'digital claims platform': {scores}")
        
    else:
        print("BM25 index not found")
    
    # Test tokenizer
    test_text = "We are investing in digital transformation! This includes AI-powered solutions."
    tokens = tokenize_for_bm25(test_text)
    print(f"Tokenized '{test_text}' -> {tokens}")
    
    # Test collection freshness check
    is_fresh = collection_exists_and_fresh("HIG")
    print(f"Collection HIG is fresh: {is_fresh}")
    
    print("\nEmbedder testing complete!")
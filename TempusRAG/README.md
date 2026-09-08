# TempusRAG

**TempusRAG** is an agentic RAG pipeline that performs multi-hop retrieval across temporally-indexed SEC 10-K filings, extracting forward-looking management commitments using structured LLM output, then reasoning across subsequent fiscal years to classify each promise as Delivered, Partial, or Silently Abandoned — producing a quantified executive credibility score (0–100).

Demonstrated on The Hartford (HIG), generalizable to any US public company via SEC EDGAR.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
# Copy .env and fill in your API keys
```

Fill in `.env` with your `GEMINI_API_KEY` and `GROQ_API_KEY`, then launch the dashboard:

```bash
streamlit run app.py
```

## File Structure

```
tempusrag/
├── data/filings/       # Cached SEC 10-K filings per ticker
├── chroma_store/       # Persistent ChromaDB collections (one per ticker)
├── logs/               # Application logs
├── src/
│   ├── models.py       # Pydantic schemas
│   ├── ingestion.py    # SEC EDGAR fetch + parsing
│   ├── chunker.py      # Section detection + chunking
│   ├── embedder.py     # Dense + sparse embeddings + ChromaDB
│   ├── retriever.py    # Hybrid search + RRF fusion
│   ├── extractor.py    # LLM Call 1 — Gemini promise extraction
│   ├── reasoner.py     # LLM Call 2 — Groq cross-year classification
│   ├── scorer.py       # Credibility scoring (0–100)
│   ├── query.py        # Flow 2 — query rewriting + HyDE
│   └── config.py       # Environment configuration
├── app.py              # Streamlit dashboard
├── pipeline.py         # End-to-end orchestrator
├── .env                # API keys and paths
├── requirements.txt    # Pinned dependencies
└── README.md
```

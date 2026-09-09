# TempusRAG: Corporate Credibility Analysis through SEC Filings

TempusRAG is an advanced AI-powered system that analyzes corporate credibility by extracting forward-looking promises from SEC filings and evaluating their delivery across subsequent years. The system combines natural language processing, cross-year reasoning, and retrieval-augmented generation to provide comprehensive credibility assessments.

## Overview

Corporate promises are often made in SEC filings but rarely tracked systematically. TempusRAG addresses this by:

1. **Extracting Promises**: Using LLMs to identify forward-looking commitments in SEC 10-K filings
2. **Cross-Year Analysis**: Searching subsequent filings for evidence of promise delivery
3. **Credibility Scoring**: Generating quantitative credibility scores across business domains
4. **Interactive Querying**: Enabling natural language questions about company performance

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Query Engine  │    │  Pipeline Core  │
│   (Streamlit)   │◄──►│   (RAG + LLM)   │◄──►│  (Orchestrator) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Vector Store  │◄──►│   Data Layer    │
                       │   (ChromaDB)    │    │ (SEC Filings)   │
                       └─────────────────┘    └─────────────────┘
```

### Processing Pipeline

The system implements a sophisticated 8-stage pipeline:

**Stage 1: SEC Filing Ingestion**
- Fetches company ticker-to-CIK mappings from SEC EDGAR
- Downloads 10-K filings for the last 5 years
- Parses HTML and extracts structured sections (MD&A, Risk Factors, etc.)
- Implements rate limiting to comply with SEC access requirements

**Stage 2: Document Chunking**
- Recursive text splitting with semantic awareness
- 500-token maximum chunks with 50-token overlap
- Preserves section context and company metadata
- Handles edge cases and prevents infinite loops

**Stage 3: Embedding Generation**
- Uses sentence-transformers (all-MiniLM-L6-v2) for dense embeddings
- Generates BM25 sparse representations for hybrid search
- Stores embeddings in persistent ChromaDB collections
- Implements caching and staleness detection

**Stage 4: Promise Extraction (LLM Call 1)**
- Uses Gemini 1.5 Flash for forward-looking statement identification
- Focuses on MD&A and Future Outlook sections
- Extracts structured Promise objects with domain classification
- Implements rate limiting and exponential backoff retry logic

**Stage 5: Cross-Year Reasoning (LLM Call 2)**
- Searches for delivery evidence in subsequent years using hybrid retrieval
- Uses Groq Compound model for judgment and scoring
- Implements async batching for performance
- Applies evidence similarity thresholds for classification

**Stage 6: Rule-Based Scoring**
- Calculates confidence scores using multiple signals
- Aggregates domain-level credibility metrics
- Applies recency weighting for temporal relevance
- Detects red flags based on delivery patterns

**Stage 7: Report Generation**
- Synthesizes analysis into CompanyCredibilityReport objects
- Provides overall credibility scores and domain breakdowns
- Includes detailed promise tracking and evidence links

**Stage 8: Interactive Querying**
- Implements RAG-based natural language interface
- Uses query rewriting and HyDE for improved retrieval
- Provides conversational context and follow-up support

## Technical Implementation

### Core Technologies

**Language Models**
- **Gemini 1.5 Flash**: Promise extraction and query processing
- **Groq Compound**: Cross-year reasoning and delivery judgment
- Rate limiting: 15 RPM for Gemini, 10 RPM for Groq (configurable)

**Vector Database**
- **ChromaDB**: Persistent storage for document embeddings
- **Sentence Transformers**: Dense embedding generation (384-dimensional)
- **BM25**: Sparse retrieval for keyword matching
- **Hybrid Search**: RRF fusion of dense and sparse results

**Web Framework**
- **Streamlit**: Modern web interface with glass morphism design
- **Real-time Analysis**: Progress tracking and error handling
- **API Key Management**: Session-only storage for security

**Data Processing**
- **SEC EDGAR API**: Automated filing ingestion
- **BeautifulSoup**: HTML parsing and section extraction
- **Pandas**: Data manipulation and analysis
- **Asyncio**: Concurrent processing for performance
### Project Structure

```
TempusRAG/
├── .env                          # API keys and configuration
├── app.py                        # Streamlit web application
├── pipeline.py                   # Pipeline testing script
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── chroma_store/                 # ChromaDB persistent storage
├── data/
│   └── filings/                  # Cached SEC filings (ticker_year.txt)
├── logs/                         # Application logs
└── src/                          # Core implementation modules
    ├── __init__.py
    ├── chunker.py                # Document chunking logic
    ├── config.py                 # Configuration management
    ├── embedder.py               # Embedding generation and storage
    ├── extractor.py              # Promise extraction (LLM Call 1)
    ├── ingestion.py              # SEC filing ingestion
    ├── models.py                 # Pydantic data models
    ├── pipeline.py               # End-to-end orchestration
    ├── query.py                  # Natural language querying (Flow 2)
    ├── reasoner.py               # Cross-year reasoning (LLM Call 2)
    ├── retriever.py              # Hybrid vector/sparse retrieval
    └── scorer.py                 # Rule-based credibility scoring
```

### Data Models

The system uses Pydantic models for type safety and validation:

**Promise**: Forward-looking commitment extracted from filings
- `promise_text`: The actual commitment text
- `domain`: Business domain (Technology, Claims, Growth, Finance, Operations)
- `deadline_mentioned`: Specific timeline if stated
- `year_made`: Fiscal year of the promise
- `confidence_score`: Automated confidence assessment

**DeliveryEvidence**: Cross-year judgment of promise fulfillment
- `promise`: Reference to original Promise object
- `evidence_text`: Supporting text from subsequent filings
- `delivery_score`: 0-100 scale judgment
- `status`: Delivered, Partial, Silently Abandoned, or Pending
- `judge_reasoning`: LLM explanation of the judgment

**CompanyCredibilityReport**: Final analysis output
- `overall_score`: Aggregate credibility score
- `domain_scores`: Performance by business domain
- `total_promises`: Count of extracted promises
- `delivered/partial/abandoned`: Status breakdown
- `red_flags`: Identified credibility concerns

### Hybrid Retrieval System

The retrieval system combines multiple approaches for optimal relevance:

**Dense Retrieval**
- Semantic similarity using sentence transformers
- Captures conceptual relationships beyond keyword matching
- 384-dimensional embedding space with cosine similarity

**Sparse Retrieval**
- BM25 algorithm for precise keyword matching
- Handles specific terms, numbers, and exact phrases
- Complementary to dense semantic search

**Fusion and Reranking**
- Reciprocal Rank Fusion (RRF) combines dense and sparse results
- Section-aware boosting (MD&A > Risk Factors > Other sections)
- Lost-in-middle reordering for optimal context presentation

### Rate Limiting and Resilience

**API Rate Management**
- Configurable request intervals per API provider
- Exponential backoff retry with jitter
- Maximum retry limits with graceful degradation

**Error Handling**
- Specific handling for 429 (rate limit), 401 (auth), network errors
- Graceful pipeline continuation despite individual component failures
- Comprehensive logging for debugging and monitoring

**Caching Strategy**
- SEC filing cache with 180-day staleness detection
- ChromaDB persistent storage for embeddings
- BM25 index auto-rebuild on data changes

## Configuration

### Environment Variables

```bash
# API Keys (required)
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
SEC_USER_AGENT=your_email@domain.com

# Storage Paths (optional)
CHROMADB_PATH=./chroma_store
DATA_DIR=./data/filings
LOG_DIR=./logs

# Rate Limiting (optional)
GEMINI_REQUESTS_PER_MINUTE=15
GROQ_REQUESTS_PER_MINUTE=10
MAX_RETRIES=3
```

### Tunable Parameters

The system exposes numerous configuration parameters in `src/config.py`:

**Chunking Parameters**
- `CHUNK_MAX_TOKENS`: Maximum tokens per chunk (default: 500)
- `CHUNK_OVERLAP_TOKENS`: Overlap between chunks (default: 50)

**Retrieval Parameters**
- `DENSE_TOP_K`: Top-K results from dense search (default: 20)
- `SPARSE_TOP_K`: Top-K results from sparse search (default: 20)
- `RRF_K`: Reciprocal rank fusion parameter (default: 60)
- `FINAL_TOP_N`: Final results returned (default: 5)

**Scoring Thresholds**
- `SIMILARITY_STRONG_EVIDENCE`: Strong delivery evidence threshold (default: 0.7)
- `SIMILARITY_WEAK_EVIDENCE`: Weak evidence threshold (default: 0.5)

**Section Prioritization**
- `SECTION_BOOST_WEIGHTS`: Relevance multipliers by section type
- `PROMISE_EXTRACTION_SECTIONS`: Sections to search for promises
## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- API keys for Gemini and Groq (free tiers available)
- Internet connection for SEC filing access

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/TempusRAG.git
   cd TempusRAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd TempusRAG
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_gemini_key_here" > .env
   echo "GROQ_API_KEY=your_groq_key_here" >> .env
   echo "SEC_USER_AGENT=your_email@domain.com" >> .env
   ```

5. **Launch the application**
   ```bash
   streamlit run app.py
   ```

   The web interface will be available at `http://localhost:8501`

### API Key Setup

**Gemini API Key**
- Visit: https://aistudio.google.com/app/apikey
- Create a new API key (free tier: 20 requests/day)
- Add to `.env` file as `GEMINI_API_KEY`

**Groq API Key**
- Visit: https://console.groq.com/keys
- Create a new API key (free tier: 70K tokens/minute)
- Add to `.env` file as `GROQ_API_KEY`

### Testing the Pipeline

Run the integrated test on sample companies:

```bash
python pipeline.py  # Tests on Hartford (HIG)
```

Or test specific components:

```bash
python src/test_integration.py  # Full integration test suite
```

## Usage

### Web Interface

1. **Enter Company Ticker**: Input a valid stock ticker (e.g., HIG, NVDA, AAPL)
2. **Run Analysis**: Click "Analyze" to start the pipeline (1-2 minutes)
3. **Review Results**: Examine credibility score, domain performance, and red flags
4. **Ask Questions**: Use the chat interface for natural language queries

### Command Line Interface

```python
from src.pipeline import run_tempusrag_pipeline

# Analyze a company
report = run_tempusrag_pipeline(
    ticker="HIG",
    company_display_name="The Hartford",
    force_reingest=False
)

print(f"Overall Score: {report.overall_score}")
print(f"Total Promises: {report.total_promises}")
print(f"Delivered: {report.delivered}")
```

### Natural Language Querying

```python
from src.query import process_query

# Ask questions about the analysis
result = process_query(
    query="What technology investments did the company promise?",
    ticker="HIG"
)

print(result['answer'])
print(f"Sources: {len(result['source_chunks'])} chunks")
```

## Performance Characteristics

### Processing Times
- **First Analysis**: 2-5 minutes (includes filing download and embedding)
- **Subsequent Analyses**: 30-60 seconds (cached embeddings)
- **Individual Queries**: 3-10 seconds (depending on complexity)

### Resource Requirements
- **Memory**: 1-2 GB RAM for typical analysis
- **Storage**: 100-500 MB per company (filings + embeddings)
- **Network**: Moderate bandwidth for SEC filing downloads

### Scalability Considerations
- **Concurrent Users**: Limited by API rate limits (10-15 RPM)
- **Company Coverage**: Supports all SEC-registered companies
- **Historical Depth**: Configurable (default: 5 years of filings)

## Limitations and Future Work

### Current Limitations

**Data Coverage**
- Limited to SEC 10-K filings (annual reports)
- English-language companies only
- Focuses on specific sections (MD&A, Future Outlook)

**Analysis Scope**
- Promise extraction may miss subtle commitments
- Cross-year reasoning limited to subsequent filings
- Domain classification uses predefined categories

**Technical Constraints**
- API rate limits affect processing speed
- Embedding model fixed at training time
- No real-time filing updates

### Future Enhancements

**Extended Data Sources**
- 10-Q quarterly reports for more frequent analysis
- Earnings call transcripts for informal promises
- Press releases and investor presentations

**Advanced Analytics**
- Sentiment analysis of promise language
- Market reaction correlation with delivery
- Peer comparison and industry benchmarking

**Technical Improvements**
- Streaming analysis for large companies
- Multi-language support for international filings
- Real-time filing monitoring and alerts

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for:

- Bug fixes and performance improvements
- New data sources and analysis methods
- Enhanced UI/UX features
- Documentation improvements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SEC EDGAR database for public company filings
- Google Gemini for advanced language understanding
- Groq for high-speed inference
- ChromaDB for efficient vector storage
- Streamlit for rapid web application development

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the development team.

---

**Disclaimer**: TempusRAG is an analytical tool for research purposes. It should not be used as the sole basis for investment decisions. Always consult qualified financial advisors and conduct your own due diligence before making investment choices.
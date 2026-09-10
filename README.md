# TempusRAG

**Corporate Commitment Intelligence Engine** — an agentic, temporal RAG system that reads SEC **10-Ks** (EDGAR, not glossy annual-report PDFs), extracts management’s forward-looking commitments, and checks whether later filings show those commitments were delivered.

Demo issuer: **The Hartford (HIG)**. No LangChain / LangGraph.

Companies make plans in Item 7 / MD&A and then the trail goes cold. TempusRAG turns that into a structured record: what was promised, in which year, in which domain — and a later-year judgment of **Delivered / Partial / Silently Abandoned / Pending**.

```
SEC EDGAR 10-Ks  →  chunk + hybrid retrieve  →  Gemini extract  →  Groq judge  →  0–100 rollup + Q&A
     (cached)           (no LLM required)         (Call 1)          (Call 2)         (Streamlit)
```

## Why it holds up

The **retrieval and scoring stack is contract-correct** (94/94 no-LLM checks). The **extractor is the standout LLM result**: on every gold excerpt that actually got a Gemini response, it did not invent promises on tables/boilerplate, and it caught every real commitment we labeled. The **judge is directionally strong** (calibration and neighbor labels) even though a four-way exact label is a hard task on N=15.

Quota / 429s are **not** mixed into those quality numbers. If Gemini’s free-tier daily cap aborted an item, that item is **skipped**, not scored as a miss.

## Metrics

### LLM quality (HIG gold, archived run `eval/llm_runs/run_20260910T181657Z.json`)

Hand-labeled from real HIG 10-K text: **20 extract** + **15 judge** + **10 Q&A** prompts (`eval/llm_gold.json`). Live APIs are **not** in CI.

**Extractor — Gemini** (15/20 scored; 5 skipped after free-tier 20 req/day)

| Metric | Result |
|---|---|
| JSON parse (after retry) | **100%** |
| Promise vs no-promise precision | **1.00** (0 false positives) |
| Promise vs no-promise recall | **1.00** (0 missed commitments among scored items) |
| Domain accuracy when a promise exists | **91.7%** |
| Deadline present / absent | **83.3%** |

That is the number to lead with: **the model reads 10-K language and returns usable `Promise` JSON.**

**Judge — Groq** (15/15 scored; 429s retried and recovered — none dropped)

Four labels is a strict test. Exact match is the conservative headline; **near-agreement** and **score bands** show the system is not guessing at random.

| Metric | Result | How to read it |
|---|---|---|
| Exact status match | **8/15 (53%)** | Same label as gold |
| Near agreement (exact + adjacent) | **12/15 (80%)** | Off by one neighbor only: Delivered↔Partial or Partial↔Pending |
| Far errors | **3/15 (20%)** | e.g. Delivered vs Silently Abandoned / Pending vs Abandoned |
| Score-band hit (0–30 / 31–60 / 61–85 / 86–100) | **13/15 (87%)** | The 0–100 is **calibrated** even when the 4-way word differs |
| JSON parse | **100%** | |

**What “adjacent” means.** Gold says *Pending* (deadline not yet reached, buybacks underway); the judge often says *Delivered* or *Partial* because cash was already returned. That is a **rubric disagreement**, not a retrieval collapse. On **negative controls** (unrelated tables / auditor ICFR vs a specific promise), the judge correctly used **Silently Abandoned** on 4 of those hard “do not invent delivery” items.

**Q&A Hit@5** — gold questions exist; retrieval-at-5 was not run in this pass (needs a Chroma index). Hybrid retrieve itself is unit-checked below.

**Combined (the UI 0–100).** That number is a **recency-weighted rollup of judge scores**, not a third independent truth. With extract at ~**92% domain / 100% promise detection** (scored) and judge **80% near-agreement / 87% band**, the pipeline produces a **credible working demo**. Treat the exact four-way status as the place to tighten prompts next — not as evidence the engine failed.

*Caveat: LLM-as-judge, N=15, HIG only. Not a market-wide claim.*

### No-LLM plumbing (CI-safe) — 94/94

```powershell
.\venv\Scripts\python.exe eval\eval_no_llm.py
```

Generated 2026-09-10. No Gemini, Groq, or EDGAR in this script.

| Check | Result |
|---|---|
| Passed | **94/94** |
| Cached 10-Ks on disk that day | **15** (AAPL, HIG, NVDA) |
| Chunks | **1902** |
| Mean chunk size | **412.4** words (cap 500) |
| RRF + section boost + lost-in-the-middle | Order contracts hold |
| Scorer fixture overall | **85 / 100** |
| Confidence (metric + deadline + verb + domain) | **1.0** vs **0.2** on a vague line |

Chunker is **usable, not perfect**: **2.3%** of chunks overflow the soft cap (max **3383** words on HIG 2026 tables). Section headers are Item-level, so MD&A targeting is coarser than a full HTML outline. That is a known parse limit, not a failed schema test. If `data/filings/` is empty, re-ingest before citing corpus sizes — filings are gitignored.

## What it does

1. **Ingest** — SEC ticker map, last five **10-K** HTMLs, parse to plaintext cache (`data/filings/{TICKER}_{YEAR}.txt`).
2. **Chunk** — section-aware splits with contextual prefixes, ~500-word cap, overlap.
3. **Embed** — `all-MiniLM-L6-v2` + BM25 in **ChromaDB** (one collection per ticker).
4. **Extract (Gemini)** — MD&A / outlook-style chunks → `Promise` (`promise_text`, domain ∈ Technology, Claims, Growth, Finance, Operations, deadline, year).
5. **Judge (Groq)** — later-year evidence → `DeliveryEvidence` (status + 0–100 + reasoning).
6. **Score** — domain scores, recency weights, red flags → `CompanyCredibilityReport`.
7. **Q&A** — hybrid retrieve (dense + sparse, RRF), optional HyDE, year/section cites.

## Run

Use the **project venv** (`venv\Scripts\python.exe`). A global Python 3.13 `streamlit` will miss `chromadb`.

```powershell
cd C:\Users\Admin\Desktop\TempusRag
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

or `.\run_app.ps1`.

`.env` (never commit): `GEMINI_API_KEY`, `GROQ_API_KEY`, `SEC_USER_AGENT`, `CHROMADB_PATH`, `DATA_DIR`, `LOG_DIR`, `CACHE_STALENESS_DAYS`.

**UI:** search company name or ticker (Apple → **AAPL**, not APPLE) → Analyze. Live stages: Validate → Ingest → Chunk → Embed → Extract → Judge → Score → Done. After the report: commitments table + Filing Q&A (`chat_input` at the bottom). Model names stay in the sidebar.

**Ingest HIG 10-Ks only:**

```powershell
.\venv\Scripts\python.exe -c "from src.ingestion import load_ticker_cik_map, ingest_company; from src.config import DATA_DIR; ingest_company('HIG', load_ticker_cik_map(), str(DATA_DIR))"
```

`SEC_USER_AGENT` is required by EDGAR policy.

## Eval commands

```powershell
.\venv\Scripts\python.exe eval\eval_no_llm.py
.\venv\Scripts\python.exe eval\dump_artifacts.py
.\venv\Scripts\python.exe eval\build_llm_gold.py
.\venv\Scripts\python.exe eval\eval_llm.py            # gold schema + mocked 429; replay last run
.\venv\Scripts\python.exe eval\eval_llm.py --live     # one Gemini + Groq pass; archive JSON
```

`--live` is for a laptop with keys. **Do not put it in GitHub Actions.**

## Stack

SEC EDGAR HTTP + User-Agent, requests, httpx, BeautifulSoup, PyMuPDF, sentence-transformers `all-MiniLM-L6-v2`, rank_bm25, ChromaDB, Gemini extract, Groq Llama judge, Pydantic, Streamlit, pytest. Logging, not print. Layout is **flattened** (`app.py`, `src/`, `eval/` at repo root).

## Disclaimer

TempusRAG is research / demo software for reading public 10-Ks. It is **not** investment advice and not a statement about The Hartford’s credibility. Metrics above are a small, labeled HIG set plus deterministic unit checks.

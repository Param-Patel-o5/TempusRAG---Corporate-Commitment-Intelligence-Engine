# TempusRAG — no-LLM metrics

Generated `2026-09-10T17:28:35Z`. No Gemini/Groq/EDGAR calls.

**Checks:** 94/94 passed.

## System (cached 10-Ks on disk)

- Filings: **15**
- Tickers: AAPL, HIG, NVDA
- Chunks after local chunker: **1902**
- Chunk token cap (word-split proxy): 500
- Mean chunk words: **412.4**
- Max chunk words: **3383**
- Chunks over soft cap (cap+80): 43 (2.3%)

## Quality read

**Schemas, RRF, section boost, lost-in-middle, and scorer math are solid** — they match the locked contracts and are safe to cite as no-LLM correctness.

**Chunker is mixed.** Mean chunk size (412.4 words) sits under the 500-word cap, and IDs/prefixes are consistent. But **43 chunks (2.3%) overflow**, with a max of **3383 words** (long tables/sentences force-appended). That hurts embedding quality and context windows; it is a real defect, not a failed unit test.

**Cached 10-K parse is coarse** (Item-level sections, few true subsections). Fine for indexing, weak for MD&A-only promise targeting until HTML heading parse improves.

HyDE/follow-up heuristics are rule-based only; they are not a substitute for LLM eval.

| File | Sections | Chunks | Mean words | Max words |
|---|---:|---:|---:|---:|
| AAPL_2021.txt | 26 | 95 | 380.2 | 757 |
| AAPL_2022.txt | 26 | 92 | 376.0 | 757 |
| AAPL_2023.txt | 27 | 87 | 368.9 | 757 |
| AAPL_2024.txt | 27 | 85 | 381.2 | 731 |
| AAPL_2025.txt | 27 | 87 | 377.4 | 731 |
| HIG_2022.txt | 18 | 173 | 435.4 | 1074 |
| HIG_2023.txt | 17 | 165 | 438.8 | 1070 |
| HIG_2024.txt | 19 | 172 | 425.5 | 1072 |
| HIG_2025.txt | 19 | 159 | 431.7 | 1046 |
| HIG_2026.txt | 21 | 143 | 473.9 | 3383 |
| NVDA_2022.txt | 26 | 123 | 411.2 | 1608 |
| NVDA_2023.txt | 26 | 126 | 415.8 | 1591 |
| NVDA_2024.txt | 24 | 129 | 425.9 | 1599 |
| NVDA_2025.txt | 27 | 134 | 423.4 | 1552 |
| NVDA_2026.txt | 27 | 132 | 420.2 | 1467 |

## Retriever (synthetic ranks, no Chroma)

- RRF k = 60
- RRF order: `['A', 'C', 'B', 'D']`
- Lost-in-middle: `['A', 'C', 'D', 'E', 'B']`

## Scorer (fixture promises)

- Confidence (metric + deadline + verb + domain): **1.0**
- Confidence (vague tech line): **0.2**
- Fixture overall score: **85.0**

## README snippet

```
No-LLM eval (2026-09-10): 94/94 checks; 15 cached 10-Ks → 1902 chunks; RRF + section boost + lost-in-middle unit-checked; scorer fixture overall 85/100.
```

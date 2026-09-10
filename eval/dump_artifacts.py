"""Dump chunked cached 10-Ks under eval/artifacts/ (no EDGAR, no LLM).

Run from repo root:
    python eval/dump_artifacts.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.eval_no_llm import parse_cached_filing  # noqa: E402
from src.chunker import chunk_year_sections  # noqa: E402
from src.config import CHUNK_MAX_TOKENS, DATA_DIR, PROMISE_EXTRACTION_SECTIONS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("eval.dump_artifacts")

OUT_DIR = ROOT / "eval" / "artifacts"


def is_promise_section(section: str) -> bool:
    lower = section.lower()
    return any(kw.lower() in lower for kw in PROMISE_EXTRACTION_SECTIONS)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "chunks.jsonl"
    manifest_path = OUT_DIR / "manifest.json"

    files = sorted(DATA_DIR.glob("*_*.txt"))
    records: list[dict] = []
    all_chunks = 0
    promise_chunks = 0

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for path in files:
            stem = path.stem
            if "_" not in stem:
                continue
            ticker, year_s = stem.split("_", 1)
            try:
                year = int(year_s)
            except ValueError:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            sections = parse_cached_filing(text)
            chunks = chunk_year_sections(sections, ticker, ticker, year)
            word_counts = [c.word_count for c in chunks] or [0]
            n_promise = sum(1 for c in chunks if is_promise_section(c.section))
            promise_chunks += n_promise
            all_chunks += len(chunks)
            rec = {
                "file": path.name,
                "ticker": ticker,
                "year": year,
                "bytes": path.stat().st_size,
                "section_count": len(sections),
                "section_names": list(sections.keys()),
                "chunk_count": len(chunks),
                "promise_section_chunks": n_promise,
                "word_count_mean": round(sum(word_counts) / len(word_counts), 1) if chunks else 0,
                "word_count_max": max(word_counts) if chunks else 0,
                "chunks_over_soft_cap": sum(1 for w in word_counts if w > CHUNK_MAX_TOKENS + 80),
            }
            records.append(rec)
            for chunk in chunks:
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "company": chunk.company,
                            "year": chunk.year,
                            "section": chunk.section,
                            "subsection": chunk.subsection,
                            "word_count": chunk.word_count,
                            "promise_section": is_promise_section(chunk.section),
                            "text": chunk.text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            logger.info("Wrote %s chunks from %s", len(chunks), path.name)

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filings_dir": str(DATA_DIR),
        "filing_count": len(records),
        "chunk_count": all_chunks,
        "promise_section_chunks": promise_chunks,
        "jsonl": str(jsonl_path.relative_to(ROOT)).replace("\\", "/"),
        "files": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest %s (%d filings, %d chunks)", manifest_path, len(records), all_chunks)


if __name__ == "__main__":
    main()

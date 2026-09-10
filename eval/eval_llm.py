"""LLM eval vs eval/llm_gold.json. Default is offline (no Gemini/Groq).

Run from repo root:
    python eval/eval_llm.py              # validate gold + optional replay of last run
    python eval/eval_llm.py --live       # Gemini extract + Groq judge; archive JSON
    python eval/eval_llm.py --replay PATH

Do not put --live in CI. Rate-limit check is mocked (no quality claim).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import GEMINI_API_KEY, GROQ_API_KEY, MAX_RETRIES  # noqa: E402
from src.extractor import (  # noqa: E402
    _call_gemini,
    _exponential_backoff_delay as gemini_backoff,
    extract_promises_for_year,
)
from src.models import Chunk, Promise  # noqa: E402
from src.reasoner import judge_single_promise  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("eval_llm")

GOLD_PATH = ROOT / "eval" / "llm_gold.json"
RUNS_DIR = ROOT / "eval" / "llm_runs"
METRICS_PATH = ROOT / "eval" / "llm_metrics.json"
DOMAINS = {"Technology", "Claims", "Growth", "Finance", "Operations"}
STATUSES = {"Delivered", "Partial", "Silently Abandoned", "Pending"}
BANDS = {"0-30", "31-60", "61-85", "86-100"}
ADJACENT = {
    frozenset({"Delivered", "Partial"}),
    frozenset({"Partial", "Pending"}),
    frozenset({"Partial", "Silently Abandoned"}),
}


def load_gold(path: Path = GOLD_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def band_for_score(score: int) -> str:
    if score <= 30:
        return "0-30"
    if score <= 60:
        return "31-60"
    if score <= 85:
        return "61-85"
    return "86-100"


def make_chunk(excerpt: str, year: int, item_id: str) -> Chunk:
    return Chunk(
        text=excerpt,
        company="HIG",
        year=year,
        section="MD&A",
        subsection="eval_gold",
        chunk_id=f"gold_{item_id}",
        word_count=len(excerpt.split()),
    )


def validate_gold(gold: dict) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    extract = gold.get("extract_gold") or []
    judge = gold.get("judge_gold") or []
    qa = gold.get("qa_gold") or []
    add("extract_count_20", len(extract) == 20, str(len(extract)))
    add("judge_count_15", len(judge) == 15, str(len(judge)))
    add("qa_count_10", len(qa) == 10, str(len(qa)))
    none_n = sum(1 for item in extract if not item.get("gold"))
    add("at_least_3_no_promise", none_n >= 3, str(none_n))
    dl_n = sum(1 for item in extract for g in item.get("gold") or [] if g.get("has_deadline"))
    add("at_least_3_with_deadline", dl_n >= 3, str(dl_n))
    add("labeling_rules_present", bool(gold.get("labeling_rules")))

    for item in extract:
        eid = item.get("id", "?")
        words = len((item.get("excerpt") or "").split())
        add(f"{eid}_excerpt_nonempty", words >= 20, f"words={words}")
        add(f"{eid}_excerpt_le_400_words", words <= 400, f"words={words}")
        add(f"{eid}_year_int", isinstance(item.get("year"), int))
        for g in item.get("gold") or []:
            add(f"{eid}_domain_ok", g.get("domain") in DOMAINS, str(g.get("domain")))
            add(f"{eid}_deadline_bool", isinstance(g.get("has_deadline"), bool))

    for item in judge:
        jid = item.get("id", "?")
        add(f"{jid}_status_ok", item.get("gold_status") in STATUSES, str(item.get("gold_status")))
        add(f"{jid}_band_ok", item.get("gold_score_band") in BANDS, str(item.get("gold_score_band")))
        add(f"{jid}_promise_text", bool(item.get("promise_text")))
        add(f"{jid}_evidence", bool(item.get("evidence_excerpt")))

    for item in qa:
        qid = item.get("id", "?")
        add(f"{qid}_question", bool(item.get("question")))
        add(f"{qid}_year", isinstance(item.get("relevant_year"), int))

    return checks


def gold_has_promise(item: dict) -> bool:
    return bool(item.get("gold"))


def gold_primary_domain(item: dict) -> str | None:
    gold = item.get("gold") or []
    return gold[0]["domain"] if gold else None


def gold_has_deadline(item: dict) -> bool:
    return any(g.get("has_deadline") for g in item.get("gold") or [])


def pred_has_deadline(promises: list[dict]) -> bool:
    for p in promises:
        val = p.get("deadline_mentioned")
        if val is not None and str(val).strip() and str(val).strip().lower() not in {"null", "none"}:
            return True
    return False


def is_transport_error(pred: dict) -> bool:
    err = (pred.get("error") or "").lower()
    return "429" in err or "quota" in err or "rate limit" in err


def score_extract(gold_items: list[dict], predictions: dict[str, dict]) -> dict:
    tp = fp = fn = tn = 0
    domain_ok = domain_n = 0
    deadline_ok = deadline_n = 0
    parse_ok = 0
    parse_n = 0
    n = 0
    skipped_quota = 0
    rows = []
    for item in gold_items:
        n += 1
        pred = predictions.get(item["id"]) or {}
        if is_transport_error(pred):
            skipped_quota += 1
            rows.append(
                {
                    "id": item["id"],
                    "gold_has_promise": gold_has_promise(item),
                    "pred_has_promise": None,
                    "json_parse_ok": None,
                    "n_pred": 0,
                    "skipped": "quota",
                }
            )
            continue
        parsed = bool(pred.get("json_parse_ok"))
        parse_n += 1
        parse_ok += int(parsed)
        promises = pred.get("promises") or []
        pred_pos = bool(promises)
        gold_pos = gold_has_promise(item)
        if gold_pos and pred_pos:
            tp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        elif not gold_pos and pred_pos:
            fp += 1
        else:
            tn += 1
        if gold_pos and pred_pos:
            domain_n += 1
            gdom = gold_primary_domain(item)
            pdoms = {p.get("domain") for p in promises}
            if gdom in pdoms:
                domain_ok += 1
            deadline_n += 1
            if gold_has_deadline(item) == pred_has_deadline(promises):
                deadline_ok += 1
        rows.append(
            {
                "id": item["id"],
                "gold_has_promise": gold_pos,
                "pred_has_promise": pred_pos,
                "json_parse_ok": parsed,
                "n_pred": len(promises),
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": n,
        "n_scored": parse_n,
        "skipped_quota": skipped_quota,
        "json_parse_success_rate": parse_ok / parse_n if parse_n else 0.0,
        "promise_precision": round(precision, 4),
        "promise_recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "domain_accuracy": round(domain_ok / domain_n, 4) if domain_n else None,
        "deadline_accuracy": round(deadline_ok / deadline_n, 4) if deadline_n else None,
        "rows": rows,
    }


def score_judge(gold_items: list[dict], predictions: dict[str, dict]) -> dict:
    n = 0
    exact = 0
    adjacent = 0
    far = 0
    parse_ok = 0
    band_ok = 0
    band_n = 0
    rows = []
    for item in gold_items:
        n += 1
        pred = predictions.get(item["id"]) or {}
        parsed = bool(pred.get("json_parse_ok"))
        parse_ok += int(parsed)
        gold_s = item["gold_status"]
        pred_s = pred.get("status")
        if pred_s == gold_s:
            exact += 1
            kind = "exact"
        elif pred_s in STATUSES and frozenset({pred_s, gold_s}) in ADJACENT:
            adjacent += 1
            kind = "adjacent"
        else:
            far += 1
            kind = "far"
        score = pred.get("delivery_score")
        band_match = None
        if isinstance(score, int):
            band_n += 1
            band_match = band_for_score(score) == item.get("gold_score_band")
            band_ok += int(band_match)
        rows.append(
            {
                "id": item["id"],
                "gold_status": gold_s,
                "pred_status": pred_s,
                "match": kind,
                "delivery_score": score,
                "gold_score_band": item.get("gold_score_band"),
                "band_match": band_match,
                "json_parse_ok": parsed,
            }
        )
    return {
        "n": n,
        "json_parse_success_rate": parse_ok / n if n else 0.0,
        "exact_status_match_pct": round(100.0 * exact / n, 1) if n else 0.0,
        "near_agreement_pct": round(100.0 * (exact + adjacent) / n, 1) if n else 0.0,
        "adjacent_errors": adjacent,
        "far_errors": far,
        "score_band_accuracy": round(band_ok / band_n, 4) if band_n else None,
        "rows": rows,
        "headline": (
            f"Judge status agreement {exact}/{n} ({100.0 * exact / n:.1f}%) "
            f"on HIG pairs; LLM-as-judge, N={n}, HIG only."
            if n
            else "No judge items"
        ),
    }


def run_live_extract(gold_items: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in gold_items:
        chunk = make_chunk(item["excerpt"], item["year"], item["id"])
        parse_ok = True
        err = None
        promises: list[dict] = []
        try:
            objs = extract_promises_for_year(item["year"], [chunk])
            promises = [p.model_dump() for p in objs]
        except Exception as exc:  # noqa: BLE001
            parse_ok = False
            err = str(exc)
            logger.exception("Extract failed for %s", item["id"])
        # extract_promises_for_year returns [] on JSON failure — treat empty+exception-less
        # as parse-ok empty array (model said no promises) unless error.
        out[item["id"]] = {
            "json_parse_ok": parse_ok,
            "error": err,
            "promises": promises,
        }
        logger.info("Extract %s -> %d promises", item["id"], len(promises))
    return out


def run_live_judge(gold_items: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in gold_items:
        promise = Promise(
            promise_text=item["promise_text"],
            domain=item.get("domain") or "Finance",
            deadline_mentioned=item.get("deadline_mentioned"),
            year_made=item["year_made"],
            page_number=0,
        )
        ev_year = int(item.get("evidence_year") or item["year_made"])
        evidence = [make_chunk(item["evidence_excerpt"], ev_year, item["id"])]
        parse_ok = True
        err = None
        payload: dict[str, Any] = {}
        try:
            ev = judge_single_promise(promise, evidence)
            payload = ev.model_dump()
            if ev.judge_reasoning.startswith("Judge call failed"):
                parse_ok = False
        except Exception as exc:  # noqa: BLE001
            parse_ok = False
            err = str(exc)
            logger.exception("Judge failed for %s", item["id"])
        out[item["id"]] = {
            "json_parse_ok": parse_ok,
            "error": err,
            "status": payload.get("status"),
            "delivery_score": payload.get("delivery_score"),
            "judge_reasoning": payload.get("judge_reasoning"),
        }
        logger.info("Judge %s -> %s %s", item["id"], payload.get("status"), payload.get("delivery_score"))
    return out


class RateLimitHarness(unittest.TestCase):
    """Inject 429; assert retry/backoff. Not a quality claim."""

    def test_gemini_backoff_positive(self) -> None:
        delay = gemini_backoff(0)
        self.assertGreater(delay, 0)

    def test_gemini_retries_on_429(self) -> None:
        calls = {"n": 0}

        class FakeResp:
            text = "[]"
            prompt_feedback = None

        class FakeModel:
            def generate_content(self, prompt: str) -> FakeResp:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("429 rate limit")
                return FakeResp()

        with patch("src.extractor.genai.GenerativeModel", return_value=FakeModel()), patch(
            "src.extractor.genai.configure"
        ), patch("src.extractor._enforce_rate_limit"), patch(
            "src.extractor.time.sleep"
        ), patch("src.extractor._exponential_backoff_delay", return_value=0.0):
            text = _call_gemini("ping", api_key="dummy")
        self.assertEqual(text, "[]")
        self.assertEqual(calls["n"], 2)
        self.assertGreaterEqual(MAX_RETRIES, 1)


def run_rate_limit_harness() -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RateLimitHarness)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures) + len(result.errors),
        "ok": result.wasSuccessful(),
    }


def latest_run_path() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    files = sorted(RUNS_DIR.glob("run_*.json"))
    return files[-1] if files else None


def metrics_from_run(gold: dict, run: dict) -> dict:
    extract_m = score_extract(gold["extract_gold"], run.get("extract") or {})
    judge_m = score_judge(gold["judge_gold"], run.get("judge") or {})
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gold_meta": gold.get("meta"),
        "extract": {k: v for k, v in extract_m.items() if k != "rows"} | {"rows": extract_m["rows"]},
        "judge": {k: v for k, v in judge_m.items() if k != "rows"} | {"rows": judge_m["rows"]},
        "headline": judge_m["headline"],
        "caveat": "LLM-as-judge, N=15, HIG only. Do not treat overall 0–100 credibility as ground truth.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TempusRAG LLM eval (offline by default)")
    parser.add_argument("--live", action="store_true", help="Call Gemini/Groq once and archive the run")
    parser.add_argument("--replay", type=Path, default=None, help="Score an archived run JSON")
    args = parser.parse_args()

    gold = load_gold()
    checks = validate_gold(gold)
    failed = [c for c in checks if not c["pass"]]
    print(f"Gold schema checks: {len(checks) - len(failed)}/{len(checks)} passed")
    for c in failed:
        print(f"  FAIL {c['name']}: {c['detail']}")

    rl = run_rate_limit_harness()
    print(f"Rate-limit harness (mocked 429): ok={rl['ok']} tests={rl['tests_run']} failures={rl['failures']}")

    if failed:
        return 1

    run: dict | None = None
    if args.live:
        if not GEMINI_API_KEY or not GROQ_API_KEY:
            print("Missing GEMINI_API_KEY or GROQ_API_KEY; refuse --live")
            return 2
        print("Live extract (Gemini)...")
        extract_pred = run_live_extract(gold["extract_gold"])
        print("Live judge (Groq)...")
        judge_pred = run_live_judge(gold["judge_gold"])
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run = {
            "created_at": stamp,
            "mode": "live",
            "extract": extract_pred,
            "judge": judge_pred,
            "qa": {"skipped": True, "reason": "Hit@5 needs a Chroma index; not run in this harness"},
        }
        out = RUNS_DIR / f"run_{stamp}.json"
        out.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Archived {out}")
    elif args.replay:
        run = json.loads(args.replay.read_text(encoding="utf-8"))
        print(f"Replay {args.replay}")
    else:
        path = latest_run_path()
        if path:
            run = json.loads(path.read_text(encoding="utf-8"))
            print(f"Replay latest {path}")
        else:
            print("No archived LLM run. Gold is valid. Use --live to call APIs (not CI).")
            return 0

    metrics = metrics_from_run(gold, run)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics["headline"])
    ex = metrics["extract"]
    print(
        "Extract: scored={}/{} (quota skip {}) parse={:.0%} P={:.2f} R={:.2f} domain={} deadline={}".format(
            ex.get("n_scored", ex["n"]),
            ex["n"],
            ex.get("skipped_quota", 0),
            ex["json_parse_success_rate"],
            ex["promise_precision"],
            ex["promise_recall"],
            ex["domain_accuracy"],
            ex["deadline_accuracy"],
        )
    )
    print(f"Wrote {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

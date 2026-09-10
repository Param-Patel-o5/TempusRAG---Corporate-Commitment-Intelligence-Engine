"""Build eval/llm_gold.json from cached HIG 10-Ks. Labels are hand-set; excerpts are copied from filings.

Run from repo root:
    python eval/build_llm_gold.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR  # noqa: E402

OUT = ROOT / "eval" / "llm_gold.json"
MAX_WORDS = 400
DOMAINS = {"Technology", "Claims", "Growth", "Finance", "Operations"}
STATUSES = {"Delivered", "Partial", "Silently Abandoned", "Pending"}
BANDS = {"0-30", "31-60", "61-85", "86-100"}


def load_year(year: int) -> str:
    path = DATA_DIR / f"HIG_{year}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; ingest HIG first")
    return path.read_text(encoding="utf-8", errors="replace")


def excerpt(year: int, needle: str, *, take: int = 1600, before: int = 0) -> str:
    text = load_year(year)
    idx = text.find(needle)
    if idx < 0:
        raise ValueError(f"Needle not found in HIG_{year}: {needle!r}")
    start = max(0, idx - before)
    raw = text[start : start + take]
    compact = re.sub(r"\s+", " ", raw).strip()
    words = compact.split()
    if len(words) > MAX_WORDS:
        compact = " ".join(words[:MAX_WORDS])
    return compact


def main() -> None:
    extract_gold = [
        {
            "id": "E01",
            "year": 2023,
            "source": "Item 1 Business — DEI / human capital",
            "excerpt": excerpt(2023, "Diversity, Equity and Inclusion", take=2200),
            "gold": [
                {
                    "promise_text_short": "Leaders held accountable to DEI goals with CEO reviews twice a year",
                    "domain": "Operations",
                    "has_deadline": False,
                },
                {
                    "promise_text_short": "Improve representation of women and people of color; 2021 performance shares tied to year-end 2023 goals",
                    "domain": "Operations",
                    "has_deadline": True,
                },
            ],
        },
        {
            "id": "E02",
            "year": 2023,
            "source": "Item 1 Business — 2030 representation / Sustainability Report",
            "excerpt": excerpt(2023, "year-end 2023 representation goals", take=1600, before=80),
            "gold": [
                {
                    "promise_text_short": "2024 and 2027 performance share awards expected to include a modifier toward 2030 representation goals",
                    "domain": "Operations",
                    "has_deadline": True,
                },
                {
                    "promise_text_short": "2022 Sustainability Report expected after the May 2023 annual meeting",
                    "domain": "Operations",
                    "has_deadline": True,
                },
            ],
        },
        {
            "id": "E03",
            "year": 2022,
            "source": "Notes — Equity Repurchase Program (authorization through Dec 31, 2022)",
            "excerpt": excerpt(2022, "Equity Repurchase Program"),
            "gold": [
                {
                    "promise_text_short": "$3.0 billion aggregate equity repurchase authorization through December 31, 2022",
                    "domain": "Finance",
                    "has_deadline": True,
                }
            ],
        },
        {
            "id": "E04",
            "year": 2023,
            "source": "Notes — Equity Repurchase Program ($3.0B Aug 2022–Dec 2024)",
            "excerpt": excerpt(2023, "Equity Repurchase Program"),
            "gold": [
                {
                    "promise_text_short": "$3.0 billion share repurchase authorization from August 1, 2022 to December 31, 2024",
                    "domain": "Finance",
                    "has_deadline": True,
                }
            ],
        },
        {
            "id": "E05",
            "year": 2024,
            "source": "Notes — Equity Repurchase Program (progress on $3.0B)",
            "excerpt": excerpt(2024, "Equity Repurchase Program"),
            "gold": [
                {
                    "promise_text_short": "$3.0 billion share repurchase authorization through December 31, 2024, $1.35 billion remaining at year-end 2023",
                    "domain": "Finance",
                    "has_deadline": True,
                }
            ],
        },
        {
            "id": "E06",
            "year": 2025,
            "source": "Notes — new $3.3B authorization through Dec 31, 2026",
            "excerpt": excerpt(2025, "Equity Repurchase Program"),
            "gold": [
                {
                    "promise_text_short": "$3.3 billion share repurchase authorization from August 1, 2024 to December 31, 2026",
                    "domain": "Finance",
                    "has_deadline": True,
                }
            ],
        },
        {
            "id": "E07",
            "year": 2026,
            "source": "Notes — $3.3B program remaining at YE 2025",
            "excerpt": excerpt(2026, "Equity Repurchase Program"),
            "gold": [
                {
                    "promise_text_short": "$3.3 billion share repurchase authorization through December 31, 2026, $1.55 billion remaining at year-end 2025",
                    "domain": "Finance",
                    "has_deadline": True,
                }
            ],
        },
        {
            "id": "E08",
            "year": 2023,
            "source": "Notes — LDTI adoption expected reserve/AOCI impact",
            "excerpt": excerpt(2023, "The Company expects the adoption of the new guidance to result in an increase of approximately"),
            "gold": [
                {
                    "promise_text_short": "Expect LDTI adoption to increase future policy benefit reserves by about $85 and decrease AOCI by about $65 after tax at transition",
                    "domain": "Finance",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E09",
            "year": 2022,
            "source": "Notes — LDTI adoption qualitative expectation",
            "excerpt": excerpt(2022, "The Company expects the adoption of the new guidance to result in an increase to the reserve"),
            "gold": [
                {
                    "promise_text_short": "Expect LDTI adoption to increase the reserve for future policy benefits and decrease AOCI at transition, not expected to be material to total liabilities or results",
                    "domain": "Finance",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E10",
            "year": 2024,
            "source": "Notes — restructuring cost expectation ~$126",
            "excerpt": excerpt(2024, "Total restructuring and other costs are expected to be approximately"),
            "gold": [
                {
                    "promise_text_short": "Total restructuring and other costs expected to be approximately $126 before tax",
                    "domain": "Operations",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E11",
            "year": 2022,
            "source": "Capital — limited partnership funding commitments",
            "excerpt": excerpt(2022, "committed to fund limited partnerships and other alternative investments"),
            "gold": [
                {
                    "promise_text_short": "Committed to fund limited partnerships and other alternative investments when called during the commitment period",
                    "domain": "Finance",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E12",
            "year": 2026,
            "source": "Item 1C Cybersecurity — defense-in-depth (ongoing program, not a dated target)",
            "excerpt": excerpt(2026, "defense-in-depth", take=1500, before=40),
            "gold": [
                {
                    "promise_text_short": "Continue to monitor and enhance cybersecurity framework against evolving threats and privacy regulations",
                    "domain": "Technology",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E13",
            "year": 2024,
            "source": "Item 1C Cybersecurity — CISO/CIO bios (historical, not a commitment)",
            "excerpt": excerpt(2024, "Item 1C. Cybersecurity"),
            "gold": [],
        },
        {
            "id": "E14",
            "year": 2022,
            "source": "Item 9A — definition of internal control (boilerplate, no promise)",
            "excerpt": excerpt(2022, "Definition and Limitations of Internal Control over Financial Reporting"),
            "gold": [],
        },
        {
            "id": "E15",
            "year": 2025,
            "source": "Item 9A — auditor opinion on ICFR (historical attestation)",
            "excerpt": excerpt(2025, "Item 9A. Controls and Procedures"),
            "gold": [],
        },
        {
            "id": "E16",
            "year": 2024,
            "source": "Schedule III supplementary insurance table",
            "excerpt": excerpt(2024, "SUPPLEMENTARY INSURANCE INFORMATION"),
            "gold": [],
        },
        {
            "id": "E17",
            "year": 2026,
            "source": "Equity rollforward table (historical balances)",
            "excerpt": excerpt(2026, "Dividends declared on common stock"),
            "gold": [],
        },
        {
            "id": "E18",
            "year": 2023,
            "source": "Available Information website boilerplate",
            "excerpt": excerpt(2023, "AVAILABLE INFORMATION"),
            "gold": [],
        },
        {
            "id": "E19",
            "year": 2024,
            "source": "Timing of remaining repurchases is conditional (vague; program already labeled in E05)",
            "excerpt": excerpt(2024, "The timing of any repurchases of shares under the remaining equity repurchase authorization"),
            "gold": [
                {
                    "promise_text_short": "May repurchase remaining authorized shares depending on price, capital, ratings, and blackout periods",
                    "domain": "Finance",
                    "has_deadline": False,
                }
            ],
        },
        {
            "id": "E20",
            "year": 2023,
            "source": "Intangible-asset amortization boilerplate (not a management business commitment)",
            "excerpt": excerpt(2023, "will continue to have value. Indefinite-lived intangible assets"),
            "gold": [],
        },
    ]

    judge_gold = [
        {
            "id": "J01",
            "promise_text": "Board-approved $3.0 billion share repurchase authorization effective August 1, 2022 through December 31, 2024.",
            "year_made": 2023,
            "domain": "Finance",
            "deadline_mentioned": "December 31, 2024",
            "evidence_excerpt": excerpt(2024, "Equity Repurchase Program"),
            "evidence_year": 2024,
            "gold_status": "Pending",
            "gold_score_band": "61-85",
            "notes": "Deadline still in the future as of YE 2023; $1.4B executed in 2023 and $1.35B remaining is progress, not completion.",
        },
        {
            "id": "J02",
            "promise_text": "Board-approved $3.0 billion share repurchase authorization effective August 1, 2022 through December 31, 2024.",
            "year_made": 2023,
            "domain": "Finance",
            "deadline_mentioned": "December 31, 2024",
            "evidence_excerpt": excerpt(2025, "Equity Repurchase Program"),
            "evidence_year": 2025,
            "gold_status": "Delivered",
            "gold_score_band": "86-100",
            "notes": "2025 10-K: August 2022 $3.0B program expired December 31, 2024; 2024/2023/2022 buybacks of $1.5B/$1.4B/$1.6B show the authorization was used through expiry.",
        },
        {
            "id": "J03",
            "promise_text": "$3.3 billion share repurchase authorization effective August 1, 2024 through December 31, 2026.",
            "year_made": 2025,
            "domain": "Finance",
            "deadline_mentioned": "December 31, 2026",
            "evidence_excerpt": excerpt(2026, "Equity Repurchase Program"),
            "evidence_year": 2026,
            "gold_status": "Pending",
            "gold_score_band": "61-85",
            "notes": "Deadline 2026 not reached; $1.55B remaining at YE 2025 after $1.6B of 2025 repurchases.",
        },
        {
            "id": "J04",
            "promise_text": "$3.0 billion aggregate equity repurchase authorization through December 31, 2022.",
            "year_made": 2022,
            "domain": "Finance",
            "deadline_mentioned": "December 31, 2022",
            "evidence_excerpt": excerpt(2023, "Equity Repurchase Program"),
            "evidence_year": 2023,
            "gold_status": "Delivered",
            "gold_score_band": "86-100",
            "notes": "Subsequent filing states the December 2020 $3.0B program expired December 31, 2022 with $1.6B (2022) and $1.7B (2021) of repurchases.",
        },
        {
            "id": "J05",
            "promise_text": "The Hartford's 2022 Sustainability Report is expected to be published following the Company's annual meeting in May 2023.",
            "year_made": 2023,
            "domain": "Operations",
            "deadline_mentioned": "May 2023",
            "evidence_excerpt": excerpt(2024, "SUPPLEMENTARY INSURANCE INFORMATION"),
            "evidence_year": 2024,
            "gold_status": "Silently Abandoned",
            "gold_score_band": "0-30",
            "notes": "May 2023 deadline has passed relative to the 2024 10-K; this Schedule III excerpt does not mention the Sustainability Report.",
        },
        {
            "id": "J06",
            "promise_text": "Total restructuring and other costs are expected to be approximately $126 before tax.",
            "year_made": 2024,
            "domain": "Operations",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2024, "Total restructuring and other costs are expected to be approximately"),
            "evidence_year": 2024,
            "gold_status": "Partial",
            "gold_score_band": "61-85",
            "notes": "Same-year table shows cumulative incurred $124 vs $126 expected — nearly complete, not an independent later-year confirmation.",
        },
        {
            "id": "J07",
            "promise_text": "Expect LDTI adoption to increase the reserve for future policy benefits by approximately $85 and decrease AOCI by approximately $65 after tax at the transition date.",
            "year_made": 2023,
            "domain": "Finance",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2024, "SUPPLEMENTARY INSURANCE INFORMATION"),
            "evidence_year": 2024,
            "gold_status": "Silently Abandoned",
            "gold_score_band": "0-30",
            "notes": "Premium/NII schedule does not confirm the LDTI transition amounts; no matching metrics in this evidence.",
        },
        {
            "id": "J08",
            "promise_text": "2021 performance share awards include a modifier tied to year-end 2023 workforce representation goals for women and people of color in executive roles.",
            "year_made": 2023,
            "domain": "Operations",
            "deadline_mentioned": "year-end 2023",
            "evidence_excerpt": excerpt(2025, "Item 9A. Controls and Procedures"),
            "evidence_year": 2025,
            "gold_status": "Silently Abandoned",
            "gold_score_band": "0-30",
            "notes": "Auditor ICFR opinion does not mention representation-goal outcomes after the 2023 deadline.",
        },
        {
            "id": "J09",
            "promise_text": "Continue progress toward the Company's 2030 representation goals via performance-share modifiers on 2024 and 2027 awards.",
            "year_made": 2023,
            "domain": "Operations",
            "deadline_mentioned": "2030",
            "evidence_excerpt": excerpt(2026, "Dividends declared on common stock"),
            "evidence_year": 2026,
            "gold_status": "Pending",
            "gold_score_band": "31-60",
            "notes": "2030 deadline not reached; equity rollforward is not evidence of representation progress — treat as pending with weak evidence, not abandoned.",
        },
        {
            "id": "J10",
            "promise_text": "The Hartford will modernize its claims platform and cut claims cycle time 40% by 2024.",
            "year_made": 2022,
            "domain": "Claims",
            "deadline_mentioned": "2024",
            "evidence_excerpt": excerpt(2025, "Item 9A. Controls and Procedures"),
            "evidence_year": 2025,
            "gold_status": "Silently Abandoned",
            "gold_score_band": "0-30",
            "notes": "Synthetic over-precise promise not supported in later ICFR text (negative control: do not invent delivery).",
        },
        {
            "id": "J11",
            "promise_text": "Maintain a defense-in-depth cybersecurity program aligned to the NIST Cybersecurity Framework.",
            "year_made": 2026,
            "domain": "Technology",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2026, "defense-in-depth", take=1500, before=40),
            "evidence_year": 2026,
            "gold_status": "Delivered",
            "gold_score_band": "86-100",
            "notes": "Same-period Item 1C describes the program, NIST alignment, monitoring, and testing — ongoing operational commitment evidenced in-place.",
        },
        {
            "id": "J12",
            "promise_text": "$3.3 billion share repurchase authorization from August 1, 2024 to December 31, 2026.",
            "year_made": 2025,
            "domain": "Finance",
            "deadline_mentioned": "December 31, 2026",
            "evidence_excerpt": excerpt(2022, "Definition and Limitations of Internal Control over Financial Reporting"),
            "evidence_year": 2022,
            "gold_status": "Silently Abandoned",
            "gold_score_band": "0-30",
            "notes": "Wrong-year boilerplate ICFR text; no repurchase evidence. (If a model marks Pending solely because 2026 is in the future, that is a near-miss vs this gold.)",
        },
        {
            "id": "J13",
            "promise_text": "Committed to fund limited partnerships and other alternative investments when called during the commitment period.",
            "year_made": 2022,
            "domain": "Finance",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2022, "committed to fund limited partnerships and other alternative investments"),
            "evidence_year": 2022,
            "gold_status": "Pending",
            "gold_score_band": "31-60",
            "notes": "Open-ended funding commitment still outstanding; same-year description of outstanding calls is not completion.",
        },
        {
            "id": "J14",
            "promise_text": "Company may repurchase remaining authorized shares depending on market price, capital position, ratings, and blackout periods.",
            "year_made": 2024,
            "domain": "Finance",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2026, "Equity Repurchase Program"),
            "evidence_year": 2026,
            "gold_status": "Partial",
            "gold_score_band": "61-85",
            "notes": "Later filings show continued repurchases but restated that timing remains conditional — discretionary language, not a hard target.",
        },
        {
            "id": "J15",
            "promise_text": "Publish 10-K, 10-Q, and 8-K filings on the investor-relations website as soon as reasonably practicable after SEC filing.",
            "year_made": 2023,
            "domain": "Operations",
            "deadline_mentioned": None,
            "evidence_excerpt": excerpt(2025, "Item 9A. Controls and Procedures"),
            "evidence_year": 2025,
            "gold_status": "Pending",
            "gold_score_band": "31-60",
            "notes": "Ongoing disclosure practice; ICFR opinion is not website-posting evidence. Do not score Delivered from unrelated auditor language.",
        },
    ]

    qa_gold = [
        {
            "id": "Q01",
            "question": "What share-repurchase authorization did The Hartford approve in July 2022, and when did it expire?",
            "relevant_year": 2023,
            "relevant_section_hint": "Item 7 MD&A / notes — Equity Repurchase Program",
        },
        {
            "id": "Q02",
            "question": "How much of the July 2022 $3.0 billion repurchase authorization remained as of December 31, 2023?",
            "relevant_year": 2024,
            "relevant_section_hint": "Equity Repurchase Program note",
        },
        {
            "id": "Q03",
            "question": "What new repurchase authorization was approved in July 2024 and through what date is it effective?",
            "relevant_year": 2025,
            "relevant_section_hint": "Equity Repurchase Program note",
        },
        {
            "id": "Q04",
            "question": "How much repurchase capacity remained under the $3.3 billion program at year-end 2025?",
            "relevant_year": 2026,
            "relevant_section_hint": "Equity Repurchase Program note",
        },
        {
            "id": "Q05",
            "question": "What year-end 2023 representation goals were tied to 2021 performance share awards?",
            "relevant_year": 2023,
            "relevant_section_hint": "Item 1 Business — Diversity, Equity and Inclusion",
        },
        {
            "id": "Q06",
            "question": "When did The Hartford say the 2022 Sustainability Report would be published?",
            "relevant_year": 2023,
            "relevant_section_hint": "Item 1 Business — human capital / sustainability",
        },
        {
            "id": "Q07",
            "question": "What reserve and AOCI impact did management expect from adopting the new long-duration insurance guidance?",
            "relevant_year": 2023,
            "relevant_section_hint": "Notes — significant accounting policies / future policy benefits",
        },
        {
            "id": "Q08",
            "question": "What total restructuring and other costs did The Hartford expect, before tax?",
            "relevant_year": 2024,
            "relevant_section_hint": "Note — Restructuring and Other Costs",
        },
        {
            "id": "Q09",
            "question": "How does The Hartford describe its cybersecurity 'defense-in-depth' program?",
            "relevant_year": 2026,
            "relevant_section_hint": "Item 1C Cybersecurity",
        },
        {
            "id": "Q10",
            "question": "Did the auditor conclude that internal control over financial reporting was effective as of December 31, 2024?",
            "relevant_year": 2025,
            "relevant_section_hint": "Item 9A Controls and Procedures",
        },
    ]

    labeling_rules = [
        "Do not invent facts that are not in the excerpt. Historical results, tables, and auditor opinions are not forward-looking commitments.",
        "A promise is a management intention, plan, authorization, or target about future activity. Accounting policy text ('if the Company intends to sell AFS securities') is not a business commitment unless labeled as such.",
        "has_deadline is true only when a calendar date, year, or named period (e.g. May 2023, year-end 2023, December 31, 2024) is attached to the commitment.",
        "Domains: Finance = capital, buybacks, dividends, reserves, investments; Operations = workforce, DEI, restructuring, disclosure practices; Technology = cyber, IT, platforms; Claims = claims handling; Growth = premium/market expansion.",
        "Delivered: later excerpt shows the commitment completed or the dated program expired after being used as described, with matching metrics or an explicit expiry-plus-execution statement.",
        "Partial: topic confirmed with incomplete metrics, discretionary remaining capacity, or nearly-complete dollar targets (e.g. $124 of $126).",
        "Pending: stated deadline has not yet been reached in the evidence year, even if there is progress. Progress without a passed deadline is not Delivered.",
        "Silently Abandoned: deadline has passed (or there is no deadline and the evidence is unrelated) and the excerpt does not address the commitment. Do not treat unrelated ICFR/table text as delivery.",
        "Adjacent judge errors: Delivered↔Partial and Partial↔Pending. Far errors: Delivered↔Silently Abandoned and Pending↔Silently Abandoned.",
        "The 0–100 credibility rollup is not gold. Score bands are a coarse check on judge calibration only.",
        "Issuer is The Hartford (HIG) only. Excerpts are whitespace-normalized copies from cached 10-K text.",
    ]

    # Validate
    assert len(extract_gold) == 20
    assert len(judge_gold) == 15
    assert len(qa_gold) == 10
    none_count = sum(1 for item in extract_gold if not item["gold"])
    deadline_count = sum(
        1 for item in extract_gold for g in item["gold"] if g["has_deadline"]
    )
    if none_count < 3:
        raise SystemExit(f"Need >=3 no-promise excerpts, got {none_count}")
    if deadline_count < 3:
        raise SystemExit(f"Need >=3 deadline promises, got {deadline_count}")
    for item in extract_gold:
        wc = len(item["excerpt"].split())
        if wc > MAX_WORDS:
            raise SystemExit(f"{item['id']} excerpt has {wc} words")
        for g in item["gold"]:
            if g["domain"] not in DOMAINS:
                raise SystemExit(f"Bad domain {g['domain']}")
    for item in judge_gold:
        if item["gold_status"] not in STATUSES:
            raise SystemExit(f"Bad status {item['gold_status']}")
        if item["gold_score_band"] not in BANDS:
            raise SystemExit(f"Bad band {item['id']} {item['gold_score_band']}")
        if not item.get("evidence_excerpt"):
            raise SystemExit(f"{item['id']} missing evidence")

    payload = {
        "meta": {
            "issuer": "HIG",
            "issuer_name": "The Hartford",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Cached SEC 10-K plaintext in data/filings/HIG_YYYY.txt (EDGAR HTML parse, whitespace-normalized).",
            "n_extract": 20,
            "n_judge": 15,
            "n_qa": 10,
            "n_extract_none": none_count,
            "n_extract_with_deadline": deadline_count,
            "caveat": "Hand-labeled from excerpts; HIG only; not a statistically powered eval.",
        },
        "labeling_rules": labeling_rules,
        "extract_gold": extract_gold,
        "judge_gold": judge_gold,
        "qa_gold": qa_gold,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} extract={len(extract_gold)} none={none_count} deadline_items={deadline_count} judge={len(judge_gold)}")


if __name__ == "__main__":
    main()

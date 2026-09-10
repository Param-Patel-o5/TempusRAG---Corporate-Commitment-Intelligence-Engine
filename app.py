"""TempusRAG dashboard — corporate commitment intelligence over SEC 10-Ks."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import streamlit as st

from src.config import GEMINI_API_KEY, GROQ_API_KEY
from src.ingestion import load_company_directory
from src.models import CompanyCredibilityReport, DeliveryEvidence
from src.pipeline import get_last_artifacts, run_tempusrag_pipeline
from src.query import process_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="TempusRAG",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STATUS_ORDER = ("Delivered", "Partial", "Silently Abandoned", "Pending")


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "ticker": "HIG",
        "company_name": "The Hartford",
        "issuer_query": "HIG",
        "report": None,
        "evidence": [],
        "conversation": [],
        "gemini_override_key": "",
        "groq_override_key": "",
        "use_custom_gemini": False,
        "use_custom_groq": False,
        "analysis_running": False,
        "selected_row": 0,
        "status_filter": "All",
        "elapsed_seconds": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], .stMarkdown, p, label {
    font-family: "Instrument Sans", "Segoe UI", sans-serif;
}

.stApp {
    background: #26251F;
    color: #F4F1EA;
}

#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}

.tr-masthead {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid #4A463C;
    padding-bottom: 0.95rem;
    margin-bottom: 1.5rem;
}
.tr-wordmark {
    font-family: "Fraunces", "Times New Roman", serif;
    font-optical-sizing: auto;
    font-size: 1.55rem;
    font-weight: 550;
    letter-spacing: -0.02em;
    color: #D97757;
    margin: 0;
    line-height: 1.1;
}
.tr-sub {
    font-size: 0.88rem;
    color: #B5AFA3;
    margin: 0.35rem 0 0 0;
}
.tr-meta {
    font-size: 0.75rem;
    color: #B5AFA3;
    text-align: right;
}

.tr-score-wrap {
    border: 1px solid #4A463C;
    background: #32302A;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1rem;
    border-radius: 10px;
}
.tr-score-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B5AFA3;
}
.tr-score {
    font-family: "Fraunces", "Times New Roman", serif;
    font-size: 3.4rem;
    font-weight: 550;
    line-height: 1;
    color: #F4F1EA;
    margin: 0.35rem 0 0.55rem 0;
}
.tr-score span {
    font-family: "Instrument Sans", sans-serif;
    font-size: 1rem;
    color: #B5AFA3;
}
.tr-thesis {
    font-size: 0.98rem;
    line-height: 1.55;
    color: #E4DFD4;
    margin: 0;
    max-width: 52rem;
}

.tr-kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #4A463C;
    border: 1px solid #4A463C;
    margin-bottom: 1.25rem;
    border-radius: 10px;
    overflow: hidden;
}
.tr-kpi {
    background: #32302A;
    padding: 0.95rem 1rem;
}
.tr-kpi .n {
    font-family: "Fraunces", serif;
    font-size: 1.55rem;
    color: #F4F1EA;
}
.tr-kpi .l {
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #B5AFA3;
    margin-top: 0.15rem;
}

.tr-flag {
    border-left: 2px solid #D97757;
    padding: 0.45rem 0.75rem;
    color: #E4DFD4;
    font-size: 0.88rem;
    margin-bottom: 0.4rem;
    background: #32302A;
    border-radius: 0 6px 6px 0;
}

.tr-cite {
    font-size: 0.75rem;
    color: #B5AFA3;
}

.tr-picked {
    font-size: 0.8rem;
    color: #B5AFA3;
    margin: 0.1rem 0 0.35rem 0;
}
.tr-picked strong {
    color: #F4F1EA;
    font-weight: 600;
}

.tr-stages {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.55rem;
    padding: 0.75rem 0.9rem;
    background: #32302A;
    border: 1px solid #4A463C;
    border-radius: 10px;
    margin: 0.5rem 0 0.75rem 0;
}
.tr-stage {
    font-size: 0.78rem;
    color: #7A756C;
    letter-spacing: 0.04em;
}
.tr-stage.is-done {
    color: #B5AFA3;
}
.tr-stage.is-active {
    color: #D97757;
    font-weight: 600;
}

div[data-testid="stSidebar"] {
    background: #32302A;
}

.stButton > button[kind="primary"] {
    background: #D97757;
    color: #26251F;
    border: none;
    border-radius: 8px;
    font-family: "Instrument Sans", sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.stButton > button[kind="secondary"] {
    background: transparent;
    color: #F4F1EA;
    border: 1px solid #4A463C;
    border-radius: 8px;
}

[data-testid="stMetricValue"] {
    font-family: "Fraunces", serif;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading SEC issuer list…")
def issuer_directory() -> list[dict[str, str]]:
    return load_company_directory()


def search_issuers(query: str, directory: list[dict[str, str]], limit: int = 8) -> list[dict[str, str]]:
    q = (query or "").strip().lower()
    if not q:
        return []

    exact_ticker: list[dict[str, str]] = []
    ticker_prefix: list[dict[str, str]] = []
    name_prefix: list[dict[str, str]] = []
    name_contains: list[dict[str, str]] = []
    ticker_contains: list[dict[str, str]] = []

    for row in directory:
        ticker = row["ticker"]
        title = row["title"]
        t_low = ticker.lower()
        n_low = title.lower()
        if t_low == q:
            exact_ticker.append(row)
        elif t_low.startswith(q):
            ticker_prefix.append(row)
        elif n_low.startswith(q):
            name_prefix.append(row)
        elif q in n_low:
            name_contains.append(row)
        elif q in t_low:
            ticker_contains.append(row)

    ranked = exact_ticker + ticker_prefix + name_prefix + name_contains + ticker_contains
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in ranked:
        if row["ticker"] in seen:
            continue
        seen.add(row["ticker"])
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique


def resolve_issuer(query: str, directory: list[dict[str, str]]) -> dict[str, str] | None:
    matches = search_issuers(query, directory, limit=8)
    if not matches:
        return None
    q = (query or "").strip().upper()
    for row in matches:
        if row["ticker"] == q:
            return row
    return matches[0]


def get_active_api_keys() -> tuple[str, str]:
    gemini = (
        st.session_state.gemini_override_key
        if st.session_state.use_custom_gemini and st.session_state.gemini_override_key
        else GEMINI_API_KEY
    )
    groq = (
        st.session_state.groq_override_key
        if st.session_state.use_custom_groq and st.session_state.groq_override_key
        else GROQ_API_KEY
    )
    return gemini or "", groq or ""


def key_label(present: bool, custom: bool) -> str:
    if not present:
        return "missing"
    return "session" if custom else "configured"


def render_sidebar() -> None:
    st.sidebar.markdown("**Credentials**")
    st.sidebar.caption("Keys stay in this browser session. They are not written to disk.")

    gemini_key, groq_key = get_active_api_keys()
    st.sidebar.text(
        f"Gemini  {key_label(bool(gemini_key), st.session_state.use_custom_gemini)}\n"
        f"Groq    {key_label(bool(groq_key), st.session_state.use_custom_groq)}"
    )

    st.session_state.use_custom_gemini = st.sidebar.checkbox(
        "Override Gemini key", value=st.session_state.use_custom_gemini
    )
    if st.session_state.use_custom_gemini:
        st.session_state.gemini_override_key = st.sidebar.text_input(
            "Gemini API key",
            type="password",
            value=st.session_state.gemini_override_key,
        )

    st.session_state.use_custom_groq = st.sidebar.checkbox(
        "Override Groq key", value=st.session_state.use_custom_groq
    )
    if st.session_state.use_custom_groq:
        st.session_state.groq_override_key = st.sidebar.text_input(
            "Groq API key",
            type="password",
            value=st.session_state.groq_override_key,
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Extraction uses Gemini. Cross-year judgment uses Groq. "
        "Free tiers are rate-limited; cached filings skip a full re-ingest."
    )


PIPELINE_STAGES = ("Validate", "Ingest", "Chunk", "Embed", "Extract", "Judge", "Score", "Done")


def stages_html(current: str) -> str:
    reached = True
    parts = ['<div class="tr-stages">']
    for name in PIPELINE_STAGES:
        cls = "tr-stage"
        if name == current:
            cls += " is-active"
            reached = False
        elif reached:
            cls += " is-done"
        parts.append(f'<span class="{cls}">{name}</span>')
    parts.append("</div>")
    return "".join(parts)


def classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "quota" in msg:
        return (
            "Provider rate limit reached. Wait and retry, or paste your own keys "
            "in the sidebar. Cached filings are reused when available."
        )
    if "401" in msg or "unauthorized" in msg or "api key" in msg:
        return "API key rejected. Check GEMINI_API_KEY / GROQ_API_KEY or the sidebar overrides."
    if "not found" in msg and "ticker" in msg:
        return str(exc)
    if "network" in msg or "connection" in msg or "edgar" in msg:
        return f"Upstream request failed: {exc}"
    return f"Analysis failed: {exc}"


def run_analysis(ticker: str) -> None:
    gemini_key, groq_key = get_active_api_keys()
    if not gemini_key:
        st.error("Gemini key required for promise extraction. Set GEMINI_API_KEY or use the sidebar.")
        return
    if not groq_key:
        st.error("Groq key required for delivery judgment. Set GROQ_API_KEY or use the sidebar.")
        return

    st.session_state.analysis_running = True
    try:
        stage_box = st.empty()
        stage_box.markdown(stages_html("Validate"), unsafe_allow_html=True)

        def on_stage(label: str) -> None:
            stage_box.markdown(stages_html(label), unsafe_allow_html=True)

        report = run_tempusrag_pipeline(
            ticker=ticker,
            company_display_name=st.session_state.get("company_name") or ticker,
            gemini_api_key=gemini_key,
            groq_api_key=groq_key,
            force_reingest=False,
            on_stage=on_stage,
        )
        artifacts = get_last_artifacts(ticker) or {}
        stage_box.markdown(stages_html("Done"), unsafe_allow_html=True)

        st.session_state.report = report
        st.session_state.evidence = artifacts.get("delivery_evidence") or []
        st.session_state.elapsed_seconds = artifacts.get("elapsed_seconds")
        st.session_state.conversation = []
        st.session_state.selected_row = 0
        st.rerun()
    except Exception as exc:
        logger.exception("Pipeline failed for %s", ticker)
        st.error(classify_error(exc))
    finally:
        st.session_state.analysis_running = False


def evidence_rows(evidence: list[DeliveryEvidence]) -> list[dict[str, Any]]:
    rows = []
    for item in evidence:
        rows.append(
            {
                "Year": item.promise.year_made,
                "Domain": item.promise.domain,
                "Status": item.status,
                "Score": item.delivery_score,
                "Deadline": item.promise.deadline_mentioned or "—",
                "Found": item.year_found if item.year_found is not None else "—",
                "Commitment": item.promise.promise_text,
                "Evidence": item.evidence_text or "—",
                "Reasoning": item.judge_reasoning,
                "Confidence": item.promise.confidence_score if item.promise.confidence_score is not None else "—",
                "Page": item.promise.page_number,
            }
        )
    return rows


def render_masthead(report: CompanyCredibilityReport | None) -> None:
    years = ""
    elapsed = ""
    if report:
        years = " · ".join(str(y) for y in report.years_analyzed) or "—"
        if st.session_state.elapsed_seconds is not None:
            elapsed = f"{st.session_state.elapsed_seconds:.0f}s last run"
    st.markdown(
        f"""
<div class="tr-masthead">
  <div>
    <p class="tr-wordmark">TempusRAG</p>
    <p class="tr-sub">Executive commitment tracking from temporally indexed 10-K filings</p>
  </div>
  <div class="tr-meta">{years}<br>{elapsed}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_toolbar() -> None:
    directory: list[dict[str, str]] = []
    try:
        directory = issuer_directory()
    except Exception as exc:
        logger.exception("Failed to load SEC issuer directory")
        st.session_state["_directory_error"] = str(exc)

    if "issuer_search_box" not in st.session_state:
        st.session_state.issuer_search_box = st.session_state.issuer_query

    search_col, btn_col = st.columns([4.2, 1.0])
    with search_col:
        st.text_input(
            "Issuer",
            placeholder="Company name or ticker — Apple, Hartford, AAPL, HIG",
            label_visibility="collapsed",
            key="issuer_search_box",
        )
        st.session_state.issuer_query = st.session_state.issuer_search_box or ""

    matches = search_issuers(st.session_state.issuer_query, directory, limit=8)

    if directory and st.session_state.issuer_query.strip() and matches:
        labels = [f"{m['ticker']}  —  {m['title']}" for m in matches]
        current_label = None
        for label, row in zip(labels, matches):
            if row["ticker"] == st.session_state.ticker:
                current_label = label
                break
        index = labels.index(current_label) if current_label in labels else 0
        picked = st.selectbox(
            "Matching issuers",
            labels,
            index=index,
            help="Resolved from the SEC issuer list.",
        )
        chosen = matches[labels.index(picked)]
        st.session_state.ticker = chosen["ticker"]
        st.session_state.company_name = chosen["title"]
    elif st.session_state.issuer_query.strip() and not matches:
        st.caption("No listed match.")
    else:
        resolved = resolve_issuer(st.session_state.ticker, directory)
        if resolved:
            st.session_state.company_name = resolved["title"]

    st.markdown(
        f'<p class="tr-picked"><strong>{st.session_state.ticker}</strong>'
        f" · {st.session_state.company_name}</p>",
        unsafe_allow_html=True,
    )

    with btn_col:
        analyze = st.button(
            "Analyze",
            type="primary",
            disabled=st.session_state.analysis_running,
            use_container_width=True,
        )

    if st.session_state.get("_directory_error") and not directory:
        st.warning(
            "Could not load the SEC issuer directory. You can still enter a ticker if it is known. "
            f"({st.session_state['_directory_error']})"
        )

    if analyze:
        ticker = st.session_state.ticker
        if not re.match(r"^[A-Z][A-Z0-9.-]{0,9}$", ticker):
            fallback = resolve_issuer(st.session_state.issuer_query, directory)
            if fallback:
                ticker = fallback["ticker"]
                st.session_state.ticker = ticker
                st.session_state.company_name = fallback["title"]
            else:
                st.error("Select a matching issuer from the list, or enter a listed ticker.")
                return
        run_analysis(ticker)


def render_hero(report: CompanyCredibilityReport) -> None:
    total = report.total_promises or 0
    delivered_pct = (report.delivered / total * 100) if total else 0.0
    years_n = len(report.years_analyzed)
    thesis = (
        f"{report.company} delivered on {report.delivered} of {total} extracted forward-looking "
        f"commitments across {years_n} fiscal year{'s' if years_n != 1 else ''} "
        f"({delivered_pct:.0f}% delivered). "
        f"{len(report.red_flags)} risk flag{'s' if len(report.red_flags) != 1 else ''}."
    )
    st.markdown(
        f"""
<div class="tr-score-wrap">
  <div class="tr-score-label">Credibility score</div>
  <div class="tr-score">{report.overall_score:.0f}<span> / 100</span></div>
  <p class="tr-thesis">{thesis}</p>
</div>
<div class="tr-kpis">
  <div class="tr-kpi"><div class="n">{report.delivered}</div><div class="l">Delivered</div></div>
  <div class="tr-kpi"><div class="n">{report.partial}</div><div class="l">Partial</div></div>
  <div class="tr-kpi"><div class="n">{report.abandoned}</div><div class="l">Abandoned</div></div>
  <div class="tr-kpi"><div class="n">{total}</div><div class="l">Commitments</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_domains_and_flags(report: CompanyCredibilityReport) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Domain scores**")
        domains = {k: v for k, v in report.domain_scores.items() if v is not None}
        if not domains:
            st.caption("No domain scores yet.")
        else:
            for name, score in sorted(domains.items(), key=lambda x: x[1], reverse=True):
                st.progress(min(max(score / 100.0, 0.0), 1.0), text=f"{name}  {score:.0f}")
    with right:
        st.markdown("**Flags**")
        if not report.red_flags:
            st.caption("None raised by the rule layer.")
        else:
            for flag in report.red_flags:
                st.markdown(f'<div class="tr-flag">{flag}</div>', unsafe_allow_html=True)


def render_commitments() -> None:
    evidence: list[DeliveryEvidence] = st.session_state.evidence or []
    st.markdown("**Commitments**")
    if not evidence:
        st.caption(
            "Row-level judgments were not returned for this run. "
            "Headline counts above still come from the scored report."
        )
        return

    rows = evidence_rows(evidence)
    statuses = ["All", *STATUS_ORDER]
    filter_col, _ = st.columns([1.2, 3])
    with filter_col:
        choice = st.selectbox("Status", statuses, label_visibility="collapsed")
        st.session_state.status_filter = choice

    if choice != "All":
        rows = [r for r in rows if r["Status"] == choice]

    display = [
        {k: r[k] for k in ("Year", "Domain", "Status", "Score", "Deadline", "Commitment")}
        for r in rows
    ]
    event = st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        height=360,
    )

    selected = 0
    try:
        selected_rows = event.selection.rows  # type: ignore[attr-defined]
        if selected_rows:
            selected = int(selected_rows[0])
    except Exception:
        selected = 0

    if not rows:
        st.caption("No commitments in this filter.")
        return

    selected = min(selected, len(rows) - 1)
    detail = rows[selected]
    st.markdown("**Selected judgment**")
    st.write(detail["Commitment"])
    st.caption(
        f"{detail['Domain']} · made {detail['Year']} · deadline {detail['Deadline']} · "
        f"evidence year {detail['Found']} · page {detail['Page']} · "
        f"confidence {detail['Confidence']}"
    )
    st.markdown("**Evidence**")
    st.write(detail["Evidence"])
    st.markdown("**Judge reasoning**")
    st.write(detail["Reasoning"])


def conversation_as_strings() -> list[str]:
    out: list[str] = []
    for message in st.session_state.conversation:
        role = message.get("role", "")
        content = message.get("content", "")
        out.append(f"{role}: {content}")
    return out


def format_chunk_cite(chunk: Any) -> str:
    year = getattr(chunk, "year", "?")
    section = getattr(chunk, "section", "")
    subsection = getattr(chunk, "subsection", "")
    path = " > ".join(p for p in (section, subsection) if p)
    return f"{year} {path}".strip()


def render_ask() -> None:
    st.markdown("**Filing Q&A**")
    st.caption("Ask after the report — answers cite retrieved 10-K chunks.")

    if not st.session_state.conversation:
        st.caption("No questions yet.")
        return

    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            cites = message.get("cites") or []
            if cites:
                st.markdown(
                    '<p class="tr-cite">' + " · ".join(cites) + "</p>",
                    unsafe_allow_html=True,
                )


def handle_chat_prompt(prompt: str) -> None:
    st.session_state.conversation.append({"role": "user", "content": prompt})
    gemini_key, _ = get_active_api_keys()
    try:
        result = process_query(
            query=prompt,
            ticker=st.session_state.ticker,
            conversation_history=conversation_as_strings()[:-1],
            api_key=gemini_key,
        )
        chunks = result.get("source_chunks") or []
        cites = [format_chunk_cite(c) for c in chunks[:5]]
        st.session_state.conversation.append(
            {
                "role": "assistant",
                "content": result.get("answer") or "No answer generated.",
                "cites": cites,
            }
        )
    except Exception as exc:
        logger.exception("Query failed")
        st.session_state.conversation.append(
            {"role": "assistant", "content": classify_error(exc)}
        )
    st.rerun()


def render_empty() -> None:
    st.caption("Demo: HIG. Score is an LLM judge, not an audit.")


def main() -> None:
    init_session_state()
    inject_css()
    render_sidebar()
    render_masthead(st.session_state.report)
    render_toolbar()

    report = st.session_state.report
    if report is None:
        render_empty()
        return

    render_hero(report)
    render_domains_and_flags(report)
    st.divider()
    table_col, ask_col = st.columns([1.55, 1])
    with table_col:
        render_commitments()
    with ask_col:
        render_ask()

    prompt = st.chat_input("Ask about this issuer’s filings")
    if prompt:
        handle_chat_prompt(prompt)


if __name__ == "__main__":
    main()

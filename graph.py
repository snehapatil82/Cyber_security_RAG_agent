"""
core/graph.py — LangGraph workflow for SENTRY.

    User Query
       |
    detect_intent
       |
   +---+----------------------------+
   |                                |
 cve_query/comparison/filter   document_query / general
   |                                |
 retrieve_live (NVD+KEV)      retrieve_reports (Chroma)
   |                                |
 retrieve_cve_cache (Chroma)        |
   +---------------+----------------+
                    |
              merge_context
                    |
              generate_answer
                    |
               compute_risk   (only if CVE(s) resolved)
                    |
             compute_confidence
                    |
                   END

This mirrors the spec's two branches (CVE path / document path) as a single
graph with a conditional edge, which keeps the workflow in one place and
easy to extend.
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from core import prompts
from core.confidence import compute_confidence
from core.llm import get_llm_client
from core.nvd_client import CVERecord, extract_cve_ids, fetch_cve
from core.risk_engine import RiskAssessment, assess_risk
from core.vectorstore import get_store

logger = logging.getLogger("sentry.graph")


class SentryState(TypedDict, total=False):
    query: str
    mode: str  # "technical" | "beginner"
    intent: str
    cve_ids: list[str]
    cve_records: dict[str, CVERecord]
    retrieved: list[dict[str, Any]]
    context: str
    answer: str
    risk: dict[str, RiskAssessment]
    confidence: Any
    errors: list[str]


def node_detect_intent(state: SentryState) -> SentryState:
    query = state["query"]
    cve_ids = extract_cve_ids(query)
    state["cve_ids"] = cve_ids

    if cve_ids and len(cve_ids) >= 2 and ("compar" in query.lower() or " vs " in query.lower()):
        state["intent"] = "comparison"
    elif cve_ids:
        state["intent"] = "cve_query"
    elif any(kw in query.lower() for kw in ["this report", "uploaded", "summarize", "summary", "executive summary", "extract all cve", "recommendations"]):
        state["intent"] = "document_query"
    elif any(kw in query.lower() for kw in ["only critical", "which vulnerabilities", "affect windows", "affect apache", "patch first", "filter"]):
        state["intent"] = "filter_query"
    else:
        state["intent"] = "general"
    return state


def node_retrieve_live(state: SentryState) -> SentryState:
    store = get_store()
    records: dict[str, CVERecord] = {}
    errors = state.get("errors", [])
    for cve_id in state.get("cve_ids", []):
        cached = store.cve_cached(cve_id)
        if cached:
            # Reconstruct a minimal record from cached metadata + text for downstream use.
            meta = cached["metadata"]
            record = CVERecord(
                cve_id=cve_id,
                description=cached["text"],
                cvss_score=meta.get("cvss_score"),
                cvss_severity=meta.get("cvss_severity", "UNKNOWN"),
                exploited_kev=bool(meta.get("exploited_kev", False)),
                patch_available=meta.get("patch_available"),
            )
            records[cve_id] = record
            continue
        try:
            record = fetch_cve(cve_id)
        except Exception as exc:
            errors.append(f"NVD lookup failed for {cve_id}: {exc}")
            continue
        if record is None:
            errors.append(f"No NVD data found for {cve_id}.")
            continue
        records[cve_id] = record
        store.upsert_cve(
            cve_id,
            record.to_text(),
            {
                "cvss_score": record.cvss_score,
                "cvss_severity": record.cvss_severity,
                "exploited_kev": record.exploited_kev,
                "patch_available": record.patch_available,
            },
        )
    state["cve_records"] = records
    state["errors"] = errors
    return state


def node_retrieve_reports(state: SentryState) -> SentryState:
    store = get_store()
    results = store.query_reports(state["query"], k=5)
    state["retrieved"] = state.get("retrieved", []) + results
    return state


def node_retrieve_cve_cache(state: SentryState) -> SentryState:
    store = get_store()
    results = store.query_cves(state["query"], k=3)
    state["retrieved"] = state.get("retrieved", []) + results
    return state


def node_merge_context(state: SentryState) -> SentryState:
    parts = []
    for cve_id, record in state.get("cve_records", {}).items():
        parts.append(f"[Source: {cve_id} (NVD/CISA KEV)]\n{record.to_text()}")
    for r in state.get("retrieved", []):
        src = r["metadata"].get("source") or r["metadata"].get("cve_id", "unknown")
        parts.append(f"[Source: {src}]\n{r['text']}")
    state["context"] = "\n\n---\n\n".join(parts)
    return state


def node_generate_answer(state: SentryState) -> SentryState:
    client = get_llm_client()
    prompt = prompts.build_answer_prompt(state["query"], state.get("context", ""), state.get("mode", "technical"))
    try:
        response = client.generate(prompt)
        state["answer"] = response.text
    except Exception as exc:
        state["answer"] = (
            "I couldn't generate an answer because no LLM provider is available or the call failed "
            f"({exc}). Please check your API key in Settings."
        )
    return state


def node_compute_risk(state: SentryState) -> SentryState:
    risk: dict[str, RiskAssessment] = {}
    for cve_id, record in state.get("cve_records", {}).items():
        risk[cve_id] = assess_risk(record)
    state["risk"] = risk
    return state


def node_compute_confidence(state: SentryState) -> SentryState:
    has_live = bool(state.get("cve_records"))
    state["confidence"] = compute_confidence(state.get("retrieved", []), has_live_cve_data=has_live)
    return state


def route_after_intent(state: SentryState) -> str:
    if state["intent"] in ("cve_query", "comparison", "filter_query") and state.get("cve_ids"):
        return "cve_path"
    return "doc_path"


def build_graph():
    graph = StateGraph(SentryState)
    graph.add_node("detect_intent", node_detect_intent)
    graph.add_node("retrieve_live", node_retrieve_live)
    graph.add_node("retrieve_cve_cache", node_retrieve_cve_cache)
    graph.add_node("retrieve_reports", node_retrieve_reports)
    graph.add_node("merge_context", node_merge_context)
    graph.add_node("generate_answer", node_generate_answer)
    graph.add_node("compute_risk", node_compute_risk)
    graph.add_node("compute_confidence", node_compute_confidence)

    graph.set_entry_point("detect_intent")
    graph.add_conditional_edges(
        "detect_intent", route_after_intent, {"cve_path": "retrieve_live", "doc_path": "retrieve_reports"}
    )
    graph.add_edge("retrieve_live", "retrieve_cve_cache")
    graph.add_edge("retrieve_cve_cache", "retrieve_reports")
    graph.add_edge("retrieve_reports", "merge_context")
    graph.add_edge("merge_context", "generate_answer")
    graph.add_edge("generate_answer", "compute_risk")
    graph.add_edge("compute_risk", "compute_confidence")
    graph.add_edge("compute_confidence", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_investigation(query: str, mode: str = "technical") -> SentryState:
    initial: SentryState = {"query": query, "mode": mode, "errors": []}
    graph = get_graph()
    result = graph.invoke(initial)
    return result

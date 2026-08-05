"""
SENTRY — Cyber Security Threat Intelligence Assistant (Enterprise SOC Edition)

Run:
    streamlit run app.py

Secrets (env vars or .streamlit/secrets.toml):
    GEMINI_API_KEY   (preferred LLM)
    GROQ_API_KEY     (fallback LLM)
    NVD_API_KEY      (optional — raises NVD rate limits)
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from core import prompts  # noqa: F401  (imported for side-effect clarity / future use)
from core.config import settings
from core.confidence import ConfidenceResult
from core.graph import run_investigation
from core.ingestion import chunk_text, extract_text
from core.llm import get_llm_client, reset_llm_client
from core.nvd_client import get_kev_catalog
from core.report_generator import InvestigationReport, generate_docx, generate_pdf
from core.risk_engine import priority_from_score
from core.storage import list_report_files, load_history, log_investigation, save_report_file
from core.vectorstore import get_store

st.set_page_config(page_title="SENTRY // Threat Intel Console", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Styling (SOC console theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#05070B; --panel:#0D131C; --panel-2:#131B27; --border:#1F2B3B; --text:#E7EEF5; --muted:#7688A0;
  --accent:#2FE6C7; --accent-dim:#0F8F7C; --accent2:#7B61FF; --danger:#FF4D5E; --warn:#FFB627; --low:#4ADE80;
}
[data-testid="stAppViewContainer"]{
  background: radial-gradient(160% 120% at 15% -10%, rgba(123,97,255,0.07), transparent 55%),
    radial-gradient(120% 100% at 90% 0%, rgba(47,230,199,0.06), transparent 50%), var(--bg);
}
[data-testid="stHeader"]{ background: transparent; }
html, body, [class*="css"]{ font-family:'Inter', sans-serif; color: var(--text); }
h1,h2,h3,h4,h5{ font-family:'Space Grotesk', sans-serif; }
.sentry-hero{ border:1px solid var(--border); border-radius:18px; background: linear-gradient(180deg, #0B1119 0%, #070B11 100%); padding:26px 32px; margin-bottom:22px; }
.sentry-title{ font-family:'Space Grotesk', sans-serif; font-size:1.8rem; font-weight:700; margin:0; color:#F3F8FC; }
.sentry-sub{ font-size:0.92rem; color:var(--muted); margin-top:4px; }
.console-label{ font-family:'JetBrains Mono', monospace; font-size:0.74rem; letter-spacing:0.14em; color:var(--accent); text-transform:uppercase; margin:14px 0 8px 0; }
[data-testid="stSidebar"]{ background: var(--panel); border-right:1px solid var(--border); }
.stButton button, .stFormSubmitButton button, .stDownloadButton button{
  background: linear-gradient(145deg, var(--accent-dim), var(--accent)) !important; color:#03110D !important;
  font-weight:700 !important; border:none !important; border-radius:8px !important; font-family:'JetBrains Mono', monospace !important;
}
[data-testid="stChatMessage"]{ background: var(--panel) !important; border:1px solid var(--border) !important; border-radius:12px !important; }
.metric-card{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:18px 20px; text-align:center; }
.metric-value{ font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; }
.metric-label{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; letter-spacing:0.08em; color:var(--muted); text-transform:uppercase; }
.sev-chip{ display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-family:'JetBrains Mono', monospace; font-size:0.72rem; font-weight:700; }
.sev-critical{ background:rgba(255,77,94,0.12); color:var(--danger); border:1px solid rgba(255,77,94,0.35); }
.sev-high{ background:rgba(255,182,39,0.12); color:var(--warn); border:1px solid rgba(255,182,39,0.35); }
.sev-medium{ background:rgba(255,182,39,0.10); color:var(--warn); border:1px solid rgba(255,182,39,0.25); }
.sev-low{ background:rgba(74,222,128,0.12); color:var(--low); border:1px solid rgba(74,222,128,0.35); }
.sev-unknown, .sev-na{ background:rgba(124,139,158,0.12); color:var(--muted); border:1px solid rgba(124,139,158,0.35); }
.src-chip{ display:inline-block; font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:var(--accent);
  background:rgba(47,230,199,0.08); border:1px solid rgba(47,230,199,0.25); padding:2px 8px; border-radius:6px; margin:2px 4px 2px 0; }
.risk-meter-track{ width:100%; height:10px; background:var(--panel-2); border-radius:999px; overflow:hidden; border:1px solid var(--border); }
.risk-meter-fill{ height:100%; border-radius:999px; }
</style>
""",
    unsafe_allow_html=True,
)

PRIORITY_CLASS = {"Critical": "sev-critical", "High": "sev-high", "Medium": "sev-medium", "Low": "sev-low"}


def sev_chip(label: str, extra: str = "") -> str:
    css = PRIORITY_CLASS.get(label, "sev-unknown")
    suffix = f" · {extra}" if extra else ""
    return f'<span class="sev-chip {css}">{label}{suffix}</span>'


def risk_meter(score: int) -> str:
    color = "var(--danger)" if score >= 80 else "var(--warn)" if score >= 60 else "#F2C744" if score >= 35 else "var(--low)"
    return (
        f'<div class="risk-meter-track"><div class="risk-meter-fill" '
        f'style="width:{score}%;background:{color};"></div></div>'
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "messages": [],
    "explanation_mode": "technical",
    "last_investigation": None,
    "nav": "Dashboard",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

store = get_store()
llm = get_llm_client()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
        '<span style="font-size:1.6rem;">🛡️</span>'
        '<span style="font-family:\'Space Grotesk\',sans-serif;font-weight:700;font-size:1.2rem;">SENTRY</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Threat Intelligence RAG Assistant")
    st.session_state.nav = st.radio(
        "Navigate",
        ["Dashboard", "Investigate (Chat)", "Upload Report", "Investigation History", "Reports", "Settings"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<p class="console-label">⚙️ Explanation Mode</p>', unsafe_allow_html=True)
    st.session_state.explanation_mode = st.radio(
        "Mode", ["technical", "beginner"], horizontal=True, label_visibility="collapsed",
        format_func=lambda m: "🧪 Technical" if m == "technical" else "🧑\u200d🎓 Beginner",
    )
    st.divider()
    st.markdown('<p class="console-label">📊 Mission Stats</p>', unsafe_allow_html=True)
    report_sources = store.report_sources()
    cached_cves = store.all_cached_cves()
    st.markdown(
        f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--muted);">
        <div>Reports indexed: <b style="color:var(--text)">{len(report_sources)}</b></div>
        <div>CVEs cached: <b style="color:var(--text)">{len(cached_cves)}</b></div>
        <div>LLM ready: <b style="color:var(--text)">{"Yes" if llm.available else "No"}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not llm.available:
        st.warning("No LLM configured. Go to Settings to add a Gemini or Groq API key.")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sentry-hero"><p class="sentry-title">🛡️ SENTRY — Threat Intel Console</p>'
    '<p class="sentry-sub">Investigate CVEs, analyze vulnerability reports, and get prioritized, '
    "source-backed answers — powered by live NVD/CISA intel + your uploaded reports.</p></div>",
    unsafe_allow_html=True,
)


# =============================================================================
# PAGE: Dashboard
# =============================================================================
def page_dashboard():
    st.markdown('<p class="console-label">📊 Security Dashboard</p>', unsafe_allow_html=True)

    cves = store.all_cached_cves()
    kev_catalog = get_kev_catalog()
    kev_ids = {e.get("cveID") for e in kev_catalog}

    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    rows = []
    for c in cves:
        score = c.get("cvss_score")
        severity = priority_from_score(int((score or 0) / 10 * 100)) if score is not None else "Low"
        counts[severity] = counts.get(severity, 0) + 1
        rows.append(
            {
                "CVE": c["cve_id"],
                "CVSS": score,
                "Severity": severity,
                "Exploited (KEV)": "Yes" if c["cve_id"] in kev_ids else "No",
                "Cached": pd.to_datetime(c.get("cached_at", 0), unit="s", errors="coerce"),
            }
        )
    df = pd.DataFrame(rows)

    cols = st.columns(4)
    for col, (label, value) in zip(cols, counts.items()):
        css = PRIORITY_CLASS.get(label, "sev-unknown")
        color_var = {"Critical": "var(--danger)", "High": "var(--warn)", "Medium": "#F2C744", "Low": "var(--low)"}[label]
        col.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{color_var}">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    if df.empty:
        st.info("No CVEs investigated yet. Head to **Investigate (Chat)** and ask about a CVE to populate the dashboard.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="console-label">Severity Distribution</p>', unsafe_allow_html=True)
        sev_df = df["Severity"].value_counts().reset_index()
        sev_df.columns = ["Severity", "Count"]
        fig = px.pie(
            sev_df, names="Severity", values="Count", hole=0.55,
            color="Severity",
            color_discrete_map={"Critical": "#FF4D5E", "High": "#FFB627", "Medium": "#F2C744", "Low": "#4ADE80"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E7EEF5", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<p class="console-label">Recently Added CVEs</p>', unsafe_allow_html=True)
        recent = df.sort_values("Cached", ascending=False).head(10)
        fig2 = px.bar(recent, x="CVE", y="CVSS", color="Severity",
                      color_discrete_map={"Critical": "#FF4D5E", "High": "#FFB627", "Medium": "#F2C744", "Low": "#4ADE80"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E7EEF5")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="console-label">🔍 Filter & Browse</p>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    sev_filter = fc1.multiselect("Severity", ["Critical", "High", "Medium", "Low"], default=[])
    exploited_filter = fc2.selectbox("Exploitation Status", ["All", "Actively Exploited", "Not Exploited"])
    search_filter = fc3.text_input("Search CVE ID")

    filtered = df.copy()
    if sev_filter:
        filtered = filtered[filtered["Severity"].isin(sev_filter)]
    if exploited_filter == "Actively Exploited":
        filtered = filtered[filtered["Exploited (KEV)"] == "Yes"]
    elif exploited_filter == "Not Exploited":
        filtered = filtered[filtered["Exploited (KEV)"] == "No"]
    if search_filter:
        filtered = filtered[filtered["CVE"].str.contains(search_filter.upper(), na=False)]

    st.dataframe(filtered.drop(columns=["Cached"]), use_container_width=True, hide_index=True)


# =============================================================================
# PAGE: Investigate (Chat)
# =============================================================================
def page_chat():
    st.markdown('<p class="console-label">💬 Investigate</p>', unsafe_allow_html=True)
    st.caption(
        "Ask about a CVE (e.g. *Explain CVE-2024-3400*), compare CVEs, query an uploaded report, "
        "or filter (*show only Critical vulnerabilities*)."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    query = st.chat_input("Ask SENTRY a question...")
    if not query:
        return

    if not llm.available:
        st.error("No LLM provider configured. Add a Gemini or Groq API key in Settings.")
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("SENTRY is investigating..."):
            result = run_investigation(query, mode=st.session_state.explanation_mode)

        answer = result.get("answer", "No answer generated.")
        st.markdown(answer)

        for err in result.get("errors", []):
            st.warning(err)

        confidence: ConfidenceResult = result["confidence"]
        conf_color = "var(--low)" if confidence.label == "High" else "var(--warn)" if confidence.label == "Medium" else "var(--danger)"
        st.markdown(
            f'<div style="margin-top:8px;"><b>Confidence:</b> '
            f'<span style="color:{conf_color};font-weight:700;">{confidence.score}% ({confidence.label})</span> '
            f"— based on {confidence.num_sources} source(s), {len(result.get('retrieved', []))} retrieved chunk(s).</div>",
            unsafe_allow_html=True,
        )
        if confidence.source_list:
            chips = "".join(f'<span class="src-chip">{s}</span>' for s in confidence.source_list)
            st.markdown(chips, unsafe_allow_html=True)

        risk_data = result.get("risk", {})
        report_payload = None
        if risk_data:
            st.markdown("---")
            for cve_id, assessment in risk_data.items():
                record = result["cve_records"].get(cve_id)
                st.markdown(f"**{cve_id}** — {sev_chip(assessment.priority, f'{assessment.score}/100')}", unsafe_allow_html=True)
                st.markdown(risk_meter(assessment.score), unsafe_allow_html=True)
                with st.expander(f"Why this score? ({cve_id})"):
                    for line in assessment.rationale:
                        st.markdown(f"- {line}")

                report_payload = InvestigationReport(
                    question=query,
                    answer=answer,
                    cve_id=cve_id,
                    severity=record.cvss_severity if record else "Unknown",
                    cvss_score=record.cvss_score if record else None,
                    risk_score=assessment.score,
                    priority=assessment.priority,
                    risk_rationale=assessment.rationale,
                    exploited_kev=record.exploited_kev if record else False,
                    affected_products=record.affected_products if record else [],
                    patch_available=record.patch_available if record else None,
                    mitigation="Apply the vendor patch referenced above. If unavailable, apply vendor-recommended "
                    "workarounds and increase monitoring for exploitation indicators.",
                    references=record.references if record else [],
                    confidence_score=confidence.score,
                    confidence_label=confidence.label,
                    sources=confidence.source_list,
                )
        else:
            report_payload = InvestigationReport(
                question=query,
                answer=answer,
                confidence_score=confidence.score,
                confidence_label=confidence.label,
                sources=confidence.source_list,
            )

        with st.expander("📎 Retrieved sources"):
            for r in result.get("retrieved", []):
                src = r["metadata"].get("source") or r["metadata"].get("cve_id", "unknown")
                st.markdown(f'<span class="src-chip">{src}</span> similarity: {r["similarity"]:.2f}', unsafe_allow_html=True)
                st.caption(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))

        # Export buttons
        ec1, ec2 = st.columns(2)
        pdf_bytes = generate_pdf(report_payload)
        docx_bytes = generate_docx(report_payload)
        stamp = int(time.time())
        pdf_name = f"sentry_investigation_{stamp}.pdf"
        docx_name = f"sentry_investigation_{stamp}.docx"
        with ec1:
            if st.download_button("⬇️ Download Investigation Report (PDF)", data=pdf_bytes, file_name=pdf_name, mime="application/pdf"):
                save_report_file(pdf_name, pdf_bytes)
        with ec2:
            if st.download_button("⬇️ Download Investigation Report (DOCX)", data=docx_bytes, file_name=docx_name,
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
                save_report_file(docx_name, docx_bytes)
        # Always persist a copy so it shows up under Reports / History regardless of click
        save_report_file(pdf_name, pdf_bytes)
        save_report_file(docx_name, docx_bytes)

        log_investigation(
            {
                "question": query,
                "answer": answer,
                "mode": st.session_state.explanation_mode,
                "cve_ids": list(risk_data.keys()) if risk_data else [],
                "confidence": confidence.score,
                "confidence_label": confidence.label,
                "risk": {k: v.score for k, v in risk_data.items()},
                "priority": {k: v.priority for k, v in risk_data.items()},
                "pdf_report": pdf_name,
                "docx_report": docx_name,
            }
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})


# =============================================================================
# PAGE: Upload Report
# =============================================================================
def page_upload():
    st.markdown('<p class="console-label">📄 Upload Vulnerability Report</p>', unsafe_allow_html=True)
    st.caption("Supported formats: PDF, DOCX, TXT, CSV, XLSX, JSON.")

    files = st.file_uploader(
        "Drop one or more report files", type=["pdf", "docx", "txt", "csv", "xlsx", "json"], accept_multiple_files=True
    )
    if files and st.button("Ingest into knowledge base"):
        for f in files:
            with st.spinner(f"Processing {f.name}..."):
                try:
                    text = extract_text(f.read(), f.name)
                    chunks = chunk_text(text)
                    n = store.add_report_chunks(chunks, f.name)
                    st.success(f"✅ Added **{n}** chunks from `{f.name}`")
                except Exception as exc:
                    st.error(f"Failed to process `{f.name}`: {exc}")

    st.markdown('<p class="console-label">📚 Indexed Reports</p>', unsafe_allow_html=True)
    sources = store.report_sources()
    if not sources:
        st.info("No reports uploaded yet.")
    else:
        for s in sources:
            st.markdown(f'<span class="src-chip">{s}</span>', unsafe_allow_html=True)

    st.markdown('<p class="console-label">⚡ Quick Actions</p>', unsafe_allow_html=True)
    if sources:
        qa1, qa2, qa3 = st.columns(3)
        if qa1.button("Summarize latest report"):
            st.session_state.messages.append({"role": "user", "content": "Summarize this report."})
            st.session_state.nav = "Investigate (Chat)"
            st.rerun()
        if qa2.button("Extract all CVEs"):
            st.session_state.messages.append({"role": "user", "content": "Extract all CVEs mentioned in this report."})
            st.session_state.nav = "Investigate (Chat)"
            st.rerun()
        if qa3.button("Generate executive summary"):
            st.session_state.messages.append({"role": "user", "content": "Generate an executive summary of this report."})
            st.session_state.nav = "Investigate (Chat)"
            st.rerun()


# =============================================================================
# PAGE: Investigation History
# =============================================================================
def page_history():
    st.markdown('<p class="console-label">🕵 Investigation History</p>', unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.info("No investigations logged yet.")
        return
    for h in history:
        ts = pd.to_datetime(h["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M")
        with st.expander(f"[{ts}] {h['question']}"):
            st.markdown(h["answer"])
            if h.get("cve_ids"):
                for cid in h["cve_ids"]:
                    priority = h.get("priority", {}).get(cid, "N/A")
                    score = h.get("risk", {}).get(cid, "N/A")
                    st.markdown(sev_chip(priority, f"{score}/100" if score != "N/A" else ""), unsafe_allow_html=True)
            st.caption(f"Confidence: {h.get('confidence', 'N/A')}% ({h.get('confidence_label', 'N/A')}) · Mode: {h.get('mode', 'N/A')}")


# =============================================================================
# PAGE: Reports
# =============================================================================
def page_reports():
    st.markdown('<p class="console-label">📑 Generated Reports</p>', unsafe_allow_html=True)
    files = list_report_files()
    if not files:
        st.info("No reports generated yet. Run an investigation and export a report to see it here.")
        return
    for path in files:
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**{path.name}**  \n<span style='color:var(--muted);font-size:0.8rem;'>{time.ctime(path.stat().st_mtime)}</span>", unsafe_allow_html=True)
        mime = "application/pdf" if path.suffix == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        col2.download_button("⬇️ Download", data=path.read_bytes(), file_name=path.name, mime=mime, key=f"dl-{path.name}")


# =============================================================================
# PAGE: Settings
# =============================================================================
def page_settings():
    st.markdown('<p class="console-label">⚙️ Settings</p>', unsafe_allow_html=True)

    st.markdown("**LLM Provider**")
    provider = st.selectbox("Preferred provider", ["auto", "gemini", "groq"], index=["auto", "gemini", "groq"].index(settings.llm_provider))
    gemini_key = st.text_input("Gemini API Key", type="password", value=settings.gemini_api_key or "")
    groq_key = st.text_input("Groq API Key", type="password", value=settings.groq_api_key or "")
    nvd_key = st.text_input("NVD API Key (optional — raises rate limits)", type="password", value=settings.nvd_api_key or "")

    if st.button("Save settings"):
        settings.llm_provider = provider
        settings.gemini_api_key = gemini_key or None
        settings.groq_api_key = groq_key or None
        settings.nvd_api_key = nvd_key or None
        reset_llm_client()
        st.success("Settings updated for this session. For persistence across restarts, set these as environment "
                    "variables or in `.streamlit/secrets.toml`.")

    st.divider()
    st.markdown("**Knowledge Base**")
    if st.button("🗑️ Clear uploaded reports"):
        get_store().report_col.delete(where={"source": {"$ne": "__never__"}}) if store.report_col.count() else None
        st.success("Cleared uploaded report chunks (CVE cache preserved).")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "Dashboard": page_dashboard,
    "Investigate (Chat)": page_chat,
    "Upload Report": page_upload,
    "Investigation History": page_history,
    "Reports": page_reports,
    "Settings": page_settings,
}
PAGES[st.session_state.nav]()

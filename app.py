import os
import sys
import time
import json
from pathlib import Path

import streamlit as st

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from clinical_graph import ClinicalRAGEngine, scrub_phi
from docling_pipeline import DoclingClinicalPipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="ClinixSafe RAG — Clinical Enterprise Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .header-sub {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .badge-doctor {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-nurse {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-billing {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pharmacist {
        background-color: #f3e8ff;
        color: #6b21a8;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .conflict-box {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .refusal-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .phi-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Engine with Caching ---
@st.cache_resource(show_spinner="Connecting to Clinical RAG Engine & Embedding Index...")
def load_engine():
    chroma_dir = str(project_root / "chroma_clinical_db")
    return ClinicalRAGEngine(
        chroma_persist_dir=chroma_dir,
        device="cpu"
    )

engine = load_engine()

# --- Sidebar: Role & Configuration ---
with st.sidebar:
    st.markdown("### 🏥 Clinical Governance Controls")
    
    # 1. Role Selector (RBAC)
    role_options = {
        "Doctor / Attending Physician": "doctor",
        "Registered Nurse (RN) / Triage": "nurse",
        "Clinical Pharmacist": "pharmacist",
        "Medical Billing / Administration": "billing"
    }
    
    selected_role_label = st.selectbox(
        "Authenticated User Role:",
        list(role_options.keys()),
        index=0,
        help="Access boundaries are enforced directly in ChromaDB's retrieval path."
    )
    user_role = role_options[selected_role_label]
    
    # Display role badge
    badge_class = f"badge-{user_role}" if user_role in ["doctor", "nurse", "billing", "pharmacist"] else "badge-doctor"
    st.markdown(f'<span class="{badge_class}">Active Clearance: {user_role.upper()}</span>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 2. Golden Evaluation Scenarios
    st.markdown("### 🧪 Enterprise Golden QA Scenarios")
    
    presets = [
        ("Conflict: Fall Risk SOP (v2 vs v3)", "For Fall risk screening and mitigation on admission, is the correct timing \"Patients scoring >45 receive a bed alarm and non-slip footwear.\" or \"Patients scoring >35 receive a bed alarm and non-slip footwear.\"?", "nurse"),
        ("Conflict: Controlled Substance Waste", "For Witnessed wastage of Schedule II medications, is the correct timing \"Two licensed staff must witness and co-sign wastage within the same shift.\" or \"Two licensed staff must witness and co-sign wastage within the same 12-hour block.\"?", "pharmacist"),
        ("Refusal: Remote Work Policy", "What is our corporate policy on employee remote-work reimbursements?", "doctor"),
        ("RBAC Rejection: Nurse on Restricted Oncology", "What are the signoff conditions described in \"Restricted Oncological Off-Label Drug Protocol\"?", "nurse"),
        ("RBAC Authorized: Doctor on Restricted Oncology", "What are the signoff conditions described in \"Restricted Oncological Off-Label Drug Protocol\"?", "doctor"),
        ("HIPAA Guard: De-Identification Test", "Patient Arthur Dent (DOB 05/10/1965, MRN #77312, SSN 123-45-6789) needs IV antibiotic administration. What is the dilution volume?", "nurse")
    ]
    
    for label, query_text, default_role in presets:
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = query_text
            st.rerun()

    st.markdown("---")
    # 3. System Status
    st.markdown("### ⚙️ System Status")
    st.write("• **Corpus Chunks:** `983` (from `_manifest.jsonl`)")
    st.write("• **LLM Model:** `medgemma-4b-it` (Local LM Studio)")
    st.write("• **Embeddings:** `all-MiniLM-L6-v2`")
    st.write("• **Reranker:** `BAAI/bge-reranker-base`")
    st.write("• **Safety Triage:** Dynamic LLM Critic Node")
    
    if st.button("🔄 Re-Ingest from _manifest.jsonl"):
        with st.spinner("Ingesting all files declared in manifest..."):
            pipeline = DoclingClinicalPipeline(run_vlm_on_images=False)
            manifest_path = project_root / "data" / "_manifest.jsonl"
            docs = pipeline.ingest_manifest(manifest_path, project_root)
            engine.index_documents(docs)
            st.success(f"Successfully indexed {len(docs)} documents!")


# --- Main Area ---
st.markdown('<div class="header-title">🏥 ClinixSafe RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Enterprise Healthcare Decision Support: Zero-Hallucination Retrieval, Database-Level RBAC, Dynamic Conflict Surfacing & Calibrated Refusal</div>', unsafe_allow_html=True)

# Query Input Form
default_query = st.session_state.get("query_input", "For Fall risk screening and mitigation on admission, is the correct timing \"Patients scoring >45 receive a bed alarm and non-slip footwear.\" or \"Patients scoring >35 receive a bed alarm and non-slip footwear.\"?")

with st.form("query_form"):
    user_query = st.text_area("Enter Clinical Inquiry:", value=default_query, height=100)
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.form_submit_button("🔍 Execute Clinical Query", use_container_width=True)
    with col2:
        st.caption("All queries pass through generalized PHI scrubbing, RBAC vector filtering, and the LLM Clinical Critic.")

# --- Results Processing ---
if submit and user_query:
    start_time = time.time()
    
    with st.spinner(f"Evaluating query under role '{user_role}' via LangGraph StateGraph..."):
        result = engine.run_query(user_query, user_role=user_role)
    
    latency = time.time() - start_time
    
    deidentified_q = result.get("deidentified_query", user_query)
    redacted_phi = result.get("redacted_phi", [])
    has_conflict = result.get("has_conflict", False)
    conflict_summary = result.get("conflict_summary", "")
    refusal_flag = result.get("refusal_flag", False)
    refusal_reason = result.get("refusal_reason", "")
    final_answer = result.get("final_answer", "")
    reranked_docs = result.get("reranked_docs", [])
    
    st.markdown("---")
    
    # 1. PHI De-identification Notice (if triggered)
    if redacted_phi:
        st.markdown(f"""
        <div class="phi-box">
            🛡️ <b>HIPAA De-Identification Active:</b> Dynamically redacted synthetic patient identifiers: 
            <code>{', '.join(redacted_phi)}</code>. No PHI entered the vector database or LLM prompt.
        </div>
        """, unsafe_allow_html=True)
        
    # 2. Conflict Warning Banner (if triggered)
    if has_conflict:
        st.markdown(f"""
        <div class="conflict-box">
            <h4>⚠️ Inter-Guideline / Version Conflict Detected</h4>
            <p>{conflict_summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Refusal Banner (if triggered)
    if refusal_flag:
        st.markdown(f"""
        <div class="refusal-box">
            <h4>🛑 Action Required: Clinical Query Refused</h4>
            <p><b>Reason:</b> {refusal_reason}</p>
            <p><i>The system fails loudly rather than producing an ungrounded or unauthorized guess.</i></p>
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Clinical Answer
    st.markdown("### 📋 Clinical Finding & Recommendation")
    st.markdown(final_answer)
    
    # 5. One-Click Citations & Source Inspector
    st.markdown("### 📑 Traceability & Verified Passages (One-Click Audit)")
    
    if reranked_docs:
        tabs = st.tabs([f"[{i+1}] {item['source'][:28]}... (Score: {item['score']})" for i, item in enumerate(reranked_docs[:4])])
        for i, (tab, item) in enumerate(zip(tabs, reranked_docs[:4])):
            with tab:
                st.markdown(f"**Document:** `{item['source']}` | **Status:** `{item['doc'].metadata.get('status', 'ACTIVE')}` | **Version:** `{item['doc'].metadata.get('version', '1')}` | **Relevance Score:** `{item['score']}`")
                st.code(item['doc'].page_content, language="markdown")
    else:
        st.info("No authorized documents retrieved from ChromaDB matching this role or query threshold.")

    # 6. Audit Trail & Diagnostics
    with st.expander("🔍 Clinical LangGraph Execution Trace & Metrics"):
        st.write(f"• **Execution Latency:** `{latency:.2f} seconds`")
        st.write(f"• **Authenticated User Role:** `{user_role}`")
        st.write(f"• **Scrubbed Query:** `{deidentified_q}`")
        st.write(f"• **RBAC Retrived Chunks from Chroma:** `{len(result.get('retrieved_docs', []))}`")
        st.write(f"• **Chunks Passing Cross-Encoder Cutoff (>=0.20):** `{len(reranked_docs)}`")
        st.write(f"• **Max Relevance Score:** `{result.get('max_relevance_score', 0.0)}`")
        st.write(f"• **Guideline Conflict Flag:** `{has_conflict}`")
        st.write(f"• **Refusal State:** `{refusal_flag}`")

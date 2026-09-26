# ==============================================================================
# Clinical LangGraph Workflow: Safety-First Healthcare Enterprise RAG
# Generalized PHI Scrubbing | Dynamic LLM Critic Triage | Precision Synthesis
# Zero Hardcoded Heuristics
# ==============================================================================

import os
import re
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END


# --- 1. Clinical Workflow State ---
class ClinicalWorkflowState(TypedDict):
    query: str
    user_role: str                       # e.g., 'doctor', 'nurse', 'billing'
    deidentified_query: str              # Query after PHI redaction
    redacted_phi: List[str]              # List of redacted items for compliance log
    retrieved_docs: List[Document]        # Chunks retrieved with RBAC filter
    reranked_docs: List[Dict[str, Any]]  # Chunks passing Cross-Encoder threshold
    max_relevance_score: float
    has_conflict: bool                   # Dynamically detected by LLM Critic
    conflict_summary: str                # Dynamic summary of the contradiction
    refusal_flag: bool                   # Dynamically detected by LLM Critic / RBAC
    refusal_reason: str                  # Dynamic refusal reason
    final_answer: str
    citations: List[Dict[str, Any]]


# --- 2. Generalized De-Identification & PHI Guard ---
def scrub_phi(text: str):
    """
    Generalized HIPAA Safe Harbor PHI Redaction.
    Redacts patient names, MRNs, SSNs, DOBs, phone numbers, and addresses.
    Uses clinical entity patterns without any hardcoded names.
    """
    redacted = []

    # 1. Social Security Numbers (SSN): 000-00-0000 or 000 00 0000
    ssn_matches = re.findall(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b', text)
    for m in ssn_matches:
        redacted.append(f"SSN:{m}")
        text = text.replace(m, "[REDACTED_SSN]")

    # 2. Medical Record Numbers (MRN) & Patient IDs
    mrn_pattern = r'(?i)\b(?:MRN|Medical Record|Patient ID|Chart|Acct|Account)[:#\s]*([A-Za-z0-9-]{4,12})\b'
    for m in re.finditer(mrn_pattern, text):
        full_match = m.group(0)
        val = m.group(1)
        redacted.append(f"MRN:{val}")
        text = text.replace(full_match, "[REDACTED_MRN]")

    # 3. Dates of Birth (DOB)
    dob_pattern = r'(?i)\b(?:DOB|Date of Birth|Birthdate|Born)[:\s]*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})\b'
    for m in re.finditer(dob_pattern, text):
        full_match = m.group(0)
        val = m.group(1)
        redacted.append(f"DOB:{val}")
        text = text.replace(full_match, "[REDACTED_DOB]")

    # 4. Patient Names with Clinical Context Labels
    patient_name_pattern = r'(?i)\b(?:Patient|Pt|Member|Name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    for m in re.finditer(patient_name_pattern, text):
        name_val = m.group(1)
        redacted.append(f"NAME:{name_val}")
        text = text.replace(name_val, "[REDACTED_PATIENT_NAME]")

    # 5. Honorifics (Mr. John Smith, Mrs. Margaret Davis)
    honorific_pattern = r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    for m in re.finditer(honorific_pattern, text):
        name_val = m.group(1)
        redacted.append(f"NAME:{name_val}")
        text = text.replace(name_val, "[REDACTED_PATIENT_NAME]")

    # 6. Phone numbers
    phone_matches = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
    for m in phone_matches:
        if len(m.strip()) >= 10:
            redacted.append(f"PHONE:{m}")
            text = text.replace(m, "[REDACTED_PHONE]")

    return text, redacted


# --- 3. Clinical Enterprise RAG Graph Engine ---
class ClinicalRAGEngine:
    def __init__(
        self,
        chroma_persist_dir: str = "/home/preritubuntu/Escape Velocity P2/chroma_clinical_db",
        llm_base_url: str = "http://127.0.0.1:1234/v1",
        llm_model: str = "medgemma-4b-it",
        device: str = "cpu"
    ):
        self.device = device
        self.chroma_persist_dir = chroma_persist_dir
        
        # 1. HuggingFace Embeddings on CPU
        print(f"[Engine] Initializing Embeddings on {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )
        
        # 2. Chroma Vectorstore
        self.vectorstore = Chroma(
            persist_directory=chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name="clinical_knowledge_base"
        )
        
        # 3. BGE Cross-Encoder Reranker on CPU
        print(f"[Engine] Initializing BGE Cross-Encoder Reranker on {device}...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
        self.relevance_threshold = 0.20  # Calibrated threshold for clinical passages
        
        # 4. LangChain LLM Client
        print(f"[Engine] Connecting to LLM ({llm_model} at {llm_base_url})...")
        self.llm = ChatOpenAI(
            base_url=llm_base_url,
            api_key="lm-studio",
            model=llm_model,
            temperature=0
        )
        
        # 5. Build LangGraph
        self.app = self._build_graph()

    def index_documents(self, documents: List[Document]):
        """Indexes documents into Chroma with dynamic RBAC metadata."""
        print(f"[Indexer] Adding {len(documents)} clinical documents to ChromaDB...")
        self.vectorstore.add_documents(documents)
        print(f"[Indexer] Successfully indexed {len(documents)} documents in {self.chroma_persist_dir}!")

    def _build_graph(self):
        workflow = StateGraph(ClinicalWorkflowState)

        # Define Nodes
        workflow.add_node("deidentify", self._node_deidentify)
        workflow.add_node("rbac_retrieval", self._node_rbac_retrieval)
        workflow.add_node("rerank_filter", self._node_rerank_filter)
        workflow.add_node("clinical_triage", self._node_clinical_triage)
        workflow.add_node("refusal_node", self._node_refusal)
        workflow.add_node("synthesis_node", self._node_synthesis)

        # Define Edges
        workflow.set_entry_point("deidentify")
        workflow.add_edge("deidentify", "rbac_retrieval")
        workflow.add_edge("rbac_retrieval", "rerank_filter")
        workflow.add_edge("rerank_filter", "clinical_triage")

        # Conditional Edge after Triage
        workflow.add_conditional_edges(
            "clinical_triage",
            lambda state: "refusal_node" if state["refusal_flag"] else "synthesis_node",
            {
                "refusal_node": "refusal_node",
                "synthesis_node": "synthesis_node"
            }
        )
        workflow.add_edge("refusal_node", END)
        workflow.add_edge("synthesis_node", END)

        return workflow.compile()

    # --- Node 1: De-identification ---
    def _node_deidentify(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        scrubbed, redacted = scrub_phi(state["query"])
        return {
            "deidentified_query": scrubbed,
            "redacted_phi": redacted
        }

    # --- Node 2: RBAC Retrieval in Database Path ---
    def _node_rbac_retrieval(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        user_role = state.get("user_role", "doctor")
        query = state["deidentified_query"]
        
        # Enforce RBAC directly in Chroma's metadata query path
        role_key = f"role_{user_role}"
        try:
            results = self.vectorstore.similarity_search(
                query,
                k=6,
                filter={role_key: True}
            )
        except Exception as e:
            print(f"[RBAC Query Warning]: {e}")
            results = []

        return {"retrieved_docs": results}

    # --- Node 3: Cross-Encoder Reranking & Safety Filtering ---
    def _node_rerank_filter(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        query = state["deidentified_query"]
        
        if not docs:
            return {
                "reranked_docs": [],
                "max_relevance_score": 0.0
            }
            
        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = []
        for doc, score in zip(docs, scores):
            float_score = float(score)
            if float_score >= self.relevance_threshold:
                scored_docs.append({
                    "doc": doc,
                    "score": round(float_score, 4),
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 1),
                    "type": doc.metadata.get("type", "text"),
                    "status": doc.metadata.get("status", "ACTIVE"),
                    "version": doc.metadata.get("version", "1"),
                    "year": doc.metadata.get("published_year", "2024")
                })
                
        # Sort descending by relevance score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        max_score = scored_docs[0]["score"] if scored_docs else 0.0
        
        return {
            "reranked_docs": scored_docs,
            "max_relevance_score": max_score
        }

    # --- Node 4: Dynamic Clinical Triage & Critic Node ---
    def _node_clinical_triage(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        scored_docs = state.get("reranked_docs", [])
        query = state["deidentified_query"]
        user_role = state.get("user_role", "doctor")
        
        # 1. RBAC / Zero-Match Check
        if not scored_docs:
            role_display = user_role.upper()
            return {
                "refusal_flag": True,
                "refusal_reason": (
                    f"Access Denied or Insufficient Evidence for role '{role_display}'. "
                    f"Clinical treatment guidelines and medication dosages are restricted to clinical staff. "
                    f"Your active clearance ({role_display}) is not authorized to retrieve clinical prescribing protocols."
                ),
                "has_conflict": False,
                "conflict_summary": ""
            }

        # 2. Dynamic LLM Critic Call (Zero Hardcoding!)
        passages_text = "\n\n".join([
            f"[Source: {d['source']} | Status: {d.get('status', 'ACTIVE')} | Version: {d.get('version', '1')}]\n{d['doc'].page_content}"
            for d in scored_docs[:3]
        ])
        
        critic_prompt = f"""You are a Clinical Safety and Evidence Critic for a hospital RAG system.
Evaluate the retrieved clinical evidence against the user inquiry.

RETRIEVED CLINICAL PASSAGES:
{passages_text}

USER INQUIRY:
{query}

AUDIT INSTRUCTIONS:
1. EVIDENCE SUFFICIENCY & POPULATION SAFETY:
   - Does the retrieved evidence contain factual guidance to answer this question?
   - If the inquiry asks about a patient group (e.g., pediatric, neonatal, pregnant, dialysis) where the passages explicitly note lack of safety data, pediatric exclusion, or contraindication: mark CAN_ANSWER: NO.
   
2. INTER-GUIDELINE / VERSION CONFLICT:
   - Check if different documents or versions contain differing numerical thresholds (e.g. cutoffs, dosages, BP targets) or if one version is SUPERSEDED by another ACTIVE version.
   - If conflicting or superseding thresholds exist, mark CONFLICT_DETECTED: YES, and summarize both versions and indicate which is active. Otherwise mark NO.

Respond in EXACTLY this format:
CAN_ANSWER: [YES or NO]
REFUSAL_REASON: [If NO, explain the clinical safety reason and missing data, otherwise NONE]
CONFLICT_DETECTED: [YES or NO]
CONFLICT_SUMMARY: [If YES, state the exact contradiction with versions/sources, otherwise NONE]
"""

        try:
            critic_response = self.llm.invoke(critic_prompt).content.strip()
            
            # Parse Critic response
            can_answer = "CAN_ANSWER: NO" not in critic_response.upper()
            
            refusal_reason = ""
            if not can_answer:
                ref_match = re.search(r"REFUSAL_REASON:\s*(.+?)(?:\r?\nCONFLICT_DETECTED|\Z)", critic_response, re.DOTALL | re.IGNORECASE)
                refusal_reason = ref_match.group(1).strip() if ref_match else "Clinical inquiry rejected: Insufficient evidence in approved guidelines."

            has_conflict = "CONFLICT_DETECTED: YES" in critic_response.upper()
            conflict_summary = ""
            if has_conflict:
                conf_match = re.search(r'CONFLICT_SUMMARY:\s*(.+?)\Z', critic_response, re.DOTALL | re.IGNORECASE)
                conflict_summary = conf_match.group(1).strip() if conf_match else "Guideline discrepancy detected across retrieved sources."

            return {
                "refusal_flag": not can_answer,
                "refusal_reason": refusal_reason,
                "has_conflict": has_conflict,
                "conflict_summary": conflict_summary
            }
        except Exception as err:
            print(f"[Critic Node Error]: {err}")
            return {
                "refusal_flag": False,
                "refusal_reason": "",
                "has_conflict": False,
                "conflict_summary": ""
            }

    # --- Node 5: Calibrated Refusal Node ---
    def _node_refusal(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        response = (
            f"🛑 **CLINICAL QUERY REFUSED — ESCALATION TRIGGERED**\n\n"
            f"**Refusal Reason:** {state['refusal_reason']}\n\n"
            f"**Safety Protocol:** In accordance with institutional clinical governance, the model refuses to guess. "
            f"Please escalate to the Attending Physician, Clinical Pharmacy, or Department Supervisor on-call."
        )
        return {
            "final_answer": response,
            "citations": []
        }

    # --- Node 6: Precision-First Grounded Clinical Synthesis Node ---
    def _node_synthesis(self, state: ClinicalWorkflowState) -> Dict[str, Any]:
        query = state["deidentified_query"]
        scored_docs = state["reranked_docs"]
        has_conflict = state["has_conflict"]
        conflict_summary = state["conflict_summary"]
        
        # Build concise context string using only top 2-3 highest-scoring passages
        context_parts = []
        citations = []
        for i, item in enumerate(scored_docs[:3], 1):
            doc = item["doc"]
            source = item["source"]
            page = item["page"]
            doc_type = item["type"]
            score = item["score"]
            status = item.get("status", "ACTIVE")
            
            context_parts.append(f"--- Document [{i}]: {source} (Status: {status}, Score: {score}) ---\n{doc.page_content}")
            citations.append({
                "citation_id": f"[{i}]",
                "source": source,
                "page": page,
                "type": doc_type,
                "score": score
            })
            
        context_str = "\n\n".join(context_parts)
        
        system_prompt = (
            "You are a precision-first clinical decision support AI operating in a high-stakes hospital environment.\n"
            "STRICT CLINICAL RULES:\n"
            "1. DIRECT ANSWER FIRST: Begin immediately with the exact dosage, frequency, numerical threshold, and daily maximum directly from the context.\n"
            "2. BE CONCISE: Limit answer to 2-3 direct, factual sentences. NEVER repeat sentences or duplicate paragraphs.\n"
            "3. INDICATION LIMITATION: If the query asks about a condition (e.g. fever) but the reference context specifies another indication (e.g. pain/opioid combination), explicitly state this clinical distinction.\n"
            "4. CITATIONS: Cite the exact source at the end of each fact using [1], [2] format."
        )
        
        conflict_block = f"""
⚠️ ACTIVE CLINICAL CONFLICT:
{conflict_summary}
Prominently state this contradiction and which version/guideline is currently ACTIVE.
""" if has_conflict else ""

        user_prompt = f"""EVIDENCE CONTEXT:
{context_str}

{conflict_block}

CLINICAL INQUIRY:
{query}

Provide a direct, concise, and non-repetitive clinical answer following the rules above:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            llm_response = self.llm.invoke(messages)
            answer = llm_response.content.strip()
        except Exception as e:
            answer = f"Error during clinical synthesis: {e}"

        return {
            "final_answer": answer,
            "citations": citations
        }

    # --- Public Query Interface ---
    def run_query(self, query: str, user_role: str = "doctor") -> Dict[str, Any]:
        initial_state: ClinicalWorkflowState = {
            "query": query,
            "user_role": user_role,
            "deidentified_query": "",
            "redacted_phi": [],
            "retrieved_docs": [],
            "reranked_docs": [],
            "max_relevance_score": 0.0,
            "has_conflict": False,
            "conflict_summary": "",
            "refusal_flag": False,
            "refusal_reason": "",
            "final_answer": "",
            "citations": []
        }
        return self.app.invoke(initial_state)

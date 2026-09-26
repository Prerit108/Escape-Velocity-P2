# 🏥 ClinixSafe RAG: Healthcare Greenfield Enterprise RAG
> **Problem Statement P-02: Retrieval a Clinical Team Can Rely On**  
> *Zero-Hallucination Retrieval, Docling Multimodal Ingestion, Role-Based Access Control (RBAC), and LangGraph Safety Decision Orchestration.*

---

## 🌟 Executive Overview
In healthcare, **a confident wrong answer is more expensive than no answer at all**, and clinical guidelines routinely contradict each other because they were written in different years by different committees.

**ClinixSafe RAG** is an auditable, safety-first enterprise retrieval system built from the ground up to address these failure modes:
1. **Multimodal Docling PDF Ingestion:** Parses heterogeneous medical PDFs into structured text, markdown tables, and cropped figures.
2. **Optional VLM Vision Support:** Analyzes clinical algorithm diagrams using LangChain and Vision-Language Models (`qwen/qwen3-vl-8b` / LM Studio).
3. **Database-Level RBAC Enforcement:** Enforces access boundaries (Doctor vs. Nurse vs. Billing) **directly inside ChromaDB's retrieval query path**, not merely at the UI layer.
4. **Active Conflict & Contradiction Surfacing:** Spots temporal and guideline discrepancies (e.g. 2021 AHA vs 2024 ESC guideline updates) and alerts clinicians instead of synthesizing a false consensus.
5. **Calibrated Refusal ("Fail Loudly"):** Refuses out-of-scope or ungrounded queries (e.g. unstudied pediatric dosages) and details the missing clinical evidence and escalation protocol.
6. **Zero PHI Leakage:** Automatically scrubs patient identifiers (Names, MRNs, SSNs, DOBs) at the query boundary.

---

## 🏗️ System Architecture

```
                                  [ Clinical User Query + User Role ]
                                                  │
                                                  ▼
                                      [ 1. PHI De-Identification ]
                                     (Scrubs Name, MRN, SSN, DOB)
                                                  │
                                                  ▼
                                    [ 2. RBAC Hybrid Retrieval ]
                        (ChromaDB Filter: filter={'role_<user_role>': True})
                                                  │
                                                  ▼
                                     [ 3. BGE Cross-Encoder ]
                                (Threshold: >= 0.20 score filtering)
                                                  │
                                                  ▼
                                      [ 4. Clinical Triage ]
                                                  │
                     ┌────────────────────────────┴────────────────────────────┐
                     │                                                         │
           (Refusal Flag == True)                                     (Evidence Adequate)
                     │                                                         │
                     ▼                                                         ▼
        [ 5. Calibrated Refusal Node ]                          [ 6. Clinical Synthesis Node ]
        • Fail loudly with specific reason                      • Grounded clinical recommendation
        • List missing parameters (e.g. pediatric)              • Active conflict & supersession warning
        • Escalate to attending specialist                      • 1-Click exact table/page citations
```

---

## 📂 Repository Structure

```
/home/preritubuntu/Escape Velocity P2/
├── data/
│   ├── pdfs/
│   │   ├── 01_Clinical_Guideline_AHA_HeartFailure_2021.pdf
│   │   ├── 02_Clinical_Guideline_ESC_HeartFailure_Update_2024.pdf
│   │   ├── 03_Hospital_Formulary_and_Renal_Dosing_Table_2024.pdf
│   │   ├── 04_Hospital_SOP_ED_Sepsis_Resuscitation_Bundle_v4.pdf
│   │   └── 05_Payer_Policy_CMS_Cardiology_Prior_Authorization_2024.pdf
│   ├── images/
│   │   └── ed_sepsis_flowchart.png
│   └── 06_Evaluation_Benchmark_Questions.json
├── docling_pipeline.py          # Multimodal Docling PDF parser + VLM client
├── clinical_graph.py            # LangGraph workflow, Chroma RBAC, & Triage Engine
├── notebook1.ipynb              # Interactive demonstration notebook
├── chroma_clinical_db/          # Persisted ChromaDB vector database
└── Readme.md                    # System documentation
```

---

## 🧪 7-Case Evaluation Benchmark

| Test ID | Test Scenario | Role | Target Capability | Result |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Sacubitril/Valsartan Renal Dosing | Doctor | Grounded Table Retrieval (Cell-level precision) | ✅ **PASSED** |
| **TC-02** | Empagliflozin eGFR Threshold | Doctor | Inter-Guideline Conflict Surfacing (2021 vs 2024) | ✅ **PASSED** |
| **TC-03** | Blood Pressure Target in Elderly | Doctor | Geriatric Target Conflict Detection | ✅ **PASSED** |
| **TC-04** | Pediatric Empagliflozin Prescription | Doctor | Calibrated Refusal & Fail Loudly Protocol | ✅ **PASSED** |
| **TC-05** | Payer Policy Expedited Appeals | Nurse | RBAC Access Restriction (Returns 0 docs) | ✅ **PASSED** |
| **TC-06** | Payer Policy Expedited Appeals | Billing | RBAC Access Authorization (Returns 72 hours) | ✅ **PASSED** |
| **TC-07** | Septic Shock Fluid Resuscitation | Nurse | PHI De-identification & Zero Identifier Leakage | ✅ **PASSED** |

---

## 🚀 How to Run the Demo

1. **Activate the Environment:**
   ```bash
   conda activate langchain_env
   ```
2. **Start LM Studio:**
   - Launch LM Studio with `medgemma-4b-it` (or any medical model) loaded on port `1234`.
3. **Open the Notebook:**
   - Open and execute [`notebook1.ipynb`](notebook1.ipynb) in VS Code or JupyterLab.

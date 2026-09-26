# ==============================================================================
# Enterprise Multimodal Clinical Ingestion Engine
# Ingests Heterogeneous Corpus: PDFs (Docling), SOPs (.md), Formularies (.json)
# Strict Role-Based Access Segregation (Zero Clinical Leakage into Billing)
# ==============================================================================

import os
import re
import io
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions


def normalize_roles(raw_roles: List[str], doc_type: str = "", path_str: str = "") -> Dict[str, bool]:
    """
    Strictly normalizes role permissions to enforce clinical vs. billing boundaries.
    Clinical treatment guidelines and dosing are strictly restricted from billing/finance.
    Billing records are restricted from bedside nursing.
    """
    flags = {
        "role_doctor": False,
        "role_nurse": False,
        "role_pharmacist": False,
        "role_admin": False,
        "role_billing": False,
    }
    
    path_lower = path_str.lower()
    type_lower = doc_type.lower()
    
    # 1. If explicit roles are declared in manifest
    if raw_roles:
        lower_roles = [r.lower().strip() for r in raw_roles]
        
        # Doctor / Physician
        doctor_aliases = ["doctor", "attending_physician", "physician", "oncologist", "psychiatrist", "medical_director"]
        if any(alias in r for r in lower_roles for alias in doctor_aliases):
            flags["role_doctor"] = True
            
        # Nurse
        nurse_aliases = ["nurse", "charge_nurse", "triage", "staff_nurse", "rn"]
        if any(alias in r for r in lower_roles for alias in nurse_aliases):
            flags["role_nurse"] = True
            
        # Pharmacist
        pharm_aliases = ["pharmacist", "clinical_pharmacist", "pharmacy"]
        if any(alias in r for r in lower_roles for alias in pharm_aliases):
            flags["role_pharmacist"] = True
            
        # Billing & Admin (Strictly only if explicitly authorized)
        admin_aliases = ["billing", "compliance", "finance", "payer_admin"]
        if any(alias in r for r in lower_roles for alias in admin_aliases):
            flags["role_billing"] = True
            flags["role_admin"] = True
            
        # If any flag is set, return it
        if any(flags.values()):
            return flags

    # 2. Strict Domain Fallback based on Document Type & Folder Structure
    # Clinical Treatment Guidelines & Drug Labels -> Doctors, Nurses, Pharmacists (NEVER BILLING)
    if type_lower in ["guideline_pdf", "guideline", "drug_label", "formulary"] or "guidelines" in path_lower or "formulary" in path_lower:
        flags["role_doctor"] = True
        flags["role_nurse"] = True
        flags["role_pharmacist"] = True
        flags["role_billing"] = False
        flags["role_admin"] = False
        return flags

    # Hospital Nursing SOPs -> Nurses, Doctors, Pharmacists (NEVER BILLING)
    if type_lower in ["sop"] or "policies_sops" in path_lower:
        flags["role_nurse"] = True
        flags["role_doctor"] = True
        flags["role_pharmacist"] = True
        flags["role_billing"] = False
        flags["role_admin"] = False
        return flags

    # Payer Policies & Insurance Reimbursement -> Billing, Compliance, Doctors (NEVER NURSES)
    if type_lower in ["payer_policy", "billing"] or "payer" in path_lower or "billing" in path_lower:
        flags["role_billing"] = True
        flags["role_admin"] = True
        flags["role_doctor"] = True
        flags["role_nurse"] = False
        flags["role_pharmacist"] = False
        return flags

    # Default general clinical
    flags["role_doctor"] = True
    flags["role_pharmacist"] = True
    return flags


class LangChainVLMClient:
    """VLM Client using LangChain ChatOpenAI for describing diagrams and figures."""
    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", model: str = "medgemma-4b-it"):
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="lm-studio",
            model=model,
            temperature=0
        )

    def describe_image(self, pil_image: Image.Image, prompt: str = None) -> str:
        if prompt is None:
            prompt = (
                "You are an expert clinical vision assistant. "
                "Describe this medical diagram/flowchart in detail: "
                "1. Visible clinical text, decisions, step numbers, and conditions. "
                "2. Clinical thresholds (e.g., blood pressure, lactate, fluid volumes). "
                "3. Sequence of care interventions. Be factual and concise."
            )
        try:
            img = pil_image.convert("RGB")
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            )
            res = self.llm.invoke([msg])
            return res.content.strip()
        except Exception as e:
            return f"[VLM Fallback: Image description unavailable ({e})]"


class DoclingClinicalPipeline:
    """Unified Ingestion Pipeline for PDFs (Docling), Markdown SOPs, and JSON records."""
    def __init__(self, run_vlm_on_images: bool = False, vlm_client: Optional[LangChainVLMClient] = None):
        self.run_vlm_on_images = run_vlm_on_images
        self.vlm_client = vlm_client or (LangChainVLMClient() if run_vlm_on_images else None)
        
        # Configure Docling for PDFs
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.generate_picture_images = run_vlm_on_images
        pipeline_options.generate_table_images = False
        pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
        
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

    def parse_pdf(self, pdf_path: Path, manifest_meta: Dict[str, Any]) -> List[Document]:
        """Parses a PDF using Docling, extracting layout, tables, and figures."""
        print(f"[Docling] Parsing PDF: {pdf_path.name}...")
        try:
            result = self.converter.convert(str(pdf_path))
            doc = result.document
        except Exception as e:
            print(f"[Docling Error] Failed on {pdf_path.name}: {e}")
            return []

        filename = pdf_path.name
        doc_id = pdf_path.stem
        raw_roles = manifest_meta.get("roles_allowed", [])
        doc_type = manifest_meta.get("doc_type", "guideline_pdf")
        role_flags = normalize_roles(raw_roles, doc_type=doc_type, path_str=str(pdf_path))
        status = manifest_meta.get("status", "ACTIVE")
        version = str(manifest_meta.get("version", "1"))

        extracted_docs: List[Document] = []

        # 1. Structured Tables
        for idx, table in enumerate(doc.tables):
            table_md = table.export_to_markdown()
            page_no = table.prov[0].page_no if table.prov else 1
            caption = table.caption_text(doc=doc) if hasattr(table, "caption_text") else ""
            
            header = f"[CLINICAL TABLE: {filename} | Page {page_no} | Status: {status}]\n"
            if caption:
                header += f"Caption: {caption}\n"
            table_content = header + table_md
            
            meta = {
                "source": filename,
                "doc_id": doc_id,
                "page": page_no,
                "type": "table",
                "status": status,
                "version": version,
                "doc_type": doc_type,
                "chunk_id": f"{doc_id}_tbl_{idx+1}_p{page_no}",
                **role_flags
            }
            extracted_docs.append(Document(page_content=table_content, metadata=meta))

        # 2. Figures (with VLM option)
        for idx, picture in enumerate(doc.pictures):
            page_no = picture.prov[0].page_no if picture.prov else 1
            fig_caption = picture.caption_text(doc=doc) if hasattr(picture, "caption_text") else ""
            fig_desc = ""
            
            if self.run_vlm_on_images and self.vlm_client:
                try:
                    pil_img = picture.get_image(doc=doc)
                    if pil_img:
                        print(f"  [VLM] Describing figure {idx+1} on page {page_no}...")
                        fig_desc = self.vlm_client.describe_image(pil_img)
                except Exception as err:
                    fig_desc = f"[VLM Fallback: Image on page {page_no}]"
                    
            fig_content = f"[CLINICAL FIGURE: {filename} | Page {page_no}]\n"
            if fig_caption:
                fig_content += f"Caption: {fig_caption}\n"
            if fig_desc:
                fig_content += f"Visual Description: {fig_desc}\n"
            else:
                fig_content += f"Clinical flowchart/diagram extracted from {filename} (Page {page_no}).\n"
                
            meta = {
                "source": filename,
                "doc_id": doc_id,
                "page": page_no,
                "type": "figure",
                "status": status,
                "version": version,
                "doc_type": doc_type,
                "chunk_id": f"{doc_id}_fig_{idx+1}_p{page_no}",
                **role_flags
            }
            extracted_docs.append(Document(page_content=fig_content, metadata=meta))

        # 3. Prose Text Blocks
        full_md = doc.export_to_markdown()
        text_chunks = self.text_splitter.split_text(full_md)
        for idx, chunk in enumerate(text_chunks):
            header = f"[Source: {filename} | Status: {status} | Version: {version}]\n"
            meta = {
                "source": filename,
                "doc_id": doc_id,
                "page": 1,
                "type": "text",
                "status": status,
                "version": version,
                "doc_type": doc_type,
                "chunk_id": f"{doc_id}_txt_{idx+1}",
                **role_flags
            }
            extracted_docs.append(Document(page_content=header + chunk, metadata=meta))

        print(f"  Extracted {len(extracted_docs)} chunks from {filename}")
        return extracted_docs

    def parse_markdown_sop(self, md_path: Path, manifest_meta: Dict[str, Any]) -> List[Document]:
        """Parses a Markdown SOP with embedded version and status tags."""
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        filename = md_path.name
        doc_id = md_path.stem
        raw_roles = manifest_meta.get("roles_allowed", [])
        doc_type = manifest_meta.get("doc_type", "sop")
        role_flags = normalize_roles(raw_roles, doc_type=doc_type, path_str=str(md_path))
        status = manifest_meta.get("status", "ACTIVE")
        version = str(manifest_meta.get("version", "1"))
        sop_family = manifest_meta.get("sop_family", doc_id)

        header = f"[HOSPITAL SOP: {sop_family} | Version: {version} | Status: {status}]\n"
        meta = {
            "source": filename,
            "doc_id": doc_id,
            "sop_family": sop_family,
            "version": version,
            "status": status,
            "page": 1,
            "type": "sop",
            "chunk_id": f"{doc_id}_sop_1",
            **role_flags
        }
        return [Document(page_content=header + content, metadata=meta)]

    def parse_json_record(self, json_path: Path, manifest_meta: Dict[str, Any]) -> List[Document]:
        """Parses structured clinical JSON (Restricted Policy or Formulary)."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        filename = json_path.name
        doc_id = json_path.stem
        raw_roles = manifest_meta.get("roles_allowed", data.get("roles_allowed", []))
        doc_type = manifest_meta.get("doc_type", "structured_json")
        role_flags = normalize_roles(raw_roles, doc_type=doc_type, path_str=str(json_path))
        status = manifest_meta.get("status", "ACTIVE")
        version = str(manifest_meta.get("version", "1"))

        text_lines = [f"[STRUCTURED RECORD: {filename} | Type: {doc_type} | Status: {status}]"]
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    text_lines.append(f"{k.upper()}: {v}")
                elif isinstance(v, list) and v and isinstance(v[0], str):
                    text_lines.append(f"{k.upper()}: {', '.join(v)}")
                elif isinstance(v, dict):
                    text_lines.append(f"{k.upper()}: {json.dumps(v)}")
        else:
            text_lines.append(json.dumps(data))

        content = "\n".join(text_lines)
        meta = {
            "source": filename,
            "doc_id": doc_id,
            "type": doc_type,
            "status": status,
            "version": version,
            "page": 1,
            "chunk_id": f"{doc_id}_json_1",
            **role_flags
        }
        return [Document(page_content=content, metadata=meta)]

    def ingest_manifest(self, manifest_file: Path, base_dir: Path) -> List[Document]:
        """Ingests all files declared in _manifest.jsonl across the entire repository."""
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        all_docs = []
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    rel_path = record.get("path", "")
                    target_file = base_dir / rel_path
                    if not target_file.exists():
                        target_file = Path(rel_path)
                    if not target_file.exists():
                        continue

                    suffix = target_file.suffix.lower()
                    if suffix == ".pdf":
                        docs = self.parse_pdf(target_file, record)
                    elif suffix in [".md", ".txt"]:
                        docs = self.parse_markdown_sop(target_file, record)
                    elif suffix == ".json":
                        docs = self.parse_json_record(target_file, record)
                    else:
                        continue

                    all_docs.extend(docs)
                except Exception as err:
                    print(f"[Manifest Ingestion Line {line_no} Warning]: {err}")

        print(f"✅ Ingested {len(all_docs)} total chunks with strict clinical/billing segregation.")
        return all_docs

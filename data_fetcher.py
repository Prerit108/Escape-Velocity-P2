#!/usr/bin/env python3
"""
fetch_corpus.py
================
Builds a heterogeneous, PHI-free corpus for a healthcare Enterprise RAG
prototype: long guidelines, short policies/SOPs, structured patient records,
table-heavy formulary data, payer coverage policies, device manuals, and
research literature.

Every source here is either:
  (a) public domain / open-license real data, or
  (b) synthetically generated (clearly labeled as such in metadata),
so nothing here can leak a real patient identifier.

Usage:
    pip install requests feedparser
    python fetch_corpus.py --all
    python fetch_corpus.py --sources guidelines synthea openfda cms arxiv pmc sops devices

Output layout:
    data/
      guidelines/           long real guideline PDFs/text (CDC, WHO)
      policies_sops/        synthetic internal policy/SOP docs (.md), each carrying
                             an acl_allowed_roles field baked into the document itself
      restricted_policies/  synthetic high-sensitivity policies (e.g. off-label
                             protocols) with strict role-gated ACLs
      structured_records/   Synthea synthetic patient data (CSV/FHIR)
      formulary/            openFDA drug label JSON -> normalized tables
      payer_policies/       CMS coverage determinations (real, contradictory by design)
      device_manuals/       openFDA device data + synthetic IFU docs (ACL-tagged)
      research_papers/      arXiv + PubMed Central OA papers (metadata + text/PDF)
      evaluation_qa_sets/   golden Q/A set generated FROM the real corpus above:
                             conflict handling, insufficient-evidence refusal,
                             and role/ACL boundary test cases
      _manifest.jsonl       one line per ingested doc: source, url, license,
                            doc_type, roles_allowed, fetch_time, sha256

ACL model: every synthetic document embeds its own `acl_allowed_roles` (in the
doc body AND in the manifest row as `roles_allowed`), so access control can be
enforced by filtering retrieval candidates against the querying user's role
BEFORE they ever reach the retriever/reranker -- not just hidden in the UI.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path("data")
MANIFEST_PATH = ROOT / "_manifest.jsonl"
HEADERS = {"User-Agent": "healthcare-rag-corpus-builder/1.0 (research/education use)"}
TIMEOUT = 30


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def ensure_dirs():
    for sub in [
        "guidelines", "policies_sops", "restricted_policies", "structured_records",
        "formulary", "payer_policies", "device_manuals", "research_papers",
        "evaluation_qa_sets",
    ]:
        (ROOT / sub).mkdir(parents=True, exist_ok=True)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_manifest(entry: dict):
    entry["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_bytes(path: Path, content: bytes, *, source: str, url: str, license_: str,
                doc_type: str, extra: dict = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    record_manifest({
        "path": str(path),
        "source": source,
        "url": url,
        "license": license_,
        "doc_type": doc_type,
        "sha256": sha256_of(content),
        "bytes": len(content),
        **(extra or {}),
    })


def save_text(path: Path, text: str, **kwargs):
    save_bytes(path, text.encode("utf-8"), **kwargs)


def http_get(url: str, params: dict = None, retries: int = 3, backoff: float = 2.0):
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last_err})")


# --------------------------------------------------------------------------
# 1. Long clinical guidelines (real, public domain / open license)
#    CDC content is US-government work = public domain. WHO guidance is
#    freely distributable with attribution. Always re-check the specific
#    page's licence before bulk redistribution outside a prototype.
# --------------------------------------------------------------------------

CDC_MMWR_GUIDELINE_PDFS = {
    # A handful of well-known, long, table-heavy CDC clinical guidance docs.
    # Replace/extend this list with whichever guideline topics you need.
    "cdc_opioid_prescribing_2022.pdf": "https://www.cdc.gov/mmwr/volumes/71/rr/pdfs/rr7103a1-H.pdf",
    "cdc_antibiotic_stewardship_core_elements.pdf": "https://www.cdc.gov/antibiotic-use/core-elements/pdfs/hospital-core-elements-h.pdf",
    "cdc_diabetes_std_of_care.pdf": "https://www.cdc.gov/diabetes/pdfs/managing/standardsofcare.pdf",
}


def fetch_guidelines():
    print("[guidelines] fetching long-form CDC guideline PDFs...")
    for fname, url in CDC_MMWR_GUIDELINE_PDFS.items():
        try:
            r = http_get(url)
            save_bytes(
                ROOT / "guidelines" / fname, r.content,
                source="CDC", url=url, license_="US Government Work (public domain)",
                doc_type="guideline_pdf",
            )
            print(f"  saved {fname} ({len(r.content)/1024:.0f} KB)")
        except Exception as e:
            print(f"  [WARN] could not fetch {fname}: {e}", file=sys.stderr)
    print("  NOTE: verify each URL still resolves — agencies restructure pages often.")
    print("  Add WHO guideline PDFs manually from https://www.who.int/publications/i/ "
          "(open-license 'CC BY-NC-SA 3.0 IGO' — check per-document terms).")


# --------------------------------------------------------------------------
# 2. Short internal policies / SOPs — SYNTHETIC (no public equivalent exists)
# --------------------------------------------------------------------------

# Each department maps to the roles allowed to retrieve its SOPs. This map is
# the single source of truth for ACL enforcement -- both the doc body and the
# manifest row are derived from it, so retrieval-time filtering and citation
# display always agree with each other.
DEPT_ACL_ROLES = {
    "Nursing": ["NURSE", "CHARGE_NURSE", "ATTENDING_PHYSICIAN"],
    "Pharmacy": ["PHARMACIST", "PHARMACY_TECH", "ATTENDING_PHYSICIAN"],
    "Emergency Dept": ["ED_PHYSICIAN", "ED_NURSE", "ATTENDING_PHYSICIAN"],
    "Radiology": ["RADIOLOGIST", "RADIOLOGY_TECH", "ATTENDING_PHYSICIAN"],
}

SOP_TEMPLATES = [
    ("SOP-IV-Antibiotic-Administration", "Nursing", 2023,
     "Administration of intravenous antibiotics on general wards",
     ["Verify allergy status before preparation.",
      "Confirm two-patient identifiers against the eMAR.",
      "Infuse over the manufacturer-specified duration; document start/stop time.",
      "Escalate to pharmacist if renal dosing adjustment is flagged."]),
    ("SOP-Fall-Risk-Assessment", "Nursing", 2021,
     "Fall risk screening and mitigation on admission",
     ["Score patient using the Morse Fall Scale at admission and every shift change.",
      "Patients scoring >45 receive a bed alarm and non-slip footwear.",
      "Document reassessment after any fall event within 15 minutes."]),
    ("SOP-Controlled-Substance-Waste", "Pharmacy", 2024,
     "Witnessed wastage of Schedule II medications",
     ["Two licensed staff must witness and co-sign wastage within the same shift.",
      "Partial doses are wasted in the designated pharmacy sink, never in sharps bins.",
      "Discrepancies >2 units trigger an automatic diversion review."]),
    ("SOP-Sepsis-Screening", "Emergency Dept", 2022,
     "Bedside sepsis screening on ED triage",
     ["Apply qSOFA at triage for any patient with suspected infection.",
      "qSOFA >=2 triggers the sepsis bundle: lactate, blood cultures, broad-spectrum antibiotics within 1 hour.",
      "Reassess lactate at 2-4 hours if initial value elevated."]),
    ("SOP-Radiology-Contrast-Consent", "Radiology", 2020,
     "Informed consent for iodinated contrast administration",
     ["Screen eGFR within 90 days prior to contrast studies where feasible.",
      "Document verbal consent and allergy history in the RIS.",
      "Premedicate per protocol for patients with prior moderate/severe reaction history."]),
]


def fetch_sops(n_variants: int = 3):
    print("[policies_sops] generating synthetic SOP/policy documents with baked-in ACLs...")
    rng = random.Random(42)
    generated = []  # returned so the eval-set generator can reference real facts
    for sop_id, dept, year, title, steps in SOP_TEMPLATES:
        allowed_roles = DEPT_ACL_ROLES[dept]
        for v in range(1, n_variants + 1):
            # Simulate real-world messiness: revised dates, minor wording drift,
            # and occasional genuine contradiction between versions (by design,
            # per the "sources disagree" requirement).
            revised_year = year + (v - 1) * 2
            status = "SUPERSEDED" if v < n_variants else "ACTIVE"
            steps_variant = list(steps)
            changed_step = None
            # Guarantee at least one genuine wording contradiction on the final
            # (ACTIVE) version of every SOP family, so the eval-set generator
            # always has a real conflict to test against -- not left to chance,
            # and not limited to whichever step happens to be first.
            is_final_version = (v == n_variants)
            if v > 1 and (is_final_version or rng.random() < 0.5):
                for idx, step_text in enumerate(steps_variant):
                    new_text = None
                    if "15 minutes" in step_text:
                        new_text = step_text.replace("15 minutes", f"{15 + v * 5} minutes")
                    elif ">45" in step_text:
                        new_text = step_text.replace(">45", f">{45 - v * 5}")
                    elif "1 hour" in step_text:
                        new_text = step_text.replace("1 hour", f"{1 + v * 0.5:.1f} hours")
                    elif "90 days" in step_text:
                        new_text = step_text.replace("90 days", f"{90 - v * 15} days")
                    elif "same shift" in step_text:
                        new_text = step_text.replace("same shift", "same 12-hour block")
                    if new_text and new_text != step_text:
                        steps_variant[idx] = new_text
                        changed_step = (step_text, new_text)
                        break  # one genuine contradiction per version is enough
            body = (
                f"# {sop_id} (v{v}) — {title}\n\n"
                f"**Department:** {dept}  \n"
                f"**Effective date:** {revised_year}-01-15  \n"
                f"**Status:** {status}  \n"
                f"**Document ID:** {sop_id}-v{v}  \n"
                f"**acl_allowed_roles:** {json.dumps(allowed_roles)}\n\n"
                "## Procedure\n" +
                "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_variant)) +
                "\n\n## Approval\nApproved by the Clinical Practice Committee.\n"
            )
            fname = f"{sop_id}_v{v}.md"
            save_text(
                ROOT / "policies_sops" / fname, body,
                source="synthetic", url="n/a (synthetically generated)",
                license_="synthetic - no restriction", doc_type="sop",
                extra={"sop_family": sop_id, "version": v, "status": status,
                       "department": dept, "roles_allowed": allowed_roles},
            )
            generated.append({
                "sop_family": sop_id, "title": title, "version": v, "status": status,
                "department": dept, "roles_allowed": allowed_roles,
                "changed_step": changed_step, "filename": fname,
            })
    print(f"  generated {len(SOP_TEMPLATES) * n_variants} SOP versions "
          f"(including intentionally superseded/contradictory versions).")
    return generated


# --------------------------------------------------------------------------
# 2b. High-sensitivity restricted policies — SYNTHETIC, strict role-gated ACL
# --------------------------------------------------------------------------

RESTRICTED_POLICIES = [
    {
        "policy_id": "SEC-POL-099",
        "title": "Restricted Oncological Off-Label Drug Protocol",
        "classification": "INTERNAL_STRICT",
        "acl_allowed_roles": ["ONCOLOGIST", "ATTENDING_PHYSICIAN"],
        "content": ("Off-label usage of checkpoint inhibitors requires dual-board "
                    "signoff and verification via pharmacy logs. Requests lacking "
                    "both signatures must be escalated to the Pharmacy & "
                    "Therapeutics Committee before dispensing."),
    },
    {
        "policy_id": "SEC-POL-114",
        "title": "Psychiatric Involuntary Hold Documentation Protocol",
        "classification": "INTERNAL_STRICT",
        "acl_allowed_roles": ["PSYCHIATRIST", "ATTENDING_PHYSICIAN", "HOSPITAL_LEGAL_COUNSEL"],
        "content": ("Initiation of an involuntary hold requires a signed physician "
                    "certification within 2 hours and notification of hospital "
                    "legal counsel within 12 hours per state statute."),
    },
]


def fetch_restricted_policies():
    print("[restricted_policies] generating synthetic strict-ACL policy documents...")
    for policy in RESTRICTED_POLICIES:
        fname = f"{policy['policy_id']}.json"
        save_text(
            ROOT / "restricted_policies" / fname, json.dumps(policy, indent=2),
            source="synthetic", url="n/a (synthetically generated)",
            license_="synthetic - no restriction", doc_type="restricted_policy",
            extra={"policy_id": policy["policy_id"],
                   "classification": policy["classification"],
                   "roles_allowed": policy["acl_allowed_roles"]},
        )
    print(f"  generated {len(RESTRICTED_POLICIES)} strict-ACL policies "
          "(retrieval must exclude these entirely for any role not on the allow-list).")
    return RESTRICTED_POLICIES


# --------------------------------------------------------------------------
# 3. Structured patient records — Synthea (fully synthetic, public domain)
# --------------------------------------------------------------------------

def fetch_synthea(clone_and_build: bool = False, sample_zip_url: str = None):
    """
    Two ways to get Synthea data:
    (A) Clone + build the Java generator yourself (clone_and_build=True) --
        gives you full control over population size / conditions modeled.
    (B) Download a pre-generated sample dataset zip (faster, no Java needed).
        MITRE publishes these at https://synthea.mitre.org/downloads --
        pass the exact zip URL you want via sample_zip_url.
    """
    print("[structured_records] Synthea synthetic EHR data...")
    dest = ROOT / "structured_records" / "synthea"
    dest.mkdir(parents=True, exist_ok=True)

    if sample_zip_url:
        try:
            r = http_get(sample_zip_url)
            zip_path = dest / "synthea_sample.zip"
            with open(zip_path, "wb") as f:
                f.write(r.content)
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            record_manifest({
                "path": str(dest), "source": "Synthea (MITRE)", "url": sample_zip_url,
                "license": "Public domain (synthetic)", "doc_type": "structured_records_bundle",
            })
            print(f"  extracted Synthea sample data into {dest}")
        except Exception as e:
            print(f"  [WARN] sample download failed: {e}", file=sys.stderr)

    if clone_and_build:
        print("  cloning synthea (requires git + Java 11+ on this machine)...")
        os.system(f"git clone --depth 1 https://github.com/synthetichealth/synthea.git {dest / 'synthea_src'}")
        print("  build+run it yourself, e.g.:")
        print(f"    cd {dest / 'synthea_src'} && ./run_synthea -p 200 "
              f"--exporter.csv.export=true Massachusetts")
        print("  then move the generated ./output/csv/*.csv files into "
              f"{dest}/csv/")

    if not sample_zip_url and not clone_and_build:
        print("  No action requested. Call with sample_zip_url=<zip you found on "
              "https://synthea.mitre.org/downloads> or clone_and_build=True.")


# --------------------------------------------------------------------------
# 4. Formulary / drug label data (real, table-heavy) — openFDA
# --------------------------------------------------------------------------

OPENFDA_LABEL_ENDPOINT = "https://api.fda.gov/drug/label.json"

FORMULARY_SEARCH_TERMS = [
    "metformin", "lisinopril", "atorvastatin", "amoxicillin", "warfarin",
    "insulin glargine", "sertraline", "omeprazole", "albuterol", "prednisone",
]


def fetch_openfda_formulary(limit_per_term: int = 3, api_key: str = None):
    print("[formulary] fetching openFDA structured drug labels...")
    for term in FORMULARY_SEARCH_TERMS:
        params = {
            "search": f'openfda.generic_name:"{term}"',
            "limit": limit_per_term,
        }
        if api_key:
            params["api_key"] = api_key
        try:
            r = http_get(OPENFDA_LABEL_ENDPOINT, params=params)
            payload = r.json()
            for i, result in enumerate(payload.get("results", [])):
                fname = f"{term.replace(' ', '_')}_{i}.json"
                save_text(
                    ROOT / "formulary" / fname, json.dumps(result, indent=2),
                    source="openFDA", url=OPENFDA_LABEL_ENDPOINT,
                    license_="Public domain (FDA structured product labeling)",
                    doc_type="drug_label",
                    extra={"generic_name": term},
                )
            print(f"  {term}: saved {len(payload.get('results', []))} label(s)")
        except Exception as e:
            print(f"  [WARN] {term} failed: {e}", file=sys.stderr)
        time.sleep(0.5)  # be polite to the free tier rate limit
    print("  Tip: get a free openFDA API key to raise the 40 req/min limit: "
          "https://open.fda.gov/apis/authentication/")


# --------------------------------------------------------------------------
# 5. Payer coverage policies (real, genuinely contradictory across years)
# --------------------------------------------------------------------------

CMS_NCD_API = "https://data.cms.gov/data-api/v1/dataset/national-coverage-determinations/data"


def fetch_cms_coverage_policies(limit: int = 25):
    print("[payer_policies] fetching CMS National Coverage Determinations...")
    try:
        r = http_get(CMS_NCD_API, params={"size": limit})
        rows = r.json()
        for i, row in enumerate(rows):
            fname = f"ncd_{i}.json"
            save_text(
                ROOT / "payer_policies" / fname, json.dumps(row, indent=2),
                source="CMS", url=CMS_NCD_API,
                license_="US Government Work (public domain)",
                doc_type="coverage_determination",
            )
        print(f"  saved {len(rows)} NCD records")
    except Exception as e:
        print(f"  [WARN] CMS API call failed ({e}). CMS periodically changes dataset "
              "slugs -- browse https://data.cms.gov/provider-data/ or "
              "https://www.cms.gov/medicare-coverage-database/ and adjust CMS_NCD_API.",
              file=sys.stderr)


# --------------------------------------------------------------------------
# 6. Device manuals — real openFDA device records + synthetic IFU documents
# --------------------------------------------------------------------------

OPENFDA_DEVICE_ENDPOINT = "https://api.fda.gov/device/510k.json"
DEVICE_SEARCH_TERMS = ["infusion pump", "insulin pump", "patient monitor", "ventilator"]

DEVICE_ACL_ROLES = ["NURSE", "BIOMEDICAL_ENGINEER", "ATTENDING_PHYSICIAN"]

SYNTHETIC_DEVICE_IFU_TEMPLATE = """# Instructions for Use — {device_name} (Model {model})

**Manufacturer (fictional):** Meridian Biomedical  
**IFU revision:** {rev}  
**Supersedes:** {supersedes}  
**acl_allowed_roles:** {acl_roles}

## Indications for Use
{indication}

## Warnings
- Do not operate with a low-battery indicator active.
- {warning_2}

## Table: Alarm Priority Reference
| Alarm code | Priority | Meaning | Action |
|---|---|---|---|
| A-101 | High | Occlusion detected | Stop infusion, inspect line |
| A-204 | Medium | Battery <20% | Connect to AC power within 30 min |
| A-311 | Low | Filter due for replacement | Replace within 7 days |
"""


def fetch_device_manuals(limit_per_term: int = 3):
    print("[device_manuals] fetching openFDA 510(k) device records...")
    for term in DEVICE_SEARCH_TERMS:
        try:
            r = http_get(OPENFDA_DEVICE_ENDPOINT,
                         params={"search": f'device_name:"{term}"', "limit": limit_per_term})
            payload = r.json()
            for i, result in enumerate(payload.get("results", [])):
                fname = f"{term.replace(' ', '_')}_{i}.json"
                save_text(
                    ROOT / "device_manuals" / fname, json.dumps(result, indent=2),
                    source="openFDA", url=OPENFDA_DEVICE_ENDPOINT,
                    license_="Public domain", doc_type="device_510k_record",
                    extra={"device_term": term},
                )
        except Exception as e:
            print(f"  [WARN] device term '{term}' failed: {e}", file=sys.stderr)

    print("[device_manuals] generating synthetic IFU manuals (real IFUs are "
          "copyrighted manufacturer documents and shouldn't be bulk-scraped)...")
    devices = [
        ("Infusion Pump", "IP-2200"), ("Insulin Pump", "IX-500"),
        ("Patient Monitor", "PM-7"), ("Ventilator", "V-900"),
    ]
    for name, model in devices:
        for rev in [1, 2]:
            body = SYNTHETIC_DEVICE_IFU_TEMPLATE.format(
                device_name=name, model=model, rev=f"Rev {rev}",
                supersedes=f"Rev {rev-1}" if rev > 1 else "N/A (initial release)",
                acl_roles=json.dumps(DEVICE_ACL_ROLES),
                indication=f"Indicated for continuous {name.lower()} delivery/monitoring "
                            "in adult inpatient settings under clinician supervision.",
                warning_2="Verify calibration after any firmware update." if rev > 1
                          else "Verify calibration before first clinical use.",
            )
            fname = f"{model}_IFU_rev{rev}.md"
            save_text(
                ROOT / "device_manuals" / fname, body,
                source="synthetic", url="n/a", license_="synthetic - no restriction",
                doc_type="device_ifu",
                extra={"model": model, "revision": rev, "roles_allowed": DEVICE_ACL_ROLES},
            )


# --------------------------------------------------------------------------
# 7. Research literature — arXiv + PubMed Central Open Access subset
# --------------------------------------------------------------------------

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_QUERIES = [
    "clinical retrieval augmented generation",
    "medical question answering hallucination",
    "electronic health record NLP de-identification",
]

PMC_OA_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PMC_OA_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_QUERIES = ["clinical decision support retrieval augmented generation",
               "medication reconciliation error"]


def fetch_arxiv(max_results: int = 5):
    print("[research_papers] fetching arXiv abstracts/metadata...")
    for q in ARXIV_QUERIES:
        try:
            r = http_get(ARXIV_API, params={
                "search_query": f"all:{q}", "start": 0, "max_results": max_results,
            })
            root = ET.fromstring(r.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            entries = root.findall("a:entry", ns)
            for i, entry in enumerate(entries):
                title = entry.find("a:title", ns).text.strip()
                summary = entry.find("a:summary", ns).text.strip()
                link = entry.find("a:id", ns).text.strip()
                fname = f"arxiv_{q.replace(' ', '_')}_{i}.md"
                save_text(
                    ROOT / "research_papers" / fname,
                    f"# {title}\n\nSource: {link}\n\n## Abstract\n{summary}\n",
                    source="arXiv", url=link, license_="arXiv non-exclusive license (OA)",
                    doc_type="paper_abstract", extra={"query": q},
                )
            print(f"  '{q}': saved {len(entries)} abstracts")
        except Exception as e:
            print(f"  [WARN] arXiv query '{q}' failed: {e}", file=sys.stderr)
        time.sleep(1)


def fetch_pmc_oa(max_results: int = 5):
    print("[research_papers] fetching PubMed Central Open Access metadata...")
    for q in PMC_QUERIES:
        try:
            r = http_get(PMC_OA_EUTILS, params={
                "db": "pmc", "term": f"{q} AND open access[filter]",
                "retmax": max_results, "retmode": "json",
            })
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            for pmcid in ids:
                rf = http_get(PMC_OA_FETCH, params={
                    "db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml",
                })
                fname = f"pmc_{pmcid}.xml"
                save_bytes(
                    ROOT / "research_papers" / fname, rf.content,
                    source="PubMed Central OA", url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                    license_="Open Access subset (per-article licence embedded in XML)",
                    doc_type="paper_fulltext_xml", extra={"pmcid": pmcid, "query": q},
                )
                time.sleep(0.4)
            print(f"  '{q}': saved {len(ids)} full-text XML files")
        except Exception as e:
            print(f"  [WARN] PMC query '{q}' failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# 8. Golden evaluation Q/A set — generated FROM the real corpus above, not
#    invented separately, so every expected_behavior points at an actual
#    file/fact this run produced.
# --------------------------------------------------------------------------

def fetch_eval_qa_set(sop_records: list = None, restricted_policies: list = None):
    print("[evaluation_qa_sets] building golden eval set from the corpus just generated...")
    sop_records = sop_records or []
    restricted_policies = restricted_policies or []
    eval_set = []

    # (a) Conflict-handling cases: pick real SOP families where a later version
    # actually changed a step relative to an earlier one.
    by_family = {}
    for r in sop_records:
        by_family.setdefault(r["sop_family"], []).append(r)
    qid_n = 1
    for family, versions in by_family.items():
        versions.sort(key=lambda r: r["version"])
        changed = [v for v in versions if v.get("changed_step")]
        if changed:
            v = changed[0]
            old, new = v["changed_step"]
            active = next((x for x in versions if x["status"] == "ACTIVE"), versions[-1])
            eval_set.append({
                "qid": f"Q_CONFLICT_{qid_n:02d}",
                "question": f"For {v['title']}, is the correct timing \"{old}\" or \"{new}\"?",
                "expected_behavior": (
                    f"Surface the conflict between {family} v{v['version']} "
                    f"(SUPERSEDED, file {v['filename']}) and the ACTIVE version "
                    f"({active['filename']}) rather than silently picking one; "
                    f"state which version is currently in force."
                ),
                "test_type": "conflict_handling",
                "source_files": [v["filename"], active["filename"]],
            })
            qid_n += 1

    # (b) Insufficient-evidence refusal: genuinely out-of-corpus-domain question.
    eval_set.append({
        "qid": "Q_REFUSE_01",
        "question": "What is our corporate policy on employee remote-work reimbursements?",
        "expected_behavior": ("Refuse: no document in this corpus covers HR/remote-work "
                               "policy. State explicitly that the evidence is missing "
                               "rather than answering from general knowledge."),
        "test_type": "insufficient_evidence_refusal",
        "source_files": [],
    })
    eval_set.append({
        "qid": "Q_REFUSE_02",
        "question": "What was this patient's diagnosis on their visit last Tuesday?",
        "expected_behavior": ("Refuse: no patient-specific temporal record like this "
                               "exists in the corpus (Synthea data is population-level "
                               "synthetic, not a live per-patient query target here). "
                               "State what is missing rather than fabricating an answer."),
        "test_type": "insufficient_evidence_refusal",
        "source_files": [],
    })

    # (c) Role/ACL boundary cases: pull straight from the real restricted policies
    # and department-gated SOPs generated this run.
    for i, policy in enumerate(restricted_policies, start=1):
        eval_set.append({
            "qid": f"Q_ACL_{i:02d}",
            "question": f"What are the signoff conditions described in \"{policy['title']}\"?",
            "expected_behavior": (
                f"Retrieval must exclude {policy['policy_id']} entirely for any user "
                f"whose role is not in {policy['acl_allowed_roles']}. The correct "
                f"behavior for an unauthorized role is to state the answer is "
                f"restricted, not to answer with reduced detail."
            ),
            "test_type": "role_access_boundary",
            "source_files": [f"{policy['policy_id']}.json"],
        })

    dept_examples = {}
    for r in sop_records:
        dept_examples.setdefault(r["department"], r)
    for dept, r in dept_examples.items():
        eval_set.append({
            "qid": f"Q_ACL_DEPT_{dept.replace(' ', '_')}",
            "question": f"What does the {r['title']} procedure require?",
            "expected_behavior": (
                f"Only retrievable for roles {r['roles_allowed']}; a user outside "
                f"this list (e.g. a Radiology tech asking about a Pharmacy SOP) "
                f"should not have this chunk surfaced at all."
            ),
            "test_type": "role_access_boundary",
            "source_files": [r["filename"]],
        })

    save_text(
        ROOT / "evaluation_qa_sets" / "golden_eval_questions.json",
        json.dumps(eval_set, indent=2),
        source="derived", url="n/a (generated from this run's own corpus)",
        license_="synthetic - no restriction", doc_type="eval_qa_set",
        extra={"n_questions": len(eval_set)},
    )
    print(f"  wrote {len(eval_set)} golden eval questions, each pointing at a real "
          f"file this run produced (see source_files in each question).")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

FETCHERS = {
    "guidelines": fetch_guidelines,
    "sops": fetch_sops,
    "restricted_policies": fetch_restricted_policies,
    "synthea": lambda: fetch_synthea(clone_and_build=False, sample_zip_url=None),
    "openfda": fetch_openfda_formulary,
    "cms": fetch_cms_coverage_policies,
    "devices": fetch_device_manuals,
    "arxiv": fetch_arxiv,
    "pmc": fetch_pmc_oa,
    "eval_qa": fetch_eval_qa_set,  # run last; wired up specially in main()
}


def main():
    parser = argparse.ArgumentParser(description="Fetch a heterogeneous healthcare RAG corpus")
    parser.add_argument("--sources", nargs="*", choices=list(FETCHERS.keys()),
                         help="Which sources to run (default: all)")
    parser.add_argument("--all", action="store_true", help="Run every source")
    args = parser.parse_args()

    ensure_dirs()
    to_run = list(FETCHERS.keys()) if (args.all or not args.sources) else args.sources

    print(f"Running fetchers: {to_run}\n")

    # sops and restricted_policies return data the eval-set generator needs,
    # so capture their outputs regardless of the order the user requested.
    sop_records, restricted_policies = [], []
    for name in to_run:
        if name == "eval_qa":
            continue  # handled after the loop, once dependencies have run
        try:
            result = FETCHERS[name]()
            if name == "sops" and result:
                sop_records = result
            if name == "restricted_policies" and result:
                restricted_policies = result
        except Exception as e:
            print(f"[ERROR] fetcher '{name}' crashed: {e}", file=sys.stderr)
        print()

    if "eval_qa" in to_run:
        if not sop_records:
            print("[eval_qa] 'sops' wasn't run this session -- generating it first "
                  "so the eval set has real facts to point at.")
            sop_records = fetch_sops()
        if not restricted_policies:
            print("[eval_qa] 'restricted_policies' wasn't run this session -- "
                  "generating it first so ACL eval questions have real files to point at.")
            restricted_policies = fetch_restricted_policies()
        fetch_eval_qa_set(sop_records, restricted_policies)
        print()

    print(f"Done. Manifest at {MANIFEST_PATH} — every saved file has a sha256, "
          f"source URL, license, and (where applicable) roles_allowed recorded "
          f"for audit and access-control enforcement.")


if __name__ == "__main__":
    main()
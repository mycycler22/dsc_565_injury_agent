import argparse
import csv
import http.client
import json
import os
import re
import time
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
 
from Bio import Entrez

COMMON_INJURIES = [
    # Head
    "concussion",
    # Shoulder / arm / hand
    "shoulder injury",
    "shoulder dislocation",
    "rotator cuff injury",
    "AC joint injury",
    "SLAP tear",
    "elbow injury",
    "UCL injury",
    "tennis elbow",
    "wrist injury",
    "scaphoid fracture",
    "hand fracture",
    # Trunk / back
    "back injury",
    "spondylolysis",
    "rib fracture",
    "sports hernia",
    # Hip / groin / thigh
    "hamstring strain",
    "quadriceps strain",
    "groin strain",
    "hip flexor strain",
    "hip labral tear",
    "femoroacetabular impingement",
    "thigh contusion",
    # Knee
    "knee sprain",
    "MCL sprain",
    "ACL tear",
    "PCL injury",
    "meniscus tear",
    "patellar tendinopathy",
    "patellar dislocation",
    "patellofemoral pain",
    # Lower leg / ankle / foot
    "ankle sprain",
    "high ankle sprain",
    "ankle fracture",
    "Achilles tendon injury",
    "Achilles tendon rupture",
    "calf strain",
    "shin splints",
    "tibial stress fracture",
    "stress fracture",
    "plantar fasciitis",
    "Jones fracture",
    "turf toe",
    "Lisfranc injury",
    # General
    "muscle strain",
    "tendinopathy",
]

DURATION = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:(?:-|–|to)\s*\d+(?:\.\d+)?\s*)?(?:days?|weeks?|months?|years?)\b",
    re.I,
)

RECOVERY = re.compile(
    r"return(?:ed|ing)? to (?:play|sport|competition|training|activity|participation)|"
    r"recover|resolv|time to|time loss|days? missed|days? lost|\bRTP\b|\bRTS\b|"
    r"asymptomatic|symptom[- ]free|healing|rehabilitat",
    re.I,
)

## Abstract section labels that are worth scanning

KEEP_LABEL = re.compile(r"RESULT|CONCLUSION|FINDING|OUTCOME|UNLABELED", re.I)
# Full-text section titles worth scanning
WANTED_SECTION = re.compile(r"result|outcome|return|recover|time to|follow", re.I)
 
MAX_CHUNK_CHARS = 6000
BATCH = 100


DELAY = 0.34

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def with_retry(fn, tries=4):
    """Call fn(), retrying transient network / NCBI errors with exponential backoff."""
    for attempt in range(tries):
        try:
            result = fn()
            time.sleep(DELAY)
            return result
        except (urllib.error.URLError, http.client.HTTPException, RuntimeError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def configure_entrez():
    global DELAY
    email = os.environ.get("NCBI_EMAIL")
    if not email:
        raise SystemExit("Set NCBI_EMAIL (NCBI requires a contact email), e.g. "
                         "export NCBI_EMAIL=you@example.com")
    Entrez.email = email
    key = os.environ.get("NCBI_API_KEY")
    if key:
        Entrez.api_key = key
        DELAY = 0.11


## Collection of PubMed Data

def search_pmids(injury, retmax):
    term = (
        f'"{injury}"[tiab] AND '
        '(athlete*[tiab] OR "return to play"[tiab] OR "return to sport"[tiab]) '
        "AND hasabstract"
    )
 
    def go():
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort="relevance")
        try:
            return Entrez.read(handle)["IdList"]
        finally:
            handle.close()
 
    return with_retry(go)


def _year(article):
    pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    if "Year" in pub_date:
        return str(pub_date["Year"])
    match = re.search(r"\d{4}", str(pub_date.get("MedlineDate", "")))
    return match.group(0) if match else ""


def fetch_structured(pmids):
    """Fetch PubMed XML and keep abstract section labels (INTRODUCTION, METHODS, ...)."""
 
    def go():
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml")
        try:
            return Entrez.read(handle)
        finally:
            handle.close()
 
    data = with_retry(go)
    papers = []
    for art in data.get("PubmedArticle", []):
        cit = art["MedlineCitation"]
        article = cit["Article"]
        sections = []
        for part in article.get("Abstract", {}).get("AbstractText", []):
            label = getattr(part, "attributes", {}).get("Label", "UNLABELED").upper()
            sections.append({"label": label, "text": str(part)})
        papers.append({
            "pmid": str(cit["PMID"]),
            "title": str(article.get("ArticleTitle", "")),
            "year": _year(article),
            "journal": str(article.get("Journal", {}).get("Title", "")),
            "pub_types": [str(p) for p in article.get("PublicationTypeList", [])],
            "sections": sections,
        })
    return papers

## Filters out papers

def key_sentences(paper):
    """Sentences in Results/Conclusion that contain BOTH a duration and recovery language."""
    hits = []
    for sec in paper["sections"]:
        if not KEEP_LABEL.search(sec["label"]):
            continue
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", sec["text"]):
            if DURATION.search(sent) and RECOVERY.search(sent):
                hits.append({"section": sec["label"], "sentence": sent.strip()})
    return hits

## PMC Full Text

def pmid_to_pmcid(pmid):
    def go():
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc")
        try:
            return Entrez.read(handle)
        finally:
            handle.close()
 
    result = with_retry(go)
    links = result[0].get("LinkSetDb", []) if result else []
    return links[0]["Link"][0]["Id"] if links else None
 
 
def _text(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""
 
 
def fulltext_chunks(pmcid):
    """Pull Results-like sections and tables from a PMC article. Skips Intro/Discussion."""
 
    def go():
        handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
        try:
            return handle.read()
        finally:
            handle.close()
 
    raw = with_retry(go)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
 
    chunks = []
    for sec in root.iter("sec"):
        title = _text(sec.find("title"))
        if not WANTED_SECTION.search(title):
            continue
        body = " ".join(_text(p) for p in sec.findall("p"))
        if body and DURATION.search(body):
            chunks.append({"section": title, "text": body[:MAX_CHUNK_CHARS]})
 
    for tw in root.iter("table-wrap"):
        caption = _text(tw.find("caption"))
        rows = []
        for tr in tw.iter("tr"):
            cells = [_text(c) for c in tr]
            if any(cells):
                rows.append(" | ".join(cells))
        table_text = "\n".join(rows)
        if table_text and (DURATION.search(table_text) or RECOVERY.search(caption)):
            chunks.append({"section": f"TABLE: {caption[:200]}",
                           "text": table_text[:MAX_CHUNK_CHARS]})
    return chunks


## Output

def build_payload(paper):
    types = ", ".join(paper["pub_types"]) or "Unknown"
    lines = [f"Title: {paper['title']} | Type: {types} | PMID: {paper['pmid']} | "
             f"Year: {paper['year']} | Injury query: {paper['injury_query']}"]
    for hit in paper["hits"]:
        lines.append(f"[{hit['section']}] {hit['sentence']}")
    for chunk in paper.get("fulltext_chunks", []):
        lines.append(f"[FULLTEXT: {chunk['section']}] {chunk['text']}")
    return "\n".join(lines) + "\n"
 
 
def collect(injuries, retmax, out_dir, fulltext):
    for injury in injuries:
        slug = slugify(injury)
        paper_dir = out_dir / "papers" / slug
        text_dir = out_dir / "text" / slug
        paper_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
 
        pmids = search_pmids(injury, retmax)
        new = [p for p in pmids if not (paper_dir / f"{p}.json").exists()]
        print(f"[{injury}] {len(pmids)} found, {len(new)} new")
 
        passed = 0
        for i in range(0, len(new), BATCH):
            for paper in fetch_structured(new[i:i + BATCH]):
                paper["injury_query"] = injury
                paper["hits"] = key_sentences(paper)
                paper["pmcid"] = None
                paper["fulltext_chunks"] = []
 
                if paper["hits"]:
                    passed += 1
                    if fulltext:
                        try:
                            paper["pmcid"] = pmid_to_pmcid(paper["pmid"])
                            if paper["pmcid"]:
                                paper["fulltext_chunks"] = fulltext_chunks(paper["pmcid"])
                        except Exception as exc:  # full text is a bonus; never fail the run
                            print(f"  full text failed for PMID {paper['pmid']}: {exc}")
                    (text_dir / f"{paper['pmid']}.txt").write_text(
                        build_payload(paper), encoding="utf-8")
 
                (paper_dir / f"{paper['pmid']}.json").write_text(
                    json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{injury}] {passed} of {len(new)} new papers passed the recovery-time filter")
 
 
def build_index(out_dir):
    rows = []
    for path in sorted((out_dir / "papers").glob("*/*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "injury_query": p["injury_query"],
            "pmid": p["pmid"],
            "year": p["year"],
            "pub_types": "; ".join(p["pub_types"]),
            "title": p["title"],
            "n_hit_sentences": len(p["hits"]),
            "pmcid": p.get("pmcid") or "",
            "n_fulltext_chunks": len(p.get("fulltext_chunks", [])),
            "passed_filter": bool(p["hits"]),
        })
    if not rows:
        print("No papers saved.")
        return
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
 
    print("\nPass rate by injury (low rate = thin literature -> expect 'no_benchmark'):")
    for injury in sorted({r["injury_query"] for r in rows}):
        group = [r for r in rows if r["injury_query"] == injury]
        n_pass = sum(r["passed_filter"] for r in group)
        print(f"  {injury:25s} {n_pass:4d} / {len(group):4d}  ({n_pass / len(group):.0%})")
    print(f"\nIndex written to {out_dir / 'index.csv'}")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injuries", nargs="+", default=COMMON_INJURIES)
    ap.add_argument("--retmax", type=int, default=200, help="max PubMed results per injury")
    ap.add_argument("--out", default="injury_data", help="output directory")
    ap.add_argument("--no-fulltext", action="store_true", help="skip PMC full-text fetch")
    args = ap.parse_args()
 
    configure_entrez()
    out_dir = Path(args.out)
    collect(args.injuries, args.retmax, out_dir, fulltext=not args.no_fulltext)
    build_index(out_dir)
 
 
if __name__ == "__main__":
    main()


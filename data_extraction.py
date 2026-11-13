import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
 
import pandas as pd
 
from data_collection import COMMON_INJURIES


DEFAULT_MODEL = "gemini-3.8-flash"

UNIT_TO_DAYS = {"days": 1.0, "weeks": 7.0, "months": 30.4, "years": 365.25}

OUTCOMES = [
    "symptom_resolution",             
    "return_to_practice",             
    "return_to_full_competition",     
    "return_to_preinjury_performance",
    "return_to_activity_unspecified", 
    "healing_or_union",              
    "other",
]

RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "injury": {"type": "string",
                   "description": "The injury this finding is about. Use the exact name from the "
                                  "injury list when one fits; otherwise a short standard name."},
        "severity_or_grade": {"type": ["string", "null"],
                              "description": "e.g. 'grade 2', 'complete rupture', 'first-time'"},
        "treatment": {"type": "string",
                      "enum": ["surgical", "conservative", "mixed", "unspecified"]},
        "outcome_definition": {"type": "string", "enum": OUTCOMES},
        "statistic_type": {"type": "string", "enum": ["mean", "median", "range_only", "other"]},
        "value": {"type": ["number", "null"],
                  "description": "The reported mean or median, exactly as written. Null if only a range."},
        "sd": {"type": ["number", "null"], "description": "Standard deviation if reported."},
        "low": {"type": ["number", "null"],
                "description": "Lower bound of a reported range, IQR or confidence interval."},
        "high": {"type": ["number", "null"],
                 "description": "Upper bound of a reported range, IQR or confidence interval."},
        "dispersion_type": {"type": "string",
                            "enum": ["SD", "IQR", "range", "95% CI", "none"]},
        "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"],
                 "description": "The unit used in the text. Do NOT convert."},
        "n": {"type": ["integer", "null"], "description": "Number of athletes/participants behind this number."},
        "population": {"type": ["string", "null"],
                       "description": "e.g. 'collegiate football players', 'elite soccer', 'adolescents'"},
        "evidence_quote": {"type": "string",
                           "description": "The exact sentence or table row the numbers came from, copied verbatim."},
    },
    "required": ["injury", "treatment", "outcome_definition", "statistic_type",
                 "dispersion_type", "unit", "evidence_quote"],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"records": {"type": "array", "items": RECORD_SCHEMA}},
    "required": ["records"],
}

SYSTEM = f"""You extract time-to-recovery findings for sports injuries from excerpts of research papers.
 
Rules:
- Extract only findings about how long recovery took: symptom resolution, return to practice, return to sport, time loss. Ignore everything else (incidence, risk factors, correlations, pain scores).
- Report only what the text states. If a field is not stated, use null (or "unspecified" / "none" where the field requires a value). Never estimate or infer numbers.
- Make one record per distinct finding: a different injury, treatment group, severity grade, population, or outcome definition is a separate record.
- Keep the number and unit exactly as written (do not convert weeks to days).
- Be precise about outcome_definition. "Symptom-free" and "cleared to play" are different outcomes.
- evidence_quote must be copied verbatim from the text, one sentence or one table row.
- Some papers contain no usable recovery time. Return an empty list in that case.
- Use the exact name from this list when the injury matches one of them: {", ".join(COMMON_INJURIES)}."""

## Helpers

def normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()
 
 
def extract(client, model, text, config, attempts=5):
    """One Gemini call per paper with schema-constrained JSON output; returns a list of records.
 
    Retries with exponential backoff on rate limits, network errors and malformed JSON.
    """
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(model=model, contents=text, config=config)
            return json.loads(response.text)["records"]
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


## Extraction of data from Text

def run_extraction(out_dir, model, limit, workers):

    from google import genai
    from google.genai import types
 
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY first, e.g. export GEMINI_API_KEY=your-key")
 
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
        response_json_schema=RESPONSE_SCHEMA,
        temperature=0,
    )
    path = out_dir / "extractions.jsonl"
 
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done.add((item["slug"], item["pmid"]))
 
    files = sorted((out_dir / "text").glob("*/*.txt"))
    todo = [f for f in files if (f.parent.name, f.stem) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(files)} text files, {len(done)} already extracted, processing {len(todo)}")
 
    lock = threading.Lock()
 
    def work(file):
        records = extract(client, model, file.read_text(encoding="utf-8"), config)
        return {"slug": file.parent.name, "pmid": file.stem, "records": records}
 
    n_done = n_records = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(path, "a", encoding="utf-8") as out:
        futures = {pool.submit(work, f): f for f in todo}
        for future in as_completed(futures):
            file = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # leave unrecorded so the next run retries it
                print(f"  failed {file.parent.name}/{file.stem}: {exc}")
                continue
            with lock:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
            n_done += 1
            n_records += len(item["records"])
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(todo)} papers, {n_records} findings so far")
    print(f"Extracted {n_records} findings from {n_done} papers")



## Building the Dataset

def build_dataset(out_dir):
    path = out_dir / "extractions.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run without --build-only first.")
 
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        slug, pmid = item["slug"], item["pmid"]
 
        text_file = out_dir / "text" / slug / f"{pmid}.txt"
        paper_file = out_dir / "papers" / slug / f"{pmid}.json"
        source_text = normalize(text_file.read_text(encoding="utf-8")) if text_file.exists() else ""
        meta = json.loads(paper_file.read_text(encoding="utf-8")) if paper_file.exists() else {}
 
        for rec in item["records"]:
            factor = UNIT_TO_DAYS.get(rec.get("unit"))
 
            def days(x):
                return None if x is None or factor is None else round(x * factor, 1)
 
            quote = rec.get("evidence_quote", "")
            rows.append({
                "pmid": pmid,
                "year": meta.get("year", ""),
                "title": meta.get("title", ""),
                "pub_types": "; ".join(meta.get("pub_types", [])),
                "injury_query": meta.get("injury_query", slug),
                "injury": (rec.get("injury") or "").strip(),
                "severity_or_grade": rec.get("severity_or_grade"),
                "treatment": rec.get("treatment"),
                "outcome_definition": rec.get("outcome_definition"),
                "statistic_type": rec.get("statistic_type"),
                "value_days": days(rec.get("value")),
                "sd_days": days(rec.get("sd")),
                "low_days": days(rec.get("low")),
                "high_days": days(rec.get("high")),
                "dispersion_type": rec.get("dispersion_type"),
                "n": rec.get("n"),
                "population": rec.get("population"),
                "evidence_quote": quote,
                # Hallucination check: the quote must appear in the text the model was given
                "quote_verified": bool(quote) and normalize(quote) in source_text,
            })
 
    if not rows:
        print("No findings extracted yet.")
        return
 
    df = pd.DataFrame(rows)
    df["n"] = pd.to_numeric(df["n"], errors="coerce").astype("Int64")
    df["injury_key"] = df["injury"].str.lower()
    df["usable"] = df["quote_verified"] & df["value_days"].notna()
    df.to_csv(out_dir / "dataset.csv", index=False)
 
    print(f"{len(df)} findings from {df['pmid'].nunique()} papers")
    print(f"  quote verified: {df['quote_verified'].mean():.0%}")
    print(f"  usable for benchmarks (verified, has a mean/median value): {int(df['usable'].sum())}")
 
    usable = df[df["usable"] & df["statistic_type"].isin(["mean", "median"])]
    if usable.empty:
        print("No usable findings, so benchmarks.csv was not written.")
        return
 
    bench = (
        usable.groupby(["injury_key", "treatment", "outcome_definition"])
        .agg(n_findings=("value_days", "size"),
             n_papers=("pmid", "nunique"),
             median_days=("value_days", "median"),
             p25_days=("value_days", lambda s: s.quantile(0.25)),
             p75_days=("value_days", lambda s: s.quantile(0.75)),
             min_days=("value_days", "min"),
             max_days=("value_days", "max"),
             total_participants=("n", "sum"))
        .reset_index()
        .rename(columns={"injury_key": "injury"})
        .sort_values(["injury", "treatment", "outcome_definition"])
    )
    bench.to_csv(out_dir / "benchmarks.csv", index=False)
    print(f"Wrote {out_dir / 'dataset.csv'} and {out_dir / 'benchmarks.csv'} "
          f"({len(bench)} benchmark groups)")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="injury_data", help="folder written by data_collection.py")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="only process this many new papers (for testing)")
    ap.add_argument("--workers", type=int, default=4, help="parallel API calls")
    ap.add_argument("--build-only", action="store_true",
                    help="skip the LLM step and just rebuild dataset.csv / benchmarks.csv")
    args = ap.parse_args()
 
    out_dir = Path(args.out)
    if not (out_dir / "text").exists():
        raise SystemExit(f"{out_dir / 'text'} not found. Run data_collection.py first.")
 
    if not args.build_only:
        run_extraction(out_dir, args.model, args.limit, args.workers)
    build_dataset(out_dir)
 
 
if __name__ == "__main__":
    main()


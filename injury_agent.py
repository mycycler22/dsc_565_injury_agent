import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from crewai.tools import BaseTool, tool
from typing import Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Missing API Key. Add to .env file")

UNIT_TO_DAYS = {"days": 1.0, "weeks": 7.0, "months": 30.4}

BAND_PAD = 0.25

DEFAULT_MODEL = "gemini-3.8-flash"

CONFIG = {"benchmarks_path": Path("injury_data/benchmarks.csv"),
          "dataset_path": Path("injury_data/dataset.csv"),
          "coverage": ""}


TOOL_CALLS = []
TOOL_LOG = []

DISCLAIMER = ("Typical times come from published studies of athletes. Individual recovery varies, "
              "so a clinician should guide any real decision.")

def _log(text):
    TOOL_LOG.append(text)
    return text

def fmt_days(d):
    """Days as printed, with a weeks or months equivalent so the agent never does arithmetic."""
    if d is None or pd.isna(d):
        return "n/a"
    d = float(d)
    if d >= 60:
        return f"{d:.0f} d (~{d / 30.4:.1f} mo)"
    if d >= 14:
        return f"{d:.0f} d (~{d / 7:.1f} wk)"
    return f"{d:.0f} d"

SAMPLE_REPORTS = [
    {"report_id": "s1", "text": "Rams CB Alex Moreno suffered a Grade 2 hamstring strain in Sunday's loss and is expected to miss 2-4 weeks, the team announced."},
    {"report_id": "s2", "text": "Guard Devon Pierce underwent surgery to repair a torn ACL in his left knee and will miss the remainder of the season."},
    {"report_id": "s3", "text": "Forward Sam Okafor (right ankle sprain) is day-to-day and listed as questionable for Tuesday's game."},
    {"report_id": "s4", "text": "Midfielder Lena Vogt has been ruled out for six weeks with a fractured fifth metatarsal."},
    {"report_id": "s5", "text": "QB Marcus Hale tore his ACL in practice, but the team expects him back in two weeks."},
    {"report_id": "s6", "text": "The coach praised the team's defense after Saturday's win and said the offense needs work."},
]

## Output

class ReportRecord(BaseModel):
    is_injury_report: bool = Field(description="True if the text reports or updates a specific injury.")
    player: Optional[str] = Field(default=None, description="Player name as written.")
    team: Optional[str] = Field(default=None, description="Team as written.")
    sport_or_league: Optional[str] = Field(default=None, description="e.g. NFL, NBA, soccer. Only if stated or obvious.")
    report_type: Optional[Literal["new_injury", "status_update", "return_from_injury", "surgery", "other"]] = None
    injury_type: Optional[str] = Field(default=None, description="e.g. strain, sprain, fracture, tear, concussion.")
    body_region: Optional[str] = Field(default=None, description="e.g. knee, hamstring, ankle.")
    specific_structure: Optional[str] = Field(default=None, description="e.g. ACL, MCL, fifth metatarsal.")
    side: Optional[Literal["left", "right", "both"]] = None
    severity_or_grade: Optional[str] = Field(default=None, description="e.g. 'Grade 2', 'mild', 'complete tear'.")
    treatment: Literal["surgical", "conservative", "mixed", "unspecified"] = "unspecified"
    availability_status: Optional[Literal["out", "doubtful", "questionable", "day_to_day",
                                          "limited", "probable", "available", "unknown"]] = None
    timeline_text: Optional[str] = Field(default=None, description="The exact phrase stating the recovery/absence timeline.")
    timeline_low: Optional[float] = Field(default=None, description="Lower number in the stated timeline, as written.")
    timeline_high: Optional[float] = Field(default=None, description="Upper number; same as low for a single number; null if open-ended.")
    timeline_unit: Optional[Literal["days", "weeks", "months"]] = Field(default=None, description="Unit used in the text. Do not convert.")
    timeline_category: Literal["specific_range", "day_to_day", "season_ending", "indefinite", "not_stated"] = "not_stated"
    benchmark_injury: Optional[str] = Field(default=None, description="'injury' value of the best-matching row from the lookup tool.")
    benchmark_treatment: Optional[str] = Field(default=None, description="'treatment' value of that row.")
    benchmark_outcome: Optional[str] = Field(default=None, description="'outcome_definition' value of that row.")
    evidence_quote: Optional[str] = Field(default=None, description="The sentence from the report supporting the injury and timeline, copied verbatim.")


## Benchmarks

def load_benchmarks():
    path = CONFIG["benchmarks_path"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["injury"] = df["injury"].astype(str).str.lower().str.strip()
    return df

def load_findings():
    path = CONFIG["dataset_path"]
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["usable"] = df["usable"].astype(str).str.lower().eq("true")
    df["injury_key"] = df["injury"].astype(str).str.lower().str.strip()
    return df

def name_mask(series, name):
    """Rows whose value equals the name, or contains it, or is contained in it."""
    key = (name or "").lower().strip()
    values = series.astype(str).str.lower().str.strip()
    if not key:
        return values.ne(values)  # all False
    exact = values == key
    if exact.any():
        return exact
    return values.str.contains(key, regex=False) | values.apply(lambda v: bool(v) and v in key)

def candidate_rows(df, injury, treatment=""):
    """Benchmark rows whose injury matches the name (exactly, or as a substring either way)."""
    key = (injury or "").lower().strip()
    if not key:
        return df.iloc[0:0]
    rows = df[df["injury"] == key]
    if rows.empty:
        rows = df[df["injury"].str.contains(key, regex=False) | df["injury"].apply(lambda x: x in key)]
    if treatment and treatment in set(rows["treatment"]):
        rows = rows[rows["treatment"] == treatment]
    return rows.sort_values("n_papers", ascending=False)

def coverage_summary(bench_df, max_items=80):
    """Which injuries the dataset covers, so the agent can map a question to a standard name."""
    top = bench_df.groupby("injury")["n_findings"].sum().sort_values(ascending=False).head(max_items)
    return "; ".join(f"{name} ({int(n)} findings)" for name, n in top.items())


@tool("Lookup recovery benchmark")
def lookup_recovery_benchmark(injury: str, treatment: str = "") -> str:
    """Look up typical recovery times from published research for an injury.

    Args:
        injury: standard injury name such as "hamstring strain" or "ACL tear".
        treatment: optional, one of "surgical", "conservative", "mixed", "unspecified".

    Returns matching benchmark rows (injury, treatment, outcome_definition, n_papers, median and
    interquartile range in days). If nothing matches, returns the injuries that do have
    benchmarks so you can retry once with a synonym.
    """
    TOOL_CALLS.append("lookup")
    df = load_benchmarks()
    if df is None:
        return "No benchmark data is available. Leave the benchmark_* fields empty."
    rows = candidate_rows(df, injury, treatment).head(8)
    if rows.empty:
        return (f"No benchmark found for '{injury}'. Injuries with benchmarks: "
                f"{', '.join(sorted(df['injury'].unique()))}. Retry once with a synonym; if nothing "
                f"fits, report that there is no benchmark (when extracting, leave the benchmark_* "
                f"fields empty).")
    lines = ["injury | treatment | outcome_definition | n_papers | median | p25 to p75 | min to max"]
    for _, r in rows.iterrows():
        lines.append(f"{r['injury']} | {r['treatment']} | {r['outcome_definition']} | "
                     f"{int(r['n_papers'])} | {fmt_days(r['median_days'])} | "
                     f"{fmt_days(r['p25_days'])} to {fmt_days(r['p75_days'])} | "
                     f"{fmt_days(r['min_days'])} to {fmt_days(r['max_days'])}")
    return _log("\n".join(lines))



@tool("Search individual findings")
def search_findings(injury: str, severity: str = "", treatment: str = "", outcome: str = "") -> str:
    """Search the individual study findings behind the benchmarks.

    Args:
        injury: standard injury name such as "hamstring strain".
        severity: optional text such as "grade 2" or "complete".
        treatment: optional, "surgical", "conservative", "mixed" or "unspecified".
        outcome: optional outcome definition such as "return_to_full_competition".

    Returns up to 8 of the best-supported findings (largest samples first) with PMID, year, study
    type, severity, treatment, outcome, the mean or median reported, the range and the sample size.
    Use it to explain differences by severity or treatment, or when the user asks for sources.
    """
    TOOL_CALLS.append("search")
    df = load_findings()
    if df is None:
        return "No individual-findings file is available."
    rows = df[df["usable"] & name_mask(df["injury_key"], injury)]
    if rows.empty:
        return f"No findings for '{injury}'."

    notes = []
    for label, column, value in (("severity", "severity_or_grade", severity),
                                 ("treatment", "treatment", treatment),
                                 ("outcome", "outcome_definition", outcome)):
        if not value:
            continue
        text = rows[column].astype(str).str.lower()
        narrowed = rows[text.str.contains(value.lower().strip(), regex=False)]
        if narrowed.empty:
            notes.append(f"No findings match {label} '{value}'; showing all {label} values.")
        else:
            rows = narrowed

    rows = rows.sort_values("n", ascending=False, na_position="last")
    lines = notes + [f"{len(rows)} matching findings; showing the {min(len(rows), 8)} with the "
                     f"largest samples:"]
    for _, r in rows.head(8).iterrows():
        study = str(r["pub_types"]).split(";")[0].strip() if pd.notna(r["pub_types"]) else ""
        year = str(r["year"]).split(".")[0] if pd.notna(r["year"]) else "year unknown"
        grade = r["severity_or_grade"] if pd.notna(r["severity_or_grade"]) else "severity not stated"
        spread = ""
        if pd.notna(r["low_days"]) and pd.notna(r["high_days"]):
            spread = f" [{r['dispersion_type']} {r['low_days']:.0f} to {r['high_days']:.0f} d]"
        n = f"n={int(r['n'])}" if pd.notna(r["n"]) else "n not reported"
        lines.append(f"PMID {r['pmid']} ({year}, {study or 'study type unknown'}) | {grade} | "
                     f"{r['treatment']} | {r['outcome_definition']} | "
                     f"{r['statistic_type']} {fmt_days(r['value_days'])}{spread} | {n}")
    return _log("\n".join(lines))



## Building Crew

def build_crew(model, verbose):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    llm = LLM(model=f"gemini/{model}", api_key=api_key, temperature=0)

    advisor = Agent(
        role="Sports Injury Recovery Advisor",
        goal="Answer questions about how long sports injuries take to recover, using only the "
             "literature dataset.",
        backstory="A sports-medicine research analyst. You never answer from memory: every number "
                  "you give comes from the dataset tools, and you say plainly when the dataset does "
                  "not cover a question.",
        tools=[lookup_recovery_benchmark, search_findings], llm=llm,
        allow_delegation=False, max_iter=8, verbose=verbose)

    task = Task(
        description=(
            "Answer the user's question about an injury and its recovery time.\n\n"
            "QUESTION:\n{question}\n\n"
            "EARLIER IN THIS CONVERSATION (may be empty):\n{history}\n\n"
            "INJURIES THE DATASET COVERS (name and number of findings):\n{coverage}\n\n"
            "Rules:\n"
            "- Never answer from your own knowledge. Every number must come from a tool result.\n"
            "- First call 'Lookup recovery benchmark' with the standard injury name from the list "
            "above that best matches the question (retry at most once with a synonym).\n"
            "- Call 'Search individual findings' when the question mentions a severity, grade or "
            "treatment, when benchmark rows differ and you need to explain why, or when the user "
            "asks for sources.\n"
            "- Quote numbers exactly as the tools print them (days, with the weeks or months "
            "equivalent). Do not add, average, convert or estimate numbers yourself.\n"
            "- Lead with the direct answer: the typical time and range. Say what the time measures "
            "in plain words, such as return to full competition or symptom resolution, because "
            "different outcome definitions give very different numbers.\n"
            "- If the user quotes an expected timeline (for example from a team report), say in one "
            "sentence whether it is shorter than, within, or longer than the typical range in the "
            "data, quoting the range as printed.\n"
            "- Say how many papers support the answer. If a row has fewer than 3 papers, say the "
            "evidence is thin.\n"
            "- If surgical and conservative treatment, or different outcomes, both appear, give "
            "each in its own sentence.\n"
            "- If the injury is not in the dataset, say so and name the closest covered injuries. "
            "Do not fill the gap from general knowledge.\n"
            "- If the question is not about an injury's recovery time, say you can only answer "
            "recovery-timeline questions.\n"
            "- Keep the answer under 150 words, in plain language, and give no advice about an "
            "individual's care."),
        expected_output="A short plain-language answer grounded in the tool results.",
        agent=advisor)

    return Crew(agents=[advisor], tasks=[task], process=Process.sequential, verbose=verbose)



def answer_question(crew, question, history=""):
    """Returns (agent_answer, full_output). Python adds the data tables and a grounding check."""
    TOOL_CALLS.clear()
    result = crew.kickoff(inputs={"question": question, "history": history or "(none)",
                                  "coverage": CONFIG["coverage"]})
    answer = (getattr(result, "raw", "") or "").strip() or "The agent returned no answer."

    shown = answer
    if not TOOL_CALLS:
        shown = "[Unverified: the agent did not consult the dataset for this answer.]\n\n" + answer
    return answer, f"{shown}\n\n{DISCLAIMER}"


def format_history(turns):
    return "\n".join(f"Q: {q}\nA: {a[:400]}" for q, a in turns)


def chat_loop(crew):
    print("Ask about injury recovery times, e.g. 'How long does a grade 2 hamstring strain take?'")
    print("Press Enter on an empty line, or type 'quit', to exit.")
    turns = []
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        try:
            answer, output = answer_question(crew, question, format_history(turns))
        except Exception as exc:  # keep the chat alive
            print(f"\nSorry, that failed: {exc}")
            continue
        print("\n" + output)
        turns = (turns + [(question, answer)])[-3:]




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*",
                    help="your question about an injury; leave empty for an interactive chat")
    ap.add_argument("--benchmarks", default="injury_data/benchmarks.csv")
    ap.add_argument("--dataset", default="injury_data/dataset.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verbose", action="store_true", help="show the agent's reasoning")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY first, e.g. export GEMINI_API_KEY=your-key")

    CONFIG["benchmarks_path"] = Path(args.benchmarks)
    CONFIG["dataset_path"] = Path(args.dataset)

    bench_df = load_benchmarks()
    if bench_df is None or bench_df.empty:
        raise SystemExit(f"{args.benchmarks} not found or empty. Run extract_dataset.py first to "
                         f"build the dataset this agent reads.")
    if load_findings() is None:
        print(f"Note: {args.dataset} not found, so the agent can't search individual findings.")
    CONFIG["coverage"] = coverage_summary(bench_df)
    print(f"Loaded {len(bench_df)} benchmark groups covering {bench_df['injury'].nunique()} injuries.")

    crew = build_crew(args.model, args.verbose)
    if args.question:
        _, output = answer_question(crew, " ".join(args.question))
        print("\n" + output)
    else:
        chat_loop(crew)


if __name__ == "__main__":
    main()
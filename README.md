Sports Injury Recovery Advisor

DSC 565 project. A chatbot that answers questions about how long sports injuries take to recover, using only a dataset built from published PubMed studies rather than the language model's memory.

Live app: https://dsc565injuryagent-cscthdcl7dvzzzdwd95va6.streamlit.app/

Try asking: "How long does recovery from ACL tear take?" or "How long does a grade 2 hamstring strain take?"

How it works
Data collection (data_collection.py): searches PubMed for each injury in a list of ~45 common sports injuries, pulls abstracts (and PMC full text when available), and keeps only papers with sentences that mention both a duration and recovery language. Saved text lives in injury_data/text/.
Dataset (injury_data/dataset.csv): individual findings extracted from those papers (injury, severity, treatment, outcome definition, mean/median days, sample size, PMID).
Agent (injury_agent.py): a CrewAI agent powered by Gemini with two tools:
Lookup recovery benchmark: typical recovery times (median and interquartile range) per injury, treatment and outcome.
Search individual findings: the specific studies behind those numbers.
The agent is instructed to quote only numbers returned by these tools, say how many papers support an answer, and say so when an injury isn't covered. Every answer ends with a disclaimer, and answers that skipped the dataset are flagged as unverified.
Web app (app.py): a Streamlit chat interface with per-session and daily question limits to protect API quota.


# Sports Injury Recovery Advisor

An AI-powered chatbot that answers questions about sports injury recovery timelines using a dataset built from published **PubMed research studies** rather than relying on a language model's general knowledge.

**Course:** DSC 565 – Generative AI Project

**Live App:** [Sports Injury Recovery Advisor](https://dsc565injuryagent-cscthdcl7dvzzzdwd95va6.streamlit.app/)
<img width="1707" height="987" alt="Screenshot 2026-11-17 at 8 40 44 AM" src="https://github.com/user-attachments/assets/28e0dc86-a1cf-4ba6-95ea-470462a234b0" />


---

## Project Overview

The Sports Injury Recovery Advisor helps users explore how long common sports injuries typically take to recover. It uses published medical research to provide evidence-based estimates and allows users to investigate the individual studies supporting those estimates.

Unlike a general-purpose chatbot, the system is designed to ground its answers in a research dataset. If an injury is not represented in the available data, the agent is instructed to acknowledge that limitation rather than guess.

## Try It Out

Ask the chatbot questions such as:

- "How long does recovery from an ACL tear take?"
- "How long does a grade 2 hamstring strain take?"
- "What is the typical recovery time for an ankle sprain?"

---

## How It Works

The project consists of four main components:

### 1. PubMed Data Collection — `data_collection.py`

Collects published research from PubMed for approximately 45 common sports injuries.

- Searches PubMed for relevant injury and recovery studies.
- Retrieves article abstracts and available PMC full text.
- Identifies sentences containing both recovery-related language and time durations.
- Saves collected research text in `injury_data/text/`.

### 2. Research Dataset — `injury_data/dataset.csv`

Contains individual recovery findings extracted from the collected research papers.

Each finding can include:

- Injury type
- Injury severity
- Treatment method
- Recovery outcome definition
- Mean or median recovery time
- Sample size
- PubMed ID (PMID)

This dataset provides the evidence used by the chatbot.

### 3. AI Agent — `injury_agent.py`

Uses **CrewAI** and **Google Gemini** to answer questions using two custom tools:

| Tool | Purpose |
|---|---|
| **Lookup Recovery Benchmark** | Retrieves typical recovery times, including medians and interquartile ranges, grouped by injury, treatment, and recovery outcome. |
| **Search Individual Findings** | Retrieves individual research findings and the studies supporting the reported recovery estimates. |

The agent follows instructions designed to keep responses grounded in the dataset:

- Uses recovery numbers returned by the research tools.
- Reports how many research papers support an answer.
- Distinguishes between different recovery outcomes.
- Acknowledges when an injury is not covered by the available research.
- Avoids guessing when supporting data is unavailable.

Answers include a disclaimer, and responses that do not use the dataset tools are flagged as unverified.

### 4. Web Application — `app.py`

Provides an interactive chatbot interface built with **Streamlit**.

Features include:

- Chat-based questions and answers.
- Per-session question limits.
- Daily question limits to help protect API usage quotas.
- Research-grounded responses through the AI agent.

---

## Technologies Used

- **Python** – Core programming language
- **PubMed / NCBI Entrez** – Research paper collection
- **Biopython** – PubMed and PMC API access
- **Pandas** – Dataset processing and analysis
- **CrewAI** – AI agent and tool orchestration
- **Google Gemini** – Language model powering the agent
- **Streamlit** – Interactive web application

## Project Goals

The main goal is to demonstrate how generative AI can be combined with a structured research dataset to provide more transparent and evidence-based answers.

Rather than asking an AI model to generate recovery estimates from memory, the project retrieves information from published studies and uses the model to explain the findings.

## Limitations

- Recovery estimates depend on the studies collected and successfully processed.
- Not every injury, severity, treatment, or recovery outcome is represented.
- Published studies may use different definitions of recovery.
- Extracted findings should be interpreted in the context of their original research.
- The chatbot is an educational research tool, not a medical diagnosis or treatment recommendation.

---

## Live Demo

Try the application here:

**[Open Sports Injury Recovery Advisor](https://dsc565injuryagent-cscthdcl7dvzzzdwd95va6.streamlit.app/)**


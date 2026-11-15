import hmac
import logging
import os
import threading
from datetime import date
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()  # a local .env can supply GEMINI_API_KEY; hosts set real environment variables
except ImportError:
    pass

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def _int_env(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DAILY_LIMIT = _int_env("DAILY_QUESTION_LIMIT", 200)
SESSION_LIMIT = _int_env("SESSION_QUESTION_LIMIT", 15)
MAX_CHARS = 500

st.set_page_config(page_title="Sports Injury Recovery Advisor", page_icon="🩹")

@st.cache_resource
def load_agent():
    """Import the agent once and point it at the dataset. Returns (module, injuries, examples)."""
    # Imported here, after the API key check below, because injury_agent exits at import time
    # when no key is set.
    import injury_agent as agent

    data_dir = Path(os.getenv("INJURY_DATA_DIR") or HERE / "injury_data")
    agent.CONFIG["benchmarks_path"] = data_dir / "benchmarks.csv"
    agent.CONFIG["dataset_path"] = data_dir / "dataset.csv"
    bench = agent.load_benchmarks()
    if bench is None or bench.empty:
        raise FileNotFoundError(f"{data_dir / 'benchmarks.csv'} not found or empty. Run "
                                f"data_extraction.py, then include the injury_data folder.")
    agent.CONFIG["coverage"] = agent.coverage_summary(bench)
    counts = bench.groupby("injury")["n_findings"].sum().sort_values(ascending=False)
    return agent, sorted(counts.index), list(counts.head(3).index)


@st.cache_resource
def get_lock():
    """injury_agent keeps per-question state in module-level lists, so answer one question at a
    time. Visitors who ask at the same moment wait for their turn."""
    return threading.Lock()


@st.cache_resource
def get_usage():
    """One counter shared by every visitor, to protect your Gemini quota."""
    return {"lock": threading.Lock(), "day": date.today(), "count": 0}


def take_daily_slot():
    usage = get_usage()
    with usage["lock"]:
        today = date.today()
        if usage["day"] != today:
            usage["day"], usage["count"] = today, 0
        if usage["count"] >= DAILY_LIMIT:
            return False
        usage["count"] += 1
        return True


def ask(agent, model, question, turns):
    """Run one question through the agent. Returns (answer, text_to_show)."""
    with get_lock():
        tool_log = getattr(agent, "TOOL_LOG", None)
        if tool_log is not None:
            tool_log.clear()  # injury_agent never empties it, so it would grow forever
        crew = agent.build_crew(model, False)
        return agent.answer_question(crew, question, agent.format_history(turns))


def sidebar_button(label):
    try:
        return st.button(label, width="stretch")
    except TypeError:  # older Streamlit versions
        return st.button(label, use_container_width=True)


st.title("Sports Injury Recovery Advisor")
st.caption("Ask how long a sports injury takes to recover. Answers come from a dataset of "
           "published studies, not from the model's memory.")

if not API_KEY:
    st.error("GEMINI_API_KEY is not set. Add it as a secret or variable in your hosting settings, "
             "or to your .env file when running locally.")
    st.stop()
try:
    agent, injuries, examples = load_agent()
except (Exception, SystemExit) as exc:
    log.exception("could not start the agent")
    st.error(f"Could not start the agent: {exc}")
    st.stop()
model = os.getenv("GEMINI_MODEL") or agent.DEFAULT_MODEL

with st.sidebar:
    st.subheader("Try asking")
    for name in examples:
        question = f"How long does recovery from {name} take?"
        if sidebar_button(question):
            st.session_state.queued = question
    with st.expander("Injuries covered"):
        st.write(", ".join(injuries))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "turns" not in st.session_state:
    st.session_state.turns = []  # (question, answer) pairs the agent sees as chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

typed = st.chat_input("Ask about an injury, e.g. how long does an ACL tear take to recover?",
                      max_chars=MAX_CHARS)
prompt = st.session_state.pop("queued", None) or typed

if prompt:
    prompt = prompt.strip()[:MAX_CHARS]
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    asked = st.session_state.get("asked", 0)
    with st.chat_message("assistant"):
        if asked >= SESSION_LIMIT:
            reply = (f"You've reached the limit of {SESSION_LIMIT} questions for this session. "
                     f"Refresh the page to start a new session.")
        elif not take_daily_slot():
            reply = "This app has reached its daily question limit. Please try again tomorrow."
        else:
            st.session_state.asked = asked + 1
            with st.spinner("Checking the dataset..."):
                try:
                    answer, reply = ask(agent, model, prompt, st.session_state.turns)
                    st.session_state.turns = (st.session_state.turns + [(prompt, answer)])[-3:]
                except Exception:
                    log.exception("answering failed")
                    reply = "Sorry, something went wrong while answering. Please try again."
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})



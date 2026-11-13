import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd
from data_collection import COMMON_INJURIES
from crewai import Agent, Task, Crew, LLM
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from crewai.tools import BaseTool


load_dotenv()

if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    raise SystemExit("Missing API Key. Add to .env file")

gemini_llm = LLM(model=os.getenv("GEMINI_MODEL", "gemini/gemini-3.8-flash"))






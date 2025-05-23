"""investment_agents.py — End‑to‑end evaluation pipeline
Optimised for OpenAI‑Python ≥ 1.0 and the latest Agents‑SDK.
• Jinja2 for clean HTML rendering
• PDF processed in 20‑page chunks to stay within token limits
"""

from __future__ import annotations

import os
import sys
import json
import re
from typing import Any, Dict, List, Generator
from functools import lru_cache

import numpy as np
import requests
import openai
import jinja2

# -----------------------------------------------------------------------------
# Optional dependencies with helpful error messages
# -----------------------------------------------------------------------------
try:
    import faiss  # type: ignore
except ModuleNotFoundError as exc:
    raise SystemExit("Faiss missing → `pip install faiss-cpu` (or faiss-gpu).") from exc

try:
    import PyPDF2
except ModuleNotFoundError as exc:
    raise SystemExit("PyPDF2 missing → `pip install PyPDF2`. ") from exc

try:
    from agents import Agent, function_tool, Runner
except ModuleNotFoundError as exc:
    raise SystemExit("OpenAI Agents‑SDK missing → `pip install openai-agents`. ") from exc

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Env var {name} not set")
    return val

OPENAI_API_KEY  = _env("OPENAI_API_KEY")
SERPAPI_API_KEY = _env("SERPAPI_API_KEY")
MODEL_NAME      = os.getenv("AGENT_MODEL", "gpt-4o")  # default model

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=2)

# resilient HTTP session
_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503]))
)

# local storage paths
DATA_DIR   = os.getenv("AGENT_DATA_DIR", os.path.join(os.getcwd(), "data"))
REPORT_DIR = os.path.join(DATA_DIR, "reports")
INDEX_PATH = os.path.join(DATA_DIR, "vstore.index")
META_PATH  = os.path.join(DATA_DIR, "vstore_meta.json")
EMBED_DIM  = 1536
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# FAISS index setup
# -----------------------------------------------------------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata: List[Dict[str, Any]] = json.load(open(META_PATH))
else:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBED_DIM))
    metadata = []

# -----------------------------------------------------------------------------
# Embedding helper
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _embed(text: str) -> np.ndarray:
    vec = client.embeddings.create(model="text-embedding-ada-002", input=[text]).data[0].embedding
    arr = np.asarray(vec, dtype="float32")
    return arr / np.linalg.norm(arr)

# -----------------------------------------------------------------------------
# PDF extraction (chunked)
# -----------------------------------------------------------------------------
CHUNK_SIZE = 20  # pages


def _page_ranges(path: str) -> Generator[tuple[int, int], None, None]:
    reader = PyPDF2.PdfReader(path)
    total = len(reader.pages)
    for start in range(1, total + 1, CHUNK_SIZE):
        yield start, min(start + CHUNK_SIZE - 1, total)


def _extract_pdf(path: str) -> str:
    """Upload once, then extract text in 20‑page chunks."""
    with open(path, "rb") as pdf:
        file_ref = client.files.create(file=pdf, purpose="user_data")

    parts: List[str] = []
    for s, e in _page_ranges(path):
        prompt = (
            f"Extract raw text from pages {s}-{e} of attached PDF.\n"
            "Return ONLY plain text (no page markers)."
        )
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": file_ref.id}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=2048,
        )
        parts.append(resp.choices[0].message.content.strip())
    return "\n".join(parts)

parse_pdf = function_tool(_extract_pdf)

# -----------------------------------------------------------------------------
# Additional FunctionTools
# -----------------------------------------------------------------------------

@function_tool
def web_search(query: str) -> str:
    resp = _session.get(
        "https://serpapi.com/search",
        params={"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 5},
        timeout=10,
    )
    resp.raise_for_status()
    return "\n".join(f"- {d['title']} — {d['snippet']} ({d['link']})" for d in resp.json().get("organic_results", []))


@function_tool
def vector_memory(action: str, *, project: str = "", summary: str = "", keywords: List[str] | None = None, rationale: str = "") -> Any:
    global index, metadata
    if action == "add":
        blob = " ".join([project, summary, " ".join(keywords or []), rationale])
        idx = len(metadata)
        index.add_with_ids(np.expand_dims(_embed(blob), 0), np.array([idx]))
        metadata.append({"project": project, "summary": summary, "keywords": keywords or [], "rationale": rationale})
        faiss.write_index(index, INDEX_PATH)
        json.dump(metadata, open(META_PATH, "w"), indent=2)
        return "stored"
    if action == "query":
        vec = _embed(summary)
        D, I = index.search(np.expand_dims(vec, 0), 3)
        return [metadata[i] for i in I[0] if i != -1]
    if action == "list":
        return metadata
    raise ValueError("vector_memory action must be add|query|list")

# -----------------------------------------------------------------------------
# Agent definitions
# -----------------------------------------------------------------------------
shared_tools = [parse_pdf]
search_tools = [parse_pdf, web_search]
vector_tools = [vector_memory]

financial_agent  = Agent("FinancialHealthAgent",  "Extract financial metrics.",              tools=shared_tools, model=MODEL_NAME)
market_agent     = Agent("MarketOpportunityAgent", "Analyse market size & competition.",          tools=search_tools, model=MODEL_NAME)
risk_agent       = Agent("RiskAssessmentAgent",    "Identify major risks.",                       tools=search_tools, model=MODEL_NAME)
report_agent     = Agent("ReportAgent",            "Return JSON: summary, keywords[], metrics{}.", tools=[],           model=MODEL_NAME)
supervisor_agent = Agent("SupervisorAgent",        "Return YES/NO or RETRY <Agent>:<reason>.",    tools=vector_tools, model=MODEL_NAME)

AGENTS = {
    "FinancialHealthAgent": financial_agent,
    "MarketOpportunityAgent": market_agent,
    "RiskAssessmentAgent": risk_agent,
}

# -----------------------------------------------------------------------------
# Jinja2 HTML template
# -----------------------------------------------------------------------------
HTML_TMPL = jinja2.Template(
    """<!doctype html><html><head><meta charset='utf-8'><title>{{ project }}</title></head><body>
<h1>{{ project|e }}</h1>
<h2>Summary</h2><p>{{ summary|e }}</p>
<h2>Keywords</h2><ul>{% for kw in keywords %}<li>{{ kw|e }}</li>{% endfor %}</ul>
<h2>Metrics</h2><table border='1'><tr><th>Metric</th><th>Value</th></tr>{% for k, v in metrics.items() %}<tr><td>{{ k|e }}</td><td>{{ v|e }}</td></tr>{% endfor %}</table>
<h2>Decision</h2><p>{{ decision|e }}</p>
<h2>Rationale</h2><p>{{ rationale|e }}</p>
</body></html>"""
)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def evaluate(pdf: str, project: str) -> Dict[str, Any]:
    # 1) extract text
    text = _extract_pdf(pdf)

    # 2) initial analysis per sub-agent
    results = {name: Runner.run_sync(agent, text).final_output for name, agent in AGENTS.items()}

    while True:
        report_input = (
            f"Financial:\n{results['FinancialHealthAgent']}\n\n"
            f"Market:\n{results['Market

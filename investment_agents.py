"""investment_agents.py — end‑to‑end evaluation pipeline
Optimised for OpenAI‑Python ≥ 1.0 and the latest Agents‑SDK.
"""

from __future__ import annotations

import os
import sys
import json
import html
import re
from typing import Any, Dict, List
from functools import lru_cache

import numpy as np
import requests
import openai

# -----------------------------------------------------------------------------
# Optional deps with explicit error messages
# -----------------------------------------------------------------------------
try:
    import faiss  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Faiss is required. Install with `pip install faiss-cpu` or `faiss-gpu`.") from exc

try:
    from agents import Agent, function_tool, Runner
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("OpenAI Agents‑SDK missing. Install with `pip install openai-agents`. ") from exc

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable {name} not set.")
    return val

OPENAI_API_KEY   = _env("OPENAI_API_KEY")
SERPAPI_API_KEY  = _env("SERPAPI_API_KEY")
MODEL_NAME       = os.getenv("AGENT_MODEL", "gpt-4o")  # override if project has access

# -----------------------------------------------------------------------------
# OpenAI & HTTP clients
# -----------------------------------------------------------------------------
client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=2)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503]))
)

# -----------------------------------------------------------------------------
# Local storage paths
# -----------------------------------------------------------------------------
DATA_DIR   = os.getenv("AGENT_DATA_DIR", os.path.join(os.getcwd(), "data"))
REPORT_DIR = os.path.join(DATA_DIR, "reports")
INDEX_PATH = os.path.join(DATA_DIR, "vstore.index")
META_PATH  = os.path.join(DATA_DIR, "vstore_meta.json")
EMBED_DIM  = 1536  # ada‑002 dims
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# FAISS index initialisation (cosine via normalised IP)
# -----------------------------------------------------------------------------
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    metadata: List[Dict[str, Any]] = json.loads(open(META_PATH).read())
else:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBED_DIM))
    metadata = []

# -----------------------------------------------------------------------------
# Embedding + PDF helper functions
# -----------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _embed(text: str) -> np.ndarray:
    vec = client.embeddings.create(model="text-embedding-ada-002", input=[text]).data[0].embedding
    vec = np.asarray(vec, dtype="float32")
    return vec / np.linalg.norm(vec)


def _extract_pdf(file_path: str) -> str:
    """Upload PDF and let GPT‑Vision extract raw text."""
    with open(file_path, "rb") as pdf:
        up_file = client.files.create(file=pdf, purpose="user_data")

    comp = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "text"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_id": up_file.id}},
                    {"type": "text", "text": "Extract the full text of this PDF."},
                ],
            }
        ],
    )
    return comp.choices[0].message.content.strip()

# Register tools
parse_pdf = function_tool(_extract_pdf)


@function_tool
def web_search(query: str) -> str:
    """Lightweight Google search via SerpAPI (top 5)"""
    resp = _session.get(
        "https://serpapi.com/search",
        params={"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 5},
        timeout=10,
    )
    resp.raise_for_status()
    return "\n".join(
        f"- {it.get('title')} — {it.get('snippet')} ({it.get('link')})"
        for it in resp.json().get("organic_results", [])
    )


def vector_memory(action: str, *, project: str = "", summary: str = "",
                  keywords: List[str] | None = None, rationale: str = "") -> Any:
    """Add/query/list vectors in FAISS store."""
    global index, metadata

    if action == "add":
        blob = " ".join([project, summary, " ".join(keywords or []), rationale])
        vec_id = len(metadata)
        index.add_with_ids(np.expand_dims(_embed(blob), 0), np.array([vec_id]))
        metadata.append({"project": project, "summary": summary, "keywords": keywords or [], "rationale": rationale})
        faiss.write_index(index, INDEX_PATH)
        open(META_PATH, "w").write(json.dumps(metadata, indent=2))
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
shared_tools  = [parse_pdf]
search_tools  = [parse_pdf, web_search]
vector_tools  = [vector_memory]

financial_agent  = Agent("FinancialHealthAgent",  instructions="Extract key financial metrics.",              tools=shared_tools, model=MODEL_NAME)
market_agent     = Agent("MarketOpportunityAgent", instructions="Analyse market size & competition.",          tools=search_tools, model=MODEL_NAME)
risk_agent       = Agent("RiskAssessmentAgent",    instructions="Identify major financial & operational risks.", tools=search_tools, model=MODEL_NAME)
report_agent     = Agent("ReportAgent",            instructions="Return pure JSON: summary, keywords[], metrics{}.", tools=[], model=MODEL_NAME)
supervisor_agent = Agent("SupervisorAgent",        instructions="Return YES/NO or RETRY <Agent>:<reason>.",       tools=vector_tools, model=MODEL_NAME)

AGENT_MAP = {
    "FinancialHealthAgent": financial_agent,
    "MarketOpportunityAgent": market_agent,
    "RiskAssessmentAgent": risk_agent,
}

# -----------------------------------------------------------------------------
# HTML report helper (escaped)
# -----------------------------------------------------------------------------

def _html(path: str, project: str, summary: str, keywords: list[str], metrics: dict[str, Any], decision: str, rationale: str):
    esc = lambda s: html.escape(str(s))
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in metrics.items())
    body = f"<h1>{esc(project)}</h1><h2>Summary</h2><p>{esc(summary)}</p><h2>Keywords</h2><ul>{''.join(f'<li>{esc(k)}</li>' for k in keywords)}</ul><h2>Metrics</h2><table>{rows}</table><h2>Decision</h2><p>{esc(decision)}</p><h2>Rationale</h2><p>{esc(rationale)}</p>"
    open(path, "w", encoding="utf-8").write(f"<!doctype html><html><head><meta charset='utf-8'></head><body>{body}</body></html>")

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def evaluate(pdf: str, project: str) -> Dict[str, Any]:
    text = _extract_pdf(pdf)

    # initial pass
    results = {k: Runner.run_sync(v, text).final_output for k, v in AGENT_MAP.items()}

    while True:
        report_payload = (
            f"Financial:\n{results['FinancialHealthAgent']}\n\n"
            f"Market:\n{results['MarketOpportunityAgent']}\n\n"
            f"Risk:\n{results['RiskAssessmentAgent']}"
        )
        raw = Runner.run_sync(report_agent, report_payload).final_output.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise ValueError("ReportAgent returned no JSON")
            data = json.loads(m.group(0))

        summary, keywords, metrics = data["summary"], data["keywords"], data["metrics"]
        sup_in = (
            f"Summary:\n{summary}\n"
            f"Keywords:{keywords}\n"
            f"Metrics:{metrics}\n"
            f"Past:{vector_memory('query', summary=str(summary))}"
        )
        sup_raw = Runner.run_sync(supervisor_agent, sup_in).final_output.strip()

        decision, *rationale_parts = sup_raw.split("\n", 1)
        rationale = rationale_parts[0] if rationale_parts else ""
        break  # exit while

    # persist & export
    vector_memory("add", project=project, summary=summary, keywords=keywords, rationale=rationale)
    html_path = os.path.join(REPORT_DIR, f"{project}.html")
    _html(html_path, project, summary, keywords, metrics, decision, rationale)

    return {
        **results,
        "summary": summary,
        "keywords": keywords,
        "metrics": metrics,
        "decision": decision,
        "rationale": rationale,
        "html": html_path,
    }

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python investment_agents.py <pdf_path> <project_name>")
    output = evaluate(sys.argv[1], sys.argv[2])
    print("Report saved to", output["html"])

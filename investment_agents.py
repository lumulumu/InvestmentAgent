"""investment_agents.py — end‑to‑end evaluation pipeline
Optimised for OpenAI‑Python ≥ 1.0 and the latest Agents‑SDK.
"""

from __future__ import annotations

import os
import sys
import json
from jinja2 import Environment, select_autoescape
import re
from typing import Any, Dict, List
from functools import lru_cache
import tempfile
import asyncio

import numpy as np
import requests
import openai
import logging
import time
from PyPDF2 import PdfReader, PdfWriter

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
    """Return an environment variable or abort if missing."""
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"Required environment variable {name} not set.")
    return val

OPENAI_API_KEY   = _env("OPENAI_API_KEY")
SERPAPI_API_KEY  = _env("SERPAPI_API_KEY")
MODEL_NAME       = os.getenv("AGENT_MODEL", "gpt-4o")  # override if project has access

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRY_POLICY = Retry(total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503])

# -----------------------------------------------------------------------------
# OpenAI & HTTP clients
# -----------------------------------------------------------------------------
client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30, max_retries=2)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=RETRY_POLICY)
)


def _openai_call(func, *args, **kwargs):
    """Call an OpenAI client function with retries."""
    for attempt in range(RETRY_POLICY.total):
        try:
            return func(*args, **kwargs)
        except openai.OpenAIError as exc:  # pragma: no cover - network
            logger.error(
                "OpenAI call %s failed (%d/%d): %s",
                func.__name__,
                attempt + 1,
                RETRY_POLICY.total,
                exc,
            )
            if attempt == RETRY_POLICY.total - 1:
                raise
            time.sleep(RETRY_POLICY.backoff_factor * (2 ** attempt))


def _serpapi_request(params: Dict[str, Any]) -> requests.Response | None:
    """Perform a SerpAPI request with retries."""
    for attempt in range(RETRY_POLICY.total):
        try:
            resp = _session.get(
                "https://serpapi.com/search",
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # pragma: no cover - network
            logger.error(
                "SerpAPI request failed (%d/%d): %s",
                attempt + 1,
                RETRY_POLICY.total,
                exc,
            )
            if attempt == RETRY_POLICY.total - 1:
                return None
            time.sleep(RETRY_POLICY.backoff_factor * (2 ** attempt))

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
    """Return the normalised embedding for ``text``."""
    vec = _openai_call(
        client.embeddings.create,
        model="text-embedding-ada-002",
        input=[text],
    ).data[0].embedding
    vec = np.asarray(vec, dtype="float32")
    return vec / np.linalg.norm(vec)


def _extract_pdf(file_path: str, chunk_pages: int = 20) -> str:
    """Upload ``file_path`` in page chunks and let GPT‑Vision return the text."""

    reader = PdfReader(open(file_path, "rb"))
    text_parts: List[str] = []

    for start in range(0, len(reader.pages), chunk_pages):
        writer = PdfWriter()
        for p in range(start, min(start + chunk_pages, len(reader.pages))):
            writer.add_page(reader.pages[p])
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            writer.write(tmp)
            tmp.flush()
            tmp_path = tmp.name  # Merke den Pfad

        with open(tmp_path, "rb") as f:
            up_file = _openai_call(client.files.create, file=f, purpose="user_data")

        # Optional: temporäre Datei löschen
        os.remove(tmp_path)

        comp = _openai_call(
            client.chat.completions.create,
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
        text_parts.append(comp.choices[0].message.content.strip())
        try:
            _openai_call(client.files.delete, up_file.id)
        except Exception:
            pass  # ignore cleanup errors

    return "\n".join(text_parts)

# Register tools
parse_pdf = function_tool(_extract_pdf)


@function_tool
def web_search(query: str) -> str:
    """Perform a lightweight Google search via SerpAPI and return the top results."""
    resp = _serpapi_request({"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 5})
    if not resp:
        return f"SerpAPI error retrieving results for '{query}'"
    return "\n".join(
        f"- {it.get('title')} — {it.get('snippet')} ({it.get('link')})"
        for it in resp.json().get("organic_results", [])
    )


def vector_memory_impl(action: str, *, project: str = "", summary: str = "",
                      keywords: List[str] | None = None, rationale: str = "") -> Any:
    """Add, query or list vectors in the persistent FAISS store."""
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

# Tool-Variante für den Agenten
vector_memory_tool = function_tool(vector_memory_impl)
# Expose the implementation directly for local calls
vector_memory = vector_memory_impl

# -----------------------------------------------------------------------------
# Agent definitions
# -----------------------------------------------------------------------------
shared_tools  = [parse_pdf]
search_tools  = [parse_pdf, web_search]
vector_tools  = [vector_memory_tool]

financial_agent  = Agent("FinancialHealthAgent",  instructions="Extract key financial metrics.",              tools=shared_tools, model=MODEL_NAME)
market_agent     = Agent("MarketOpportunityAgent", instructions="Analyse market size & competition.",          tools=search_tools, model=MODEL_NAME)
risk_agent       = Agent("RiskAssessmentAgent",    instructions="Identify major financial & operational risks.", tools=search_tools, model=MODEL_NAME)
report_agent     = Agent("ReportAgent",            instructions="Return ONLY valid minified JSON: {\"summary\":..., \"keywords\":[], \"metrics\":{}}. No comments, no trailing commas, no explanations.", tools=[], model=MODEL_NAME)
supervisor_agent = Agent(
    "SupervisorAgent",
    instructions="Return YES/NO or RETRY <Agent>:<reason>.",
    tools=vector_tools,
    model=MODEL_NAME,
)

# Mapping from agent names to instances for orchestration
AGENT_MAP = {
    "FinancialHealthAgent": financial_agent,
    "MarketOpportunityAgent": market_agent,
    "RiskAssessmentAgent": risk_agent,
}

# -----------------------------------------------------------------------------
# HTML report helper using Jinja2 template (autoescaped)
# -----------------------------------------------------------------------------

_env = Environment(autoescape=select_autoescape(["html"]))
_HTML_TEMPLATE = _env.from_string(
    """<!doctype html><html><head><meta charset='utf-8'></head><body>"
    "<h1>{{ project }}</h1>"
    "<h2>Summary</h2><p>{{ summary }}</p>"
    "<h2>Keywords</h2><ul>{% for k in keywords %}<li>{{ k }}</li>{% endfor %}</ul>"
    "<h2>Metrics</h2><table>{% for k, v in metrics.items() %}<tr><td>{{ k }}</td><td>{{ v }}</td></tr>{% endfor %}</table>"
    "<h2>Decision</h2><p>{{ decision }}</p>"
    "<h2>Rationale</h2><p>{{ rationale }}</p>"
    "</body></html>"""  # noqa: E501
)

def _html(path: str, project: str, summary: str, keywords: list[str], metrics: dict[str, Any], decision: str, rationale: str):
    """Write a small HTML report summarising the agent output."""
    html_content = _HTML_TEMPLATE.render(
        project=project,
        summary=summary,
        keywords=keywords,
        metrics=metrics,
        decision=decision,
        rationale=rationale,
    )
    open(path, "w", encoding="utf-8").write(html_content)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

async def evaluate(pdf: str, project: str) -> Dict[str, Any]:
    """Run the full agent pipeline on ``pdf`` and return the aggregated result."""
    text = _extract_pdf(pdf)

    # Run all specialist agents on the extracted text concurrently
    tasks = {k: Runner.run(v, text) for k, v in AGENT_MAP.items()}
    raw_results = await asyncio.gather(*tasks.values())
    results = {k: r.final_output for k, r in zip(tasks.keys(), raw_results)}

    while True:
        # Supervisor may request a retry of one of the agents. Currently we only
        # run a single pass, but the loop allows for future extensions.
        report_payload = (
            f"Financial:\n{results['FinancialHealthAgent']}\n\n"
            f"Market:\n{results['MarketOpportunityAgent']}\n\n"
            f"Risk:\n{results['RiskAssessmentAgent']}"
        )
        raw = Runner.run_sync(report_agent, report_payload).final_output.strip()
        print("RAW OUTPUT FROM REPORT AGENT:\n", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise ValueError("ReportAgent returned no JSON")
            print("REGEX-MATCHED JSON:\n", m.group(0))
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

    # Persist the interaction history and write the HTML report
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
    import asyncio

    async def main():
        output = await evaluate(sys.argv[1], sys.argv[2])
        print("Report saved to", output["html"])

    try:
        asyncio.run(main())
    except RuntimeError as e:
        # Fallback für laufenden Event-Loop (z.B. Jupyter, VS Code)
        if "already running" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
        else:
            raise

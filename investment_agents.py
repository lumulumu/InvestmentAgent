"""investment_agents.py — end‑to‑end evaluation pipeline
Optimised for OpenAI‑Python ≥ 1.0 and the latest Agents‑SDK.
"""

from __future__ import annotations

import os
import sys
import json
import re
from typing import Any, Dict, List
from dataclasses import dataclass
from functools import lru_cache
import tempfile
import asyncio

import numpy as np
import requests
import openai
import logging
import time
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject, DictionaryObject, StreamObject


@dataclass
class EvaluationResult:
    """Return type for :func:`evaluate`."""

    summary: str
    keywords: List[str]
    metrics: Dict[str, Any]
    decision: str
    rationale: str
    markdown: str
    pdf: str

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
from vector_store import VectorStore

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
MODEL_NAME       = os.getenv("AGENT_MODEL", "o4-mini")  # override if project has access

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRY_POLICY = Retry(total=3, backoff_factor=.5, status_forcelist=[429, 500, 502, 503])

# -----------------------------------------------------------------------------
# OpenAI & HTTP clients
# -----------------------------------------------------------------------------
client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=120, max_retries=2)

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
# Vector store initialisation
# -----------------------------------------------------------------------------
vstore = VectorStore(INDEX_PATH, META_PATH, EMBED_DIM)
vstore.load()

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


def vector_memory_impl(
    action: str,
    *,
    project: str = "",
    summary: str = "",
    keywords: List[str] | None = None,
    rationale: str = "",
) -> Any:
    """Add, query or list vectors in the persistent store."""

    if action == "add":
        blob = " ".join([project, summary, " ".join(keywords or []), rationale])
        vstore.add(_embed(blob), {
            "project": project,
            "summary": summary,
            "keywords": keywords or [],
            "rationale": rationale,
        })
        return "stored"

    if action == "query":
        vec = _embed(summary)
        return vstore.query(vec)

    if action == "list":
        return vstore.list()

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

financial_agent  = Agent("FinancialHealthAgent",  instructions="Extract key financial metrics. Return bullet points or \"metric: value\" pairs. Give an detailed assessment like an M&A Specialist of the Companies Financial Situation. incl. the business model and a potential outlook into the next 5 years.",              tools=shared_tools, model=MODEL_NAME)
market_agent     = Agent("MarketOpportunityAgent", instructions="Analyse market size and competition. Create an executive summary summarising the opportunity. List The 5 biggest competitors. Highligh key products.",          tools=search_tools, model=MODEL_NAME)
risk_agent       = Agent("RiskAssessmentAgent",    instructions="Identify major financial and operational risks. What are the biggest risks of this investment opportunity considering the Market, Product and Macro Economic situation.", tools=search_tools, model=MODEL_NAME)
alveus_fit_agent = Agent(
    "AlveusFitAgent",
    instructions="Bewerte den Alveus-Fit: Distanz des Targets zu M\u00fcnchen, mindestens 50 Mitarbeiter, vorhandene zweite F\u00fchrungsebene und deren Gestaltung sowie ein EBIT von mindestens 1,5 Mio Euro. Gib stichpunktartige Einsch\u00e4tzungen oder \"Faktor: Details\" zur\u00fcck.",
    tools=search_tools,
    model=MODEL_NAME,
)
report_agent     = Agent(
    "ReportAgent",
    instructions=(
        "Return a concise Markdown report containing sections minimum the following sections but can be much longer: 'Executive Summary', "
        " 'Metrics', 'Decision', 'Alveus Fit', 'Risk', 'Rationale', 'Long Report' . Use bullet lists where appropriate."
    ),
    tools=[],
    model=MODEL_NAME,
)
supervisor_agent = Agent(
    "SupervisorAgent",
    instructions="Return YES/NO or RETRY <Agent>:<reason>. Use the \"Past:\" section, sourced from vector_memory, to judge whether current results improve on previous ones. You are the investment manager and responsible to create an actionable report for the investment committee. If you decide to retry an agent, provide a reason why the previous result was not sufficient.",
    tools=vector_tools,
    model=MODEL_NAME,
)

# Mapping from agent names to instances for orchestration
AGENT_MAP = {
    "FinancialHealthAgent": financial_agent,
    "MarketOpportunityAgent": market_agent,
    "RiskAssessmentAgent": risk_agent,
    "AlveusFitAgent": alveus_fit_agent,
}

# -----------------------------------------------------------------------------
# Markdown report helper
# -----------------------------------------------------------------------------

def _write_markdown(path: str, content: str) -> None:
    """Persist ``content`` to ``path`` as UTF-8."""
    open(path, "w", encoding="utf-8").write(content)


def _markdown_to_pdf(content: str, path: str) -> None:
    """Render Markdown ``content`` to a very simple PDF file."""
    writer = PdfWriter()
    width, height = 595, 842  # A4 size in points
    page = writer.add_blank_page(width=width, height=height)

    # Basic Helvetica font resource
    font_dict = {
        NameObject("/F1"): DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
    }
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject(font_dict)})

    y = height - 40
    lines = ["BT", "/F1 12 Tf"]
    for line in content.splitlines():
        line = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        lines.append(f"1 0 0 1 50 {int(y)} Tm ({line}) Tj")
        y -= 14
        if y < 40:
            break  # single page only
    lines.append("ET")

    stream = StreamObject()
    stream._data = "\n".join(lines).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with open(path, "wb") as fh:
        writer.write(fh)


def _extract_md_section(md: str, heading: str) -> str:
    """Return the text underneath ``# {heading}``."""
    pattern = rf"# {re.escape(heading)}\n(.*?)(?:\n#|$)"
    m = re.search(pattern, md, re.S)
    return m.group(1).strip() if m else ""


def _extract_md_list(md: str, heading: str) -> List[str]:
    section = _extract_md_section(md, heading)
    return [re.sub(r"^[-*]\s*", "", line).strip() for line in section.splitlines() if line.strip()]


def _extract_md_kv(md: str, heading: str) -> Dict[str, Any]:
    section = _extract_md_section(md, heading)
    out: Dict[str, Any] = {}
    for line in section.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip()] = v.strip()
    return out

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

async def evaluate(pdf: str, project: str) -> EvaluationResult:
    """Run the full agent pipeline on ``pdf`` and return the aggregated result."""
    text = _extract_pdf(pdf)

    # Run all specialist agents on the extracted text concurrently
    tasks = {k: Runner.run(v, text) for k, v in AGENT_MAP.items()}
    raw_results = await asyncio.gather(*tasks.values())
    results = {k: r.final_output for k, r in zip(tasks.keys(), raw_results)}

    # Compile supervisor input from specialist results and past projects
    sup_in = (
        f"Financial:\n{results['FinancialHealthAgent']}\n\n"
        f"Market:\n{results['MarketOpportunityAgent']}\n\n"
        f"Risk:\n{results['RiskAssessmentAgent']}\n\n"
        f"AlveusFit:\n{results['AlveusFitAgent']}\n\n"
        f"Past:{vector_memory('query', summary=' '.join(results.values()))}"
    )
    sup_raw = Runner.run_sync(supervisor_agent, sup_in).final_output.strip()

    decision, *rationale_parts = sup_raw.split("\n", 1)
    rationale = rationale_parts[0] if rationale_parts else ""

    report_payload = (
        f"Financial:\n{results['FinancialHealthAgent']}\n\n"
        f"Market:\n{results['MarketOpportunityAgent']}\n\n"
        f"Risk:\n{results['RiskAssessmentAgent']}\n\n"
        f"AlveusFit:\n{results['AlveusFitAgent']}\n\n"
        f"Decision:\n{decision}\n\n"
        f"Rationale:\n{rationale}"
    )
    markdown = Runner.run_sync(report_agent, report_payload).final_output.strip()
    print("RAW OUTPUT FROM REPORT AGENT:\n", markdown)

    summary = _extract_md_section(markdown, "Summary")
    keywords = _extract_md_list(markdown, "Keywords")
    metrics = _extract_md_kv(markdown, "Metrics")

    # Persist the interaction history and write the HTML report
    vector_memory("add", project=project, summary=summary, keywords=keywords, rationale=rationale)
    md_path = os.path.join(REPORT_DIR, f"{project}.md")
    _write_markdown(md_path, markdown)
    pdf_path = os.path.join(REPORT_DIR, f"{project}.pdf")
    try:
        _markdown_to_pdf(markdown, pdf_path)
    except Exception as exc:  # pragma: no cover - optional rendering errors
        logger.error("Failed to render PDF: %s", exc)
        pdf_path = ""

    return EvaluationResult(
        summary=summary,
        keywords=keywords,
        metrics=metrics,
        decision=decision,
        rationale=rationale,
        markdown=md_path,
        pdf=pdf_path,
    )

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python investment_agents.py <pdf_path> <project_name>")
    import asyncio

    async def main():
        output = await evaluate(sys.argv[1], sys.argv[2])
        print("Report saved to", output.markdown)
        if output.pdf:
            print("PDF version:", output.pdf)

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

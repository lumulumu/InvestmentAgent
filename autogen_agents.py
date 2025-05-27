"""autogen_agents.py -- AutoGen-based evaluation pipeline
This module mirrors investment_agents.py using Microsoft AutoGen for agent orchestration.
It focuses on memory usage by combining AutoGen's summarising memory with the
existing FAISS-based vector store.
"""

from __future__ import annotations

import os
import re
import asyncio
import tempfile
from typing import Any, Dict, List
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import openai
from PyPDF2 import PdfReader, PdfWriter
import markdown2
import pdfkit

# AutoGen imports -- only loaded when the module is used
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_core.memory import ListMemory
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Microsoft AutoGen is required. Install with `pip install pyautogen`."
    ) from exc

from vector_store import VectorStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RETRY_ATTEMPTS = 3
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
MODEL_NAME = os.getenv("AGENT_MODEL", "gpt-4o")
DATA_DIR = os.getenv("AGENT_DATA_DIR", os.path.join(os.getcwd(), "data"))
REPORT_DIR = os.path.join(DATA_DIR, "reports")
INDEX_PATH = os.path.join(DATA_DIR, "vstore.index")
META_PATH = os.path.join(DATA_DIR, "vstore_meta.json")
EMBED_DIM = 1536
os.makedirs(REPORT_DIR, exist_ok=True)

client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=120, max_retries=2)

vstore = VectorStore(INDEX_PATH, META_PATH, EMBED_DIM)
vstore.load()

# ---------------------------------------------------------------------------
# PDF + Embedding helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _embed(text: str) -> np.ndarray:
    vec = client.embeddings.create(model="text-embedding-ada-002", input=[text]).data[0].embedding
    arr = np.asarray(vec, dtype="float32")
    return arr / np.linalg.norm(arr)


def _extract_pdf(file_path: str, chunk_pages: int = 20) -> str:
    reader = PdfReader(open(file_path, "rb"))
    text_parts: List[str] = []
    for start in range(0, len(reader.pages), chunk_pages):
        writer = PdfWriter()
        for p in range(start, min(start + chunk_pages, len(reader.pages))):
            writer.add_page(reader.pages[p])
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            writer.write(tmp)
            tmp.flush()
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            up_file = client.files.create(file=f, purpose="user_data")
        os.remove(tmp_path)
        comp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "text"},
            messages=[
                {"role": "user", "content": [{"type": "file", "file": {"file_id": up_file.id}},
                                               {"type": "text", "text": "Extract the full text of this PDF."}]}
            ],
        )
        text_parts.append(comp.choices[0].message.content.strip())
        try:
            client.files.delete(up_file.id)
        except Exception:
            pass
    return "\n".join(text_parts)


def web_search(query: str) -> str:
    """Lightweight web search via SerpAPI."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(total=RETRY_ATTEMPTS, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))

    resp = session.get(
        "https://serpapi.com/search",
        params={"q": query, "engine": "google", "api_key": SERPAPI_API_KEY, "num": 5},
        timeout=10,
    )
    if resp.status_code != 200:
        return f"SerpAPI error: {resp.text}"
    data = resp.json().get("organic_results", [])
    return "\n".join(f"- {it.get('title')} — {it.get('snippet')} ({it.get('link')})" for it in data)


# Vector memory tool
def vector_memory(action: str, *, project: str = "", summary: str = "", keywords: List[str] | None = None, rationale: str = "") -> Any:
    if action == "add":
        blob = " ".join([project, summary, " ".join(keywords or []), rationale])
        vstore.add(_embed(blob), {"project": project, "summary": summary, "keywords": keywords or [], "rationale": rationale})
        return "stored"
    if action == "query":
        vec = _embed(summary)
        return vstore.query(vec)
    if action == "list":
        return vstore.list()
    raise ValueError("action must be add|query|list")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def make_agent(name: str, instructions: str, tools: List[Any] | None = None) -> AssistantAgent:
    memory = ListMemory(name=f"{name}_mem")
    model_client = OpenAIChatCompletionClient(model=MODEL_NAME, api_key=OPENAI_API_KEY)
    agent = AssistantAgent(
        name=name,
        system_message=instructions,
        model_client=model_client,
        memory=[memory],
    )
    for tool in tools or []:
        agent.register_tool(tool)
    return agent


financial_agent = make_agent(
    "FinancialHealthAgent",
    "Extract key financial metrics. Return bullet points or 'metric: value' pairs. Give a detailed assessment like an M&A Specialist of the company's financial situation including business model and outlook for the next 5 years.",
    tools=[_extract_pdf]
)

market_agent = make_agent(
    "MarketOpportunityAgent",
    "Analyse market size and competition. Create an executive summary of the opportunity. List the five biggest competitors and highlight key products.",
    tools=[_extract_pdf, web_search]
)

risk_agent = make_agent(
    "RiskAssessmentAgent",
    "Identify major financial and operational risks considering market, product and macroeconomic situation.",
    tools=[_extract_pdf, web_search]
)

alveus_fit_agent = make_agent(
    "AlveusFitAgent",
    "Bewerte den Alveus-Fit: Distanz des Targets zu München, mindestens 50 Mitarbeiter, vorhandene zweite Führungsebene und deren Gestaltung sowie ein EBIT von mindestens 1,5 Mio Euro. Gib stichpunktartige Einschätzungen zurück.",
    tools=[_extract_pdf, web_search]
)

report_agent = make_agent(
    "ReportAgent",
    "Return a concise Markdown report containing at least the following sections: 'Executive Summary', 'Metrics', 'Decision', 'Alveus Fit', 'Risk', 'Rationale', 'Long Report'. Use bullet lists where appropriate.",
)

supervisor_agent = make_agent(
    "SupervisorAgent",
    "Return YES/NO or RETRY <Agent>:<reason>. Use the 'Past:' section, sourced from vector_memory, to judge whether current results improve on previous ones. You are the investment manager and responsible to create an actionable report for the investment committee.",
    tools=[vector_memory]
)

AGENT_MAP = {
    "FinancialHealthAgent": financial_agent,
    "MarketOpportunityAgent": market_agent,
    "RiskAssessmentAgent": risk_agent,
    "AlveusFitAgent": alveus_fit_agent,
}


@dataclass
class EvaluationResult:
    summary: str
    keywords: List[str]
    metrics: Dict[str, Any]
    decision: str
    rationale: str
    markdown: str


# ---------------------------------------------------------------------------
# Helper to write markdown
# ---------------------------------------------------------------------------

def _write_markdown(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def _extract_md_section(md: str, heading: str) -> str:
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


def _write_pdf(md_path: str, pdf_path: str) -> None:
    """Konvertiert eine Markdown-Datei in ein PDF und speichert es."""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    html_content = markdown2.markdown(md_content)
    pdfkit.from_string(html_content, pdf_path)


# ---------------------------------------------------------------------------
# Agent invocation helpers
# ---------------------------------------------------------------------------

async def _run_agent(agent: AssistantAgent, text: str) -> str:
    user = UserProxyAgent("orchestrator", human_input_mode="NEVER")
    await user.initiate_chat(agent, message=text)
    return user.last_message(agent).content


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def evaluate(pdf: str, project: str) -> EvaluationResult:
    text = _extract_pdf(pdf)
    tasks = {k: asyncio.create_task(_run_agent(a, text)) for k, a in AGENT_MAP.items()}
    await asyncio.gather(*tasks.values())
    results = {k: t.result() for k, t in tasks.items()}

    sup_input = (
        f"Financial:\n{results['FinancialHealthAgent']}\n\n"
        f"Market:\n{results['MarketOpportunityAgent']}\n\n"
        f"Risk:\n{results['RiskAssessmentAgent']}\n\n"
        f"AlveusFit:\n{results['AlveusFitAgent']}\n\n"
        f"Past:{vector_memory('query', summary=' '.join(results.values()))}"
    )
    sup_text = await _run_agent(supervisor_agent, sup_input)
    decision, *rationale_parts = sup_text.split("\n", 1)
    rationale = rationale_parts[0] if rationale_parts else ""

    report_payload = (
        f"Financial:\n{results['FinancialHealthAgent']}\n\n"
        f"Market:\n{results['MarketOpportunityAgent']}\n\n"
        f"Risk:\n{results['RiskAssessmentAgent']}\n\n"
        f"AlveusFit:\n{results['AlveusFitAgent']}\n\n"
        f"Decision:\n{decision}\n\n"
        f"Rationale:\n{rationale}"
    )
    markdown = await _run_agent(report_agent, report_payload)

    summary = _extract_md_section(markdown, "Summary")
    keywords = _extract_md_list(markdown, "Keywords")
    metrics = _extract_md_kv(markdown, "Metrics")

    vector_memory("add", project=project, summary=summary, keywords=keywords, rationale=rationale)
    md_path = os.path.join(REPORT_DIR, f"{project}.md")
    _write_markdown(md_path, markdown)

    # PDF-Export
    pdf_path = os.path.join(REPORT_DIR, f"{project}.pdf")
    _write_pdf(md_path, pdf_path)

    return EvaluationResult(
        summary=summary,
        keywords=keywords,
        metrics=metrics,
        decision=decision,
        rationale=rationale,
        markdown=md_path,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: python autogen_agents.py <pdf_path> <project_name>")

    async def main():
        result = await evaluate(sys.argv[1], sys.argv[2])
        print("Report saved to", result.markdown)

    asyncio.run(main())

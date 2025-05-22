import os
import json
from typing import Any, Dict, List

import numpy as np
import requests
import openai
import faiss
from openai_agents import Agent, Tool, LLM  # use official openai-agents package

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or _raise_env_error("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") or _raise_env_error("SERPAPI_API_KEY")
openai.api_key = OPENAI_API_KEY

llm = LLM(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
session = requests.Session()

# Paths for persistence
DATA_DIR = "/mnt/data"
INDEX_PATH = os.path.join(DATA_DIR, "vstore.index")
META_PATH = os.path.join(DATA_DIR, "vstore_meta.json")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
EMBEDDING_DIM = 1536
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Initialize FAISS index and metadata
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []

# Helpers

def _raise_env_error(var: str):
    raise EnvironmentError(f"Required environment variable {var} not set.")

def embed_text(text: str) -> List[float]:
    resp = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return resp['data'][0]['embedding']

# Tools

def parse_pdf(file_path: str) -> str:
    uploaded = openai.File.create(file=open(file_path, 'rb'), purpose='analysis')
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": [
                {"type": "input_file", "file_id": uploaded.id},
                {"type": "input_text", "text": "Bitte extrahiere den gesamten Text aus dem PDF-Dokument."}
            ]}
        ]
    )
    return response.choices[0].message.content

parse_pdf_tool = Tool(name="parse_pdf", description="Use OpenAI file upload for PDF parsing", func=parse_pdf)

def web_search(query: str) -> str:
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": "google", "num": 5}
    resp = session.get("https://serpapi.com/search", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return "\n".join(
        f"- {i.get('title','')}: {i.get('snippet','')} ({i.get('link','')})"
        for i in data.get("organic_results", [])
    )

web_search_tool = Tool(name="web_search", description="Web search", func=web_search)

def vector_memory(action: str, project: str = "", summary: str = "", keywords: List[str] = None, rationale: str = "") -> Any:
    global index, metadata
    if action == "add":
        text = f"{project} {summary} {keywords} {rationale}"
        vec = embed_text(text)
        index.add(np.array([vec], dtype='float32'))
        metadata.append({"project": project, "summary": summary, "keywords": keywords or [], "rationale": rationale})
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, 'w') as f:
            json.dump(metadata, f)
        return f"Vector entry added for project '{project}'"
    if action == "query":
        vec = embed_text(summary)
        D, I = index.search(np.array([vec], dtype='float32'), 3)
        return [metadata[i] for i in I[0] if i < len(metadata)]
    if action == "list":
        return metadata

vector_memory_tool = Tool(name="vector_memory", description="Manage vector memory", func=vector_memory)

# Agent Definitions
financial_agent = Agent(
    name="FinancialHealthAgent",
    llm=llm,
    tools=[parse_pdf_tool],
    system_message="Extract financial metrics such as revenue, EBITDA, profit margins, growth rates, and balance sheet highlights."
)
market_agent = Agent(
    name="MarketOpportunityAgent",
    llm=llm,
    tools=[parse_pdf_tool, web_search_tool],
    system_message="Analyze market size, growth trends, competition, and barriers; use web_search for validation."
)
risk_agent = Agent(
    name="RiskAssessmentAgent",
    llm=llm,
    tools=[parse_pdf_tool, web_search_tool],
    system_message="Identify and rate key risks (financial, operational, regulatory, technological)."
)
report_agent = Agent(
    name="ReportAgent",
    llm=llm,
    tools=[],
    system_message=(
        "Generate a JSON report including fields:"
        " 'summary' (short overview),"
        " 'keywords' (list of key terms),"  
        " 'metrics' (dict of extracted financial numbers: revenue, EBITDA, growth rate, etc.)."
    )
)
supervisor_agent = Agent(
    name="SupervisorAgent",
    llm=llm,
    tools=[vector_memory_tool],
    system_message=(
        "Review report, compare with past via vector_memory.query, then either 'YES', 'NO', or 'RETRY <AgentName>: <feedback>'."
    )
)

# HTML Report Writer

def write_html_report(output_path: str, project: str, summary: str, keywords: List[str], metrics: Dict[str, Any], decision: str, rationale: str) -> None:
    metrics_html = ''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">    
    <head><meta charset=\"UTF-8\"><title>Report: {project}</title></head>
    <body>
      <h1>Investment Report: {project}</h1>
      <h2>Summary</h2>
      <p>{summary}</p>
      <h2>Keywords</h2>
      <ul>{''.join(f'<li>{kw}</li>' for kw in keywords)}</ul>
      <h2>Key Metrics</h2>
      <table border=\"1\"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metrics_html}</tbody></table>
      <h2>Decision</h2>
      <p>{decision}</p>
      <h2>Rationale</h2>
      <p>{rationale}</p>
    </body>
    </html>
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

# Orchestrierung with retry logic

def evaluate_investment(pdf_path: str, project_name: str) -> Dict[str, Any]:
    text = parse_pdf(pdf_path)
    sub_inputs = {"FinancialHealthAgent": text, "MarketOpportunityAgent": text, "RiskAssessmentAgent": text}
    sub_agents = {"FinancialHealthAgent": financial_agent, "MarketOpportunityAgent": market_agent, "RiskAssessmentAgent": risk_agent}
    sub_results: Dict[str, Any] = {}

    # initial run
    for name, agent in sub_agents.items():
        sub_results[name] = agent.run(input=sub_inputs[name])

    while True:
        # compile report
        report_out = report_agent.run(
            input=f"Financial:\n{sub_results['FinancialHealthAgent']}\n\n"
                  f"Market:\n{sub_results['MarketOpportunityAgent']}\n\n"
                  f"Risk:\n{sub_results['RiskAssessmentAgent']}"
        )
        data = json.loads(report_out)
        summary = data.get("summary", "")
        keywords = data.get("keywords", [])
        metrics = data.get("metrics", {})

        # query memory
        past = vector_memory(action="query", summary=summary)
        supervisor_out = supervisor_agent.run(
            input=f"Summary:\n{summary}\nKeywords:{keywords}\nMetrics:{metrics}\nSimilar:{past}"
        )

        if supervisor_out.startswith("RETRY"):
            _, rest = supervisor_out.split(" ", 1)
            agent_name, feedback = rest.split(":", 1)
            agent_name = agent_name.strip()
            feedback = feedback.strip()
            if agent_name in sub_agents:
                new_input = f"{feedback}\nOriginal content:\n{sub_inputs[agent_name]}"
                sub_results[agent_name] = sub_agents[agent_name].run(input=new_input)
                continue
            else:
                break
        else:
            parts = supervisor_out.split("\n", 1)
            decision = parts[0]
            rationale = parts[1] if len(parts) > 1 else ""
            break

    # store in vector memory
    vector_memory(action="add", project=project_name, summary=summary, keywords=keywords, rationale=rationale)
    html_path = os.path.join(REPORT_DIR, f"{project_name}.html")
    write_html_report(html_path, project_name, summary, keywords, metrics, decision, rationale)

    return {"financial": sub_results['FinancialHealthAgent'],
            "market": sub_results['MarketOpportunityAgent'],
            "risk": sub_results['RiskAssessmentAgent'],
            "summary": summary, "keywords": keywords,
            "metrics": metrics, "decision": decision, "rationale": rationale,
            "html_report": html_path}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python investment_agents.py <pdf_path> <project_name>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    project_name = sys.argv[2]
    res = evaluate_investment(pdf_path, project_name)
    print(f"HTML report generated at: {res['html_report']}")

# InvestmentAgent

This project demonstrates an evaluation pipeline built with the OpenAI Agents SDK. Multiple agents extract information from a PDF, perform brief web searches and generate a decision report.

## Installation

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Environment Variables

The script requires the following variables:

- `OPENAI_API_KEY` – OpenAI API key
- `SERPAPI_API_KEY` – SerpAPI key used for Google search
- `AGENT_MODEL` – optional model name, defaults to `gpt-4o`
- `AGENT_DATA_DIR` – optional directory for persistent data (defaults to `./data`)

## Usage

Run the evaluation with:

```bash
python investment_agents.py ./KPIs.pdf test
```

The PDF path is the first argument and `test` is the project name used to store
the results. Generated HTML reports are written to `data/reports/<project>.html`.

HTML output is rendered via a small Jinja2 template to ensure proper escaping and maintainability.

The embedding memory is handled through `vector_store.VectorStore`, which currently wraps a FAISS index. This abstraction allows alternative backends to be plugged in easily.

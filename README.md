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

The PDF path is the first
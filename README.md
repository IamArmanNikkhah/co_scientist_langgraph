# AI Co-Scientist (LangGraph)

## Environment & API Keys

This project uses environment variables for configuration. For local development, create a `.env` file in the project root with the following keys as needed:

```
# OpenAI
OPENAI_API_KEY=sk-...

# SerpAPI (Google Scholar)
SERPAPI_API_KEY=...

# Optional: override HTTP user agent for outbound requests
HTTP_USER_AGENT=co-scientist/0.1 (+https://yourdomain.example)
```

The package automatically loads `.env` on import (via `python-dotenv`). In production, prefer setting environment variables through your CI/secret manager and avoid committing `.env` files.

## Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```
python -m co_scientist_langgraph --goal "..." \
  --lit-chronology-file path/to/chronology.txt \
  --max-iterations 3
```

To supply raw literature content (used by the observation aggregator):

```
python -m co_scientist_langgraph --goal "..." --literature-file path/to/lit.txt
```

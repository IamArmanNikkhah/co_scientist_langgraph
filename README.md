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

## Using the portable run script (macOS Terminal)

The project includes a simple script that sets everything up and runs the app for you.

1) Open Terminal and go to the project folder:

```
cd /path/to/co_scientist_langgraph
```

2) Make sure the script is executable (you only need to do this once):

```
chmod +x run.sh
```

3) Set up the environment (creates a local `.venv` and installs dependencies):

```
./run.sh setup
```

4) Start the interactive REPL (recommended to explore):

```
./run.sh interactive --goal "Your research goal here" --max-iterations 5
```

- In the REPL, type `help` to see commands (e.g., `next`, `approve`, `upload-lit <path>`, `auto 3`, `checkpoint-dir ./checkpoints`).
- Press `Ctrl+D` or type `exit` to quit.

5) Run in non-interactive (batch) mode:

```
./run.sh batch --goal "Plant growth vs sunlight" --max-iterations 3
```

### Tips (macOS)

- If you use zsh or bash, the commands above work as-is.
- If you see “permission denied,” run `chmod +x run.sh` and try again.
- If Python 3 is not found, install it (e.g., via Homebrew: `brew install python`) or run with a specific interpreter:

```
PYTHON_BIN=python3.11 ./run.sh setup
```

### Environment variables (.env)

- Copy `ENV.sample` to `.env` and fill in keys if you need web tools (Perplexity, Scholar) or custom settings. The app auto-loads `.env`.
- Do not commit your `.env` file.

"""
co_scientist_langgraph
Modular, production-grade LangGraph implementation of the "AI co-scientist" system.

Package layout:
- state: Typed shared graph state and common types
- tools: Utility tools implemented with the official @tool API
- prompts: Prompt builders for each agent
- nodes: Agent node factories
- app: Graph builder (nodes + conditional routing)
- cli: Command-line entry point
"""

from .state import GraphState, TaskName  # re-export for convenience

# Load .env at import time for local development (no-op in prod if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

__all__ = [
    "GraphState",
    "TaskName",
]



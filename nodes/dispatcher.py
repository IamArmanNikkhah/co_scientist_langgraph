from __future__ import annotations

from typing import Dict, List

from langchain_openai import ChatOpenAI

from ..state import GraphState
from ..workers import AsyncTaskQueue


def make_dispatcher_node(supervisor_llm: ChatOpenAI, worker_llm: ChatOpenAI | None = None, worker_count: int = 2):
    async def dispatcher(state: GraphState) -> GraphState:
        """
        Execute the Supervisor-selected task via a small worker pool.

        This node does not implement the task logic itself; it orchestrates by
        scheduling the corresponding graph nodes as async jobs through the queue.
        Current implementation runs the selected task once (quantity parameter
        remains meaningful for generation nodes themselves).
        """
        s = state
        decision = s.get("decision", {}) or {}
        next_task = str(decision.get("next_task", "")).strip()
        params: Dict = decision.get("parameters", {}) or {}

        # Build jobs that call the downstream nodes indirectly by returning deltas
        # The LangGraph engine will apply returned partial state updates
        queue = AsyncTaskQueue(worker_count=worker_count)

        async def run_selected() -> Dict:
            # Dispatcher simply signals to proceed; actual transitions are defined in the graph
            # We surface parameters so downstream nodes have them in state
            merged_params = {**(s.get("parameters") or {}), **params}
            rm = dict(s.get("run_metadata") or {})
            rm["worker_pool"] = {"workers": worker_count, "jobs_submitted": 1}
            rm["last_scheduled_task"] = next_task
            return {"parameters": merged_params, "run_metadata": rm}

        # For now we submit a single job for the chosen task; future extension could batch
        if next_task:
            queue.submit(run_selected)
            results: List[Dict] = await queue.run_until_empty()
            # Merge the first result (single job)
            merged: Dict = {}
            for r in results:
                merged.update(r or {})
            return merged
        return {}

    return dispatcher



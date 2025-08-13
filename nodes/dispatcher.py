from __future__ import annotations

from typing import Dict, List, Any

from langchain_openai import ChatOpenAI

from ..state import GraphState
from ..workers import AsyncTaskQueue
from .reflection import make_reflection_node
from .ranking import make_ranking_node
from .generation import make_generation_node


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

        def chunk_list(items: List[str], size: int) -> List[List[str]]:
            size = max(1, int(size))
            return [items[i : i + size] for i in range(0, len(items), size)]

        async def run_reflection_shards() -> Dict[str, Any]:
            reflect = make_reflection_node(worker_llm or supervisor_llm)
            target_ids: List[str] = list(params.get("priority_hypothesis_ids", []))
            if not target_ids:
                # derive from state if not provided
                target_ids = [h.get("id") for h in s.get("hypotheses", []) if not h.get("is_reviewed")]
            shard_size = int(params.get("shard_size", 3))
            shards = chunk_list(target_ids, shard_size) or [[]]

            # Build one job per shard
            for shard in shards:
                async def job(sh=shard):
                    sub_state: GraphState = dict(s)
                    sub_params = {**(sub_state.get("parameters") or {}), **params, "priority_hypothesis_ids": sh}
                    sub_state["parameters"] = sub_params
                    delta = await reflect(sub_state)  # returns partial state
                    return delta or {}
                queue.submit(job)

            results = await queue.run_until_empty()
            # Merge partials deterministically
            merged: Dict[str, Any] = {"errors": list(s.get("errors") or [])}
            # Merge hypotheses by id
            by_id: Dict[str, Dict[str, Any]] = {h.get("id"): dict(h) for h in (s.get("hypotheses") or []) if h.get("id")}
            newly_reviewed: List[str] = []
            total_reviews_added = 0
            for r in results:
                for k, v in (r or {}).items():
                    if k == "hypotheses":
                        for h in (v or []):
                            hid = h.get("id")
                            if not hid:
                                continue
                            by_id[hid] = dict(h)
                            if h.get("is_reviewed"):
                                newly_reviewed.append(hid)
                    elif k == "run_metadata":
                        # accumulate reflection counts if present
                        total_reviews_added += int(v.get("last_reflection_count", 0))
                    elif k == "errors":
                        merged["errors"].extend(v or [])
                    else:
                        merged[k] = v
            merged["hypotheses"] = list(by_id.values())
            rm = dict(s.get("run_metadata") or {})
            rm["worker_pool"] = {"workers": worker_count, "jobs_submitted": len(results)}
            rm["last_scheduled_task"] = "reflect"
            rm["last_reflection_count"] = total_reviews_added or rm.get("last_reflection_count", 0)
            if newly_reviewed:
                prev = set(rm.get("newly_reviewed_hypotheses", []))
                rm["newly_reviewed_hypotheses"] = list(prev | set(newly_reviewed))
            merged["run_metadata"] = rm
            return merged

        async def run_ranking_shards() -> Dict[str, Any]:
            rank_fn = make_ranking_node(worker_llm or supervisor_llm)
            all_new: List[str] = list(params.get("newly_reviewed_ids", []))
            if not all_new:
                all_new = [h.get("id") for h in s.get("hypotheses", []) if h.get("is_reviewed") and not h.get("is_ranked")]
            shard_size = int(params.get("shard_size", 2))
            shards = chunk_list(all_new, shard_size) or [[]]

            for shard in shards:
                async def job(sh=shard):
                    sub_state: GraphState = dict(s)
                    # ranking node expects ids under decision.parameters normally; it also discovers automatically
                    # To guide deterministically, we pass via decision.parameters for this sub-call
                    sub_state["decision"] = {
                        **(s.get("decision") or {}),
                        "parameters": {**params, "newly_reviewed_ids": sh},
                    }
                    delta = await rank_fn(sub_state)
                    return delta or {}
                queue.submit(job)

            results = await queue.run_until_empty()
            merged: Dict[str, Any] = {"errors": list(s.get("errors") or [])}
            by_id: Dict[str, Dict[str, Any]] = {h.get("id"): dict(h) for h in (s.get("hypotheses") or []) if h.get("id")}
            for r in results:
                for k, v in (r or {}).items():
                    if k == "hypotheses":
                        for h in (v or []):
                            hid = h.get("id")
                            if not hid:
                                continue
                            by_id[hid] = dict(h)
                    elif k == "errors":
                        merged["errors"].extend(v or [])
                    else:
                        merged[k] = v
            merged["hypotheses"] = sorted(list(by_id.values()), key=lambda x: x.get("elo_rating", 1200), reverse=True)
            rm = dict(s.get("run_metadata") or {})
            rm["worker_pool"] = {"workers": worker_count, "jobs_submitted": len(results)}
            rm["last_scheduled_task"] = "rank"
            merged["run_metadata"] = rm
            return merged

        if next_task == "reflect":
            return await run_reflection_shards()
        if next_task == "rank":
            return await run_ranking_shards()

        # For other tasks, submit a single parameter propagation job
        if next_task:
            queue.submit(run_selected)
            results: List[Dict] = await queue.run_until_empty()
            merged: Dict = {}
            for r in results:
                merged.update(r or {})
            return merged
        return {}

    return dispatcher



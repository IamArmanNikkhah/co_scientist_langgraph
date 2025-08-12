from __future__ import annotations

import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from ..state import GraphState, safe_append_error
from ..tools import extract_json_from_text


def make_bootstrap_node(llm: ChatOpenAI):
    async def bootstrap_init(state: GraphState) -> GraphState:
        """Initialize state from natural language research_goal if config is missing."""
        if state.get("research_plan_config"):
            return {}

        goal = (state.get("research_goal") or "").strip()
        if not goal:
            safe_append_error(state, "No research_goal provided; cannot initialize.")
            return {}

        prompt = f"""
You parse a natural language research_goal into JSON with keys: preferences (string), attributes (list), constraints (list), evaluation_criteria (list).

Return ONLY the JSON object.

research_goal: "{goal}"
"""
        try:
            resp = await llm.ainvoke(prompt)
            content = getattr(resp, "content", str(resp))
            json_str = extract_json_from_text.invoke({"text": content}) or content
            parsed = json.loads(json_str)
        except Exception as e:
            safe_append_error(state, f"Initializer parsing failed: {e}")
            parsed = {"preferences": "", "attributes": [], "constraints": [], "evaluation_criteria": []}

        init: Dict[str, Any] = {
            "research_plan_config": parsed,
            "hypotheses": [],
            "tournament_state": {},
            "proximity_graph": {},
            "meta_review": {},
            "meta_review_critique": "",
            "scientific_observations": None,
            "decision": {},
            "next_task": None,
            "parameters": {},
            "errors": state.get("errors", []),
            "run_metadata": {
                "iteration_count": 0,
                "last_task": None,
                "total_hypotheses": 0,
                "top_elo_score": 1000,
                "iterations_since_improvement": 0,
                "unreviewed_hypotheses": 0,
                "newly_reviewed_hypotheses": 0,
                "last_reflection_count": 0,
                "previous_review_count": 0,
                "decision_history": [],
            },
        }
        return init

    return bootstrap_init



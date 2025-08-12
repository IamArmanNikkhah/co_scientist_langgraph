from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


TaskName = Literal["generate", "reflect", "rank", "evolve", "meta_review", "terminate"]


class GraphState(TypedDict, total=False):
    """
    Shared state passed between nodes (agents). Mirrors the LangFlow system and the paper requirements.

    Fields:
    - research_goal: The high-level research objective.
    - research_plan_config: Parsed configuration derived from the research goal.
    - hypotheses: List of hypothesis dicts with fields: id, content, elo_rating, reviews, is_reviewed, is_ranked.
    - meta_review: Structured meta-review content (dict) or raw content string.
    - meta_review_critique: String form of meta-review insights.
    - scientific_observations: Formatted observations text for Reflection.
    - literature_content: Optional raw literature text provided by user.
    - decision: Latest Supervisor decision payload.
    - next_task: Next task selected by Supervisor.
    - parameters: Parameters for the selected agent (from Supervisor).
    - run_metadata: Operational metadata (iteration_count, last_task, etc.).
    - errors: Collected error messages for debugging and transparency.
    """

    research_goal: str
    research_plan_config: Dict[str, Any]
    hypotheses: List[Dict[str, Any]]
    meta_review: Any
    meta_review_critique: Optional[str]
    scientific_observations: Optional[str]
    literature_content: Optional[str]
    decision: Dict[str, Any]
    next_task: Optional[TaskName]
    parameters: Dict[str, Any]
    run_metadata: Dict[str, Any]
    errors: List[str]


def safe_append_error(state: GraphState, message: str) -> None:
    """Append an error message to the state's errors list safely."""
    errs = state.get("errors") or []
    errs.append(message)
    state["errors"] = errs



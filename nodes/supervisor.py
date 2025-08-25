from __future__ import annotations

import json
import re
import statistics
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..prompts.supervisor import build_supervisor_prompt
from ..state import GraphState, safe_append_error
from ..tools import extract_json_from_text


def _enhanced_metrics(hypotheses: List[Dict[str, Any]], stats: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    def extract_title(content: str) -> str:
        m = re.search(r"### Proposed Hypothesis:\s*(.*?)\n", content or "", re.IGNORECASE)
        return m.group(1) if m else "Untitled Hypothesis"

    def tokenize(title: str) -> List[str]:
        return [t for t in re.findall(r"[A-Za-z0-9_]+", title.lower()) if len(t) > 2]

    tokens = [set(tokenize(extract_title(h.get("content", "")))) for h in hypotheses if h.get("content")]
    diversity = 0.0
    if len(tokens) >= 2:
        dists: List[float] = []
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                a, b = tokens[i], tokens[j]
                union = len(a | b)
                inter = len(a & b)
                dists.append(0.0 if union == 0 else 1.0 - inter / union)
        diversity = float(statistics.mean(dists)) if dists else 0.0

    recent = history[-5:]
    series: List[float] = []
    for rec in recent:
        if rec.get("post_top_elo") is not None:
            series.append(float(rec["post_top_elo"]))
        elif rec.get("pre_top_elo") is not None:
            series.append(float(rec["pre_top_elo"]))
    momentum = 0.0
    if len(series) >= 2:
        delta = series[-1] - series[0]
        steps = max(1, len(series) - 1)
        momentum = max(-1.0, min(1.0, (delta / steps) / 100.0))

    stagnation = 0.5 * (min(10, stats.get("iterations_since_improvement", 0)) / 10.0) + 0.25 * (
        1.0 - max(-1.0, min(1.0, momentum))
    )
    stagnation = float(max(0.0, min(1.0, stagnation)))

    cand_signal = min(1.0, stats.get("evolution_candidates", 0) / 3.0)
    momentum01 = (momentum + 1.0) / 2.0
    breakthrough = float(max(0.0, min(1.0, 0.4 * cand_signal + 0.35 * diversity + 0.25 * momentum01)))

    phase = (
        "breakthrough"
        if stats.get("top_elo_score", 0) > 1500 and stats.get("iterations_since_improvement", 0) <= 1
        else ("exploratory" if stats.get("iteration_count", 0) < 8 else "convergence")
    )

    return {
        "research_phase": phase,
        "discovery_momentum": momentum,
        "hypothesis_diversity": diversity,
        "workflow_efficiency": float(1.0 - min(10, stats.get("iterations_since_improvement", 0)) / 10.0),
        "stagnation_risk": stagnation,
        "breakthrough_probability": breakthrough,
        "bottleneck_indicators": [
            *( ["pending_reviews"] if stats.get("unreviewed_hypotheses", 0) > 0 else [] ),
            *( ["pending_ranking"] if stats.get("newly_reviewed_hypotheses", 0) > 0 else [] ),
        ],
    }


def _validate_precedence(next_task: str, stats: Dict[str, Any]) -> Optional[str]:
    """
    Check precedence rules and return suggested standard step if violation detected.
    Returns None if no violation, or the suggested standard step if violation found.
    """
    unreviewed = stats.get("unreviewed_hypotheses", 0)
    newly_reviewed = stats.get("newly_reviewed_hypotheses", 0)
    new_reviews_since_last = stats.get("new_reviews_since_last", 0)

    if next_task == "meta_review" and unreviewed > 0:
        return "reflect"
    if next_task == "meta_review" and newly_reviewed > 0:
        return "rank"
    if next_task == "meta_review" and (unreviewed > 0 or newly_reviewed > 0 or new_reviews_since_last < 5):
        if unreviewed > 0:
            return "reflect"
        elif newly_reviewed > 0:
            return "rank"
        else:
            return "meta_review"  # Allow meta_review if just insufficient reviews
    if unreviewed > 0 and next_task not in ["reflect", "terminate"]:
        return "reflect"
    if newly_reviewed > 0 and next_task not in ["rank", "terminate"]:
        return "rank"
    return None


def make_supervisor_node(llm: ChatOpenAI, max_iterations: int):
    async def supervisor_node(state: GraphState) -> GraphState:
        s = state
        goal = s.get("research_goal", "")
        config = s.get("research_plan_config", {})
        preferences = config.get("preferences", "")
        hypotheses = s.get("hypotheses", [])
        meta_review_data = s.get("meta_review", {}) or {}
        meta_review_content = meta_review_data.get("structured_content") or s.get("meta_review_critique", "") or ""

        run_meta = s.get("run_metadata", {}) or {}
        iteration = int(run_meta.get("iteration_count", 0))

        # Termination guard by iteration cap
        if iteration >= max_iterations:
            decision_payload = {
                "next_task": "terminate",
                "parameters": {"reason": "iteration_limit", "final_hypothesis_id": (hypotheses[0]["id"] if hypotheses else "")},
                "rationale": f"Reached iteration cap {max_iterations}",
                "strategic_context": {"decision_confidence": 1.0, "predicted_impact": "End run", "risk_assessment": "Low"},
            }
            return {
                "decision": decision_payload,
                "next_task": "terminate",
                "parameters": decision_payload["parameters"],
                "run_metadata": {**run_meta, "last_task": "terminate"},
            }

        # Stats
        total_reviews = sum(len(h.get("reviews", [])) for h in hypotheses)
        previous_review_count = int(run_meta.get("previous_review_count", 0))
        new_reviews_count = max(0, total_reviews - previous_review_count)
        evolution_candidates = [h for h in hypotheses if h.get("elo_rating", 1200) > 1300 and h.get("is_reviewed")]

        stats = {
            "iteration_count": iteration,
            "total_hypotheses": len(hypotheses),
            "unreviewed_hypotheses": sum(1 for h in hypotheses if not h.get("is_reviewed")),
            "newly_reviewed_hypotheses": sum(1 for h in hypotheses if h.get("is_reviewed") and not h.get("is_ranked")),
            "top_elo_score": max([h.get("elo_rating", 1200) for h in hypotheses], default=1200),
            "last_task": run_meta.get("last_task"),
            "iterations_since_improvement": int(run_meta.get("iterations_since_improvement", 0)),
            "total_reviews": total_reviews,
            "new_reviews_since_last": new_reviews_count,
            "evolution_candidates": len(evolution_candidates),
            "has_meta_review": bool(meta_review_content),
            "last_literature_iteration": run_meta.get("last_literature_iteration"),
        }

        # Decision history outcome fill
        history: List[Dict[str, Any]] = list(run_meta.get("decision_history", []))
        if history:
            last = history[-1]
            if last.get("post_top_elo") is None:
                last["post_top_elo"] = stats["top_elo_score"]

        enhanced = _enhanced_metrics(hypotheses, stats, history)

        summary = [
            {
                "id": h.get("id"),
                "title": re.search(r"### Proposed Hypothesis:\s*(.*?)\n", h.get("content", "") or "", re.IGNORECASE).group(1)
                if re.search(r"### Proposed Hypothesis:\s*(.*?)\n", h.get("content", "") or "", re.IGNORECASE)
                else "Untitled",
                "elo_rating": h.get("elo_rating", 1200),
                "is_reviewed": h.get("is_reviewed", False),
                "is_ranked": h.get("is_ranked", False),
            }
            for h in hypotheses
        ]

        prompt = build_supervisor_prompt(stats, enhanced, meta_review_content, goal, preferences, summary)

        decision: Dict[str, Any] = {}
        for attempt in range(2):
            try:
                resp = await llm.ainvoke(prompt)
                text = getattr(resp, "content", str(resp))
                json_str = extract_json_from_text.invoke({"text": text})
                if not json_str:
                    raise ValueError("No JSON found in supervisor response.")
                decision = json.loads(json_str)
                next_task = decision.get("next_task")
                parameters = decision.get("parameters", {})
                if not isinstance(parameters, dict) or not parameters:
                    raise ValueError("parameters missing/empty.")
                
                # Check for precedence violations and add warning if detected
                suggested_step = _validate_precedence(str(next_task), stats)
                if suggested_step:
                    warning_msg = f"Warning: non-standard step detected. The standard step is: {suggested_step}"
                    safe_append_error(s, warning_msg)
                
                break
            except Exception as e:
                if attempt == 0:
                    prompt = f"Previous response invalid ({e}). Return ONLY a valid JSON per schema.\n\n{prompt}"
                else:
                    raise

        next_task = decision.get("next_task", "terminate")
        parameters = decision.get("parameters", {})
        rationale = decision.get("rationale", "")

        # Supervisor-side defaulting of generation mode when LLM omitted it
        if str(next_task) == "generate":
            gen_mode = str(parameters.get("generation_mode", "")).strip().lower()
            if not gen_mode:
                try:
                    # Heuristic aligned with prompt guidance
                    use_debate = (
                        enhanced.get("research_phase") == "exploratory"
                        or float(enhanced.get("hypothesis_diversity", 0.0)) < 0.35
                        or float(enhanced.get("stagnation_risk", 0.0)) >= 0.6
                    )
                except Exception:
                    use_debate = False
                parameters["generation_mode"] = "debate" if use_debate else "standard"
                if use_debate and "debate_max_turns" not in parameters:
                    parameters["debate_max_turns"] = 6

        prev_top = int(run_meta.get("top_elo_score", 1200))
        its_since = int(run_meta.get("iterations_since_improvement", 0))
        its_since = 0 if stats["top_elo_score"] > prev_top else its_since + 1
        history.append({"iteration": iteration, "action": next_task, "pre_top_elo": stats["top_elo_score"], "post_top_elo": None})

        new_run_meta = {
            **run_meta,
            "iteration_count": iteration + 1,
            "last_task": next_task,
            "total_hypotheses": stats["total_hypotheses"],
            "top_elo_score": stats["top_elo_score"],
            "iterations_since_improvement": its_since,
            "unreviewed_hypotheses": stats["unreviewed_hypotheses"],
            "newly_reviewed_hypotheses": stats["newly_reviewed_hypotheses"],
            "previous_review_count": stats["total_reviews"],
            "evolution_candidates": stats["evolution_candidates"],
            "meta_review_available": stats["has_meta_review"],
            "decision_history": history,
            "last_literature_iteration": iteration if next_task == "literature" else run_meta.get("last_literature_iteration"),
        }

        decision_payload = {
            "next_task": next_task,
            "parameters": parameters,
            "rationale": rationale,
            "state": s,
            "iteration": new_run_meta["iteration_count"],
            "statistics": stats,
            "enhanced_statistics": enhanced,
            "research_goal": goal,
            "strategic_context": decision.get("strategic_context", {}),
        }

        return {
            "decision": decision_payload,
            "next_task": next_task,
            "parameters": parameters,
            "run_metadata": new_run_meta,
        }

    return supervisor_node



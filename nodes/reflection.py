from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from ..prompts.reflection import build_reflection_prompt
from ..state import GraphState, safe_append_error
from ..tools import extract_json_from_text


def make_reflection_node(llm: ChatOpenAI):
    async def reflection_node(state: GraphState) -> GraphState:
        s = state
        decision = s.get("decision", {}) or {}
        params = decision.get("parameters", {}) or {}
        strategic_context = decision.get("rationale", "Evaluate hypotheses to advance the research goal.")
        priority_ids = params.get("priority_hypothesis_ids", [])
        review_depth = params.get("review_depth", "standard")
        goal = s.get("research_goal", "")
        observations = s.get("scientific_observations", "") or ""

        hypotheses = s.get("hypotheses", [])
        unreviewed = [h for h in hypotheses if not h.get("is_reviewed", False)]
        targets = [h for h in unreviewed if h.get("id") in priority_ids] if priority_ids else unreviewed
        if not targets:
            return {}

        # Few-shot calibration examples: best & worst reviewed
        reviewed = [h for h in hypotheses if h.get("is_reviewed")]
        good_example = bad_example = None
        if len(reviewed) >= 2:
            reviewed.sort(key=lambda h: h.get("elo_rating", 1200))
            worst, best = reviewed[0], reviewed[-1]

            def fmt(h: Dict[str, Any], default_cls: str) -> str:
                last_review = (h.get("reviews") or [{}])[-1]
                pa = last_review.get("paper_analysis", {})
                return (
                    f"Hypothesis: {h.get('content','')}\n"
                    f"Review Summary: {last_review.get('qualitative_feedback',{}).get('summary','')}\n"
                    f"Final Classification: {pa.get('classification', default_cls)}"
                )

            good_example, bad_example = fmt(best, "missing piece"), fmt(worst, "disproved")

        count = 0
        for h in targets:
            try:
                prompt = build_reflection_prompt(
                    hypothesis=h.get("content", ""),
                    goal=goal,
                    observations=observations,
                    review_depth=review_depth,
                    strategic_context=strategic_context,
                    good_example=good_example,
                    bad_example=bad_example,
                )
                resp = await llm.ainvoke(prompt)
                text = getattr(resp, "content", str(resp))
                json_str = extract_json_from_text.invoke({"text": text}) or text
                data = json.loads(json_str)
                # Build review object
                scores = data.get("scores", {})
                review_obj: Dict[str, Any] = {
                    "id": str(uuid.uuid4()),
                    "hypothesis_id": h.get("id"),
                    "reviewer_type": "reflection_agent",
                    "scores": {
                        "overall": scores.get("overall", 6),
                        "novelty": scores.get("novelty", 6),
                        "validity": scores.get("validity", 7),
                        "testability": scores.get("testability", 6),
                        "specificity": scores.get("specificity", 6),
                    },
                    "qualitative_feedback": {
                        "strengths": data.get("strengths", []),
                        "weaknesses": data.get("weaknesses", []),
                        "suggestions": data.get("suggestions", []),
                        "summary": data.get("full_analysis", "")[:280],
                    },
                    "paper_analysis": {
                        "classification": data.get("classification", "neutral"),
                        "full_analysis": data.get("full_analysis", ""),
                    },
                    "flags": {
                        "explains_observations": data.get("classification", "").lower() in [
                            "missing piece",
                            "already explained",
                        ],
                        "contradicts_known_facts": data.get("classification", "").lower() == "disproved",
                    },
                }
                h.setdefault("reviews", []).append(review_obj)
                h["is_reviewed"] = True
                # Elo adjustment
                adjust = {
                    "missing piece": 50,
                    "neutral": 0,
                    "already explained": -10,
                    "other explanations more likely": -25,
                    "disproved": -75,
                }
                h["elo_rating"] = h.get("elo_rating", 1200) + adjust.get(data.get("classification", "neutral"), 0)
                count += 1
            except Exception as e:
                safe_append_error(s, f"Reflection error for {h.get('id','?')}: {e}")
                continue

        s.setdefault("run_metadata", {})
        s["run_metadata"]["last_reflection_count"] = count
        s["run_metadata"]["newly_reviewed_hypotheses"] = [h.get("id") for h in targets]
        return {"hypotheses": hypotheses, "run_metadata": s["run_metadata"]}

    return reflection_node



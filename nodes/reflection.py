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
                    literature_chronology=s.get("articles_with_reasoning_text", "") or "",
                )
                resp = await llm.ainvoke(prompt)
                text = getattr(resp, "content", str(resp))
                json_str = extract_json_from_text.invoke({"text": text}) or text
                
                # Try to parse JSON, with fallback for malformed responses
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # Create fallback review from raw text
                    data = {
                        "full_analysis": text[:2000] if text else "Failed to generate review",
                        "classification": "neutral",
                        "scores": {"overall": 5, "novelty": 5, "validity": 5, "testability": 5, "specificity": 5},
                        "strengths": ["Review generated from fallback due to JSON parsing error"],
                        "weaknesses": ["Unable to parse structured review"],
                        "suggestions": ["Re-run review with clearer instructions"]
                    }
                # Build review object
                scores = data.get("scores", {})
                # Ensure null-safety for full_analysis field
                full_analysis = data.get("full_analysis") or ""
                if not isinstance(full_analysis, str):
                    full_analysis = str(full_analysis) if full_analysis is not None else ""
                
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
                        "strengths": data.get("strengths") or [],
                        "weaknesses": data.get("weaknesses") or [],
                        "suggestions": data.get("suggestions") or [],
                        "summary": full_analysis[:280] if full_analysis else "",
                    },
                    "paper_analysis": {
                        "classification": data.get("classification", "neutral"),
                        "full_analysis": full_analysis,
                    },
                    "flags": {
                        "explains_observations": data.get("classification", "").lower() in [
                            "missing piece",
                            "already explained",
                        ],
                        "contradicts_known_facts": data.get("classification", "").lower() == "disproved",
                    },
                }
                # Capture deep verification block if present
                dv = data.get("deep_verification")
                if isinstance(dv, dict):
                    review_obj["deep_verification"] = dv
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
                # Mark as reviewed even if review failed to prevent workflow blocking
                h["is_reviewed"] = True
                # Create minimal fallback review
                fallback_review = {
                    "id": str(uuid.uuid4()),
                    "hypothesis_id": h.get("id"),
                    "reviewer_type": "reflection_agent",
                    "scores": {"overall": 3, "novelty": 3, "validity": 3, "testability": 3, "specificity": 3},
                    "qualitative_feedback": {
                        "strengths": [],
                        "weaknesses": ["Review failed due to technical error"],
                        "suggestions": ["Retry review process"],
                        "summary": f"Review failed: {str(e)[:200]}",
                    },
                    "paper_analysis": {
                        "classification": "neutral",
                        "full_analysis": f"Review could not be completed due to error: {str(e)}",
                    },
                    "flags": {"explains_observations": False, "contradicts_known_facts": False},
                }
                h.setdefault("reviews", []).append(fallback_review)
                continue

        s.setdefault("run_metadata", {})
        s["run_metadata"]["last_reflection_count"] = count
        s["run_metadata"]["newly_reviewed_hypotheses"] = [h.get("id") for h in targets]
        return {"hypotheses": hypotheses, "run_metadata": s["run_metadata"]}

    return reflection_node



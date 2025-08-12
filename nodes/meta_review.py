from __future__ import annotations

import json
import statistics
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from ..prompts.meta_review import build_meta_review_prompt
from ..state import GraphState, safe_append_error


def make_meta_review_node(llm: ChatOpenAI):
    async def meta_review_node(state: GraphState) -> GraphState:
        s = state
        goal = s.get("research_goal", "")
        prefs = s.get("research_plan_config", {}).get("preferences", "")
        decision = s.get("decision", {}) or {}
        params = decision.get("parameters", {}) or {}
        focus = params.get("focus", "identify_patterns")

        hyps = s.get("hypotheses", [])
        elo_ratings = [h.get("elo_rating", 1200) for h in hyps]
        all_reviews = [rv for h in hyps for rv in h.get("reviews", [])]

        novelty = [r.get("scores", {}).get("novelty") for r in all_reviews if r.get("scores")]
        validity = [r.get("scores", {}).get("validity") for r in all_reviews if r.get("scores")]
        classifications = [r.get("paper_analysis", {}).get("classification", "neutral") for r in all_reviews if r.get("paper_analysis")]

        quant = {
            "hypothesis_count": len(hyps),
            "top_elo_score": max(elo_ratings) if elo_ratings else 1200,
            "elo_stdev": round(statistics.stdev(elo_ratings), 2) if len(elo_ratings) > 1 else 0,
            "average_novelty": round(statistics.mean([n for n in novelty if isinstance(n, (int, float))]), 2) if novelty else None,
            "average_validity": round(statistics.mean([v for v in validity if isinstance(v, (int, float))]), 2) if validity else None,
            "classification_distribution": {c: classifications.count(c) for c in set(classifications)} if classifications else {},
        }
        reviews_text = "\n\n---\n\n".join(
            [f"ID {h['id'][:8]} (ELO: {h.get('elo_rating',1200)}):\n" + json.dumps(rv, indent=2) for h in hyps for rv in h.get("reviews", [])]
        ) or "No reviews available."

        prompt = build_meta_review_prompt(goal, prefs, focus, quant, reviews_text)
        try:
            resp = await llm.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))
            m = __import__("re").search(r"\{.*\}", text, __import__("re").DOTALL)
            if not m:
                raise ValueError("Meta-review did not return JSON.")
            content = json.loads(m.group(0))
            s["meta_review"] = {"structured_content": content, "scope": params.get("scope", "full_history"), "focus": focus}
            s["meta_review_critique"] = json.dumps(content, indent=2)
            return {"meta_review": s["meta_review"], "meta_review_critique": s["meta_review_critique"]}
        except Exception as e:
            safe_append_error(s, f"Meta-review error: {e}")
            s["meta_review"] = {"error": str(e)}
            s["meta_review_critique"] = f"Meta-review failed: {e}"
            return {"meta_review": s["meta_review"], "meta_review_critique": s["meta_review_critique"]}

    return meta_review_node



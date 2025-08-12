from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from ..state import GraphState, safe_append_error


def make_evolution_node(llm: ChatOpenAI):
    async def evolution_node(state: GraphState) -> GraphState:
        s = state
        decision = s.get("decision", {}) or {}
        params = decision.get("parameters", {}) or {}
        goal = s.get("research_goal", "")
        strategy = params.get("strategy", "analogize")
        targets: List[str] = params.get("target_hypothesis_ids", [])
        prefs = s.get("research_plan_config", {}).get("preferences", "")
        hyps = s.get("hypotheses", [])
        if not hyps:
            return {}
        by_id = {h["id"]: h for h in hyps}

        new_contents: List[Dict[str, Any]] = []

        try:
            if strategy == "refine":
                if not targets:
                    raise ValueError("Refine requires a target hypothesis id.")
                t = by_id[targets[0]]
                wk_prompt = (
                    "You are a critical reviewer. Identify the single biggest weakness of the following hypothesis and its reviews.\n"
                    f"Hypothesis:\n{t.get('content','')}\nReviews:\n{json.dumps((t.get('reviews') or [{}])[0], indent=2)}\n"
                    "Respond with ONLY the weakness phrase."
                )
                wk_resp = await llm.ainvoke(wk_prompt)
                weakness = getattr(wk_resp, "content", str(wk_resp)).strip()
                prompt = (
                    f"You refine the hypothesis to address a specific weakness.\nGoal: {goal}\nPreferences: {prefs}\n"
                    f"Original:\n{t.get('content','')}\nWeakness: {weakness}\nReturn the refined hypothesis text."
                )
                resp = await llm.ainvoke(prompt)
                content = getattr(resp, "content", str(resp))
                new_contents.append({"content": content, "evolved_from": [t["id"]], "evolution_type": "refinement"})
            elif strategy == "combine":
                if len(targets) < 2:
                    raise ValueError("Combine requires at least two target ids.")
                h1, h2 = by_id[targets[0]], by_id[targets[1]]
                prompt = (
                    f"You synthesize a superior hypothesis by combining two parents.\nGoal: {goal}\nPreferences: {prefs}\n"
                    f"Parent A:\n{h1.get('content','')}\nParent B:\n{h2.get('content','')}\nReturn the synthesized hypothesis."
                )
                resp = await llm.ainvoke(prompt)
                content = getattr(resp, "content", str(resp))
                new_contents.append({
                    "content": content,
                    "evolved_from": [h1["id"], h2["id"]],
                    "evolution_type": "combination",
                })
            else:
                inspiration = [by_id[t] for t in targets if t in by_id] if targets else hyps[: min(3, len(hyps))]
                neg = sorted(hyps, key=lambda x: x.get("elo_rating", 1200))[: min(3, len(hyps))]
                neg_text = "\n\n".join(f"- {h.get('content','')}" for h in neg)
                insp_text = "\n\n".join(f"Concept {i+1}: {h.get('content','')}" for i, h in enumerate(inspiration))
                prompt = (
                    f"You generate a novel hypothesis using analogical thinking.\nGoal: {goal}\nPreferences: {prefs}\n"
                    f"Inspiration:\n{insp_text}\nAvoid (unsuccessful avenues):\n{neg_text}\nReturn the novel hypothesis."
                )
                resp = await llm.ainvoke(prompt)
                content = getattr(resp, "content", str(resp))
                new_contents.append({
                    "content": content,
                    "evolved_from": [h["id"] for h in inspiration],
                    "evolution_type": "analogical",
                })
        except Exception as e:
            safe_append_error(s, f"Evolution error: {e}")
            return {}

        for nc in new_contents:
            if str(nc["content"]).strip():
                s.setdefault("hypotheses", []).insert(0, {
                    "id": str(uuid.uuid4()),
                    "content": nc["content"],
                    "elo_rating": 1200.0,
                    "reviews": [],
                    "is_reviewed": False,
                    "is_ranked": False,
                    "evolved_from": nc["evolved_from"],
                    "evolution_type": nc["evolution_type"],
                })
        return {"hypotheses": s["hypotheses"]}

    return evolution_node



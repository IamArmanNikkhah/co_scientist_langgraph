from __future__ import annotations

import uuid
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from ..prompts.generation import literature_strategy_prompt
from ..state import GraphState, safe_append_error


def make_generation_node(llm: ChatOpenAI):
    async def generation_node(state: GraphState) -> GraphState:
        s = state
        goal = (s.get("research_goal") or "").strip()
        if not goal:
            safe_append_error(s, "Generation: research_goal is empty.")
            return {}
        params = s.get("parameters", {}) or {}
        qty = int(params.get("quantity", 1))
        focus_area = str(params.get("focus_area", "")).strip()

        config = s.get("research_plan_config", {})
        preferences = config.get("preferences", "")
        constraints = config.get("constraints", []) or []

        # Assemble context from top hypotheses with reviews
        hyps = s.get("hypotheses", [])
        sorted_h = sorted(hyps, key=lambda h: h.get("elo_rating", 0), reverse=True)
        top = sorted_h[:3]
        context_parts: List[str] = []
        for i, h in enumerate(top, 1):
            ctx = f"Existing Hypothesis #{i} (ELO: {h.get('elo_rating', 0)}):\n{h.get('content', '')}"
            reviews = h.get("reviews", [])
            if reviews:
                ctx += "\nAssociated Reviews:"
                for r in reviews:
                    fb = r.get("qualitative_feedback", {})
                    ctx += f"\n- Overall: {r.get('scores', {}).get('overall', 'N/A')}/10; Summary: {fb.get('summary', '')}"
            context_parts.append(ctx)
        source_context = "\n\n---\n\n".join(context_parts)

        summary = s.get("meta_review_critique", "") or ""
        user_instructions = f"Supervisor Focus: {focus_area}".strip()

        added = 0
        for _ in range(qty):
            prompt = literature_strategy_prompt(
                goal=goal,
                summary=summary,
                preferences=preferences,
                constraints=constraints,
                instructions=user_instructions,
                source_hypotheses_context=source_context,
            )
            try:
                resp = await llm.ainvoke(prompt)
                text = getattr(resp, "content", str(resp)) or ""
                hypothesis_text = text.strip()
                if not hypothesis_text:
                    raise ValueError("Empty generation output.")

                new_h = {
                    "id": str(uuid.uuid4()),
                    "content": hypothesis_text,
                    "elo_rating": 1200.0,
                    "reviews": [],
                    "is_reviewed": False,
                    "is_ranked": False,
                }
                s.setdefault("hypotheses", []).append(new_h)
                added += 1
            except Exception as e:
                safe_append_error(s, f"Generation error: {e}")
                continue

        return {"hypotheses": s.get("hypotheses", [])}

    return generation_node



from __future__ import annotations

import uuid
from typing import Any, Dict, List
import re

from langchain_openai import ChatOpenAI

from ..prompts.generation import (
    literature_strategy_prompt,
    debate_generation_prompt,
)
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
        generation_mode = str(params.get("generation_mode", "standard")).strip().lower()
        debate_max_turns = int(params.get("debate_max_turns", 6))

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

        # Prefer curated literature chronology if available; otherwise meta-review summary
        chronology = s.get("articles_with_reasoning_text", "") or ""
        summary = chronology or (s.get("meta_review_critique", "") or "")
        user_instructions = f"Supervisor Focus: {focus_area}".strip()

        added = 0

        async def run_debate_once() -> str:
            transcript: List[str] = []
            last_turn_text = ""
            for turn in range(max(1, debate_max_turns)):
                speaker = "A" if turn % 2 == 0 else "B"
                prompt = debate_generation_prompt(
                    goal=goal,
                    summary=summary,
                    preferences=preferences,
                    constraints=constraints,
                    source_hypotheses_context=source_context,
                    transcript=transcript,
                    speaker_label=speaker,
                    focus_area=focus_area,
                )
                resp = await llm.ainvoke(prompt)
                text = getattr(resp, "content", str(resp)) or ""
                last_turn_text = text.strip()

                # Check for termination signal
                if re.search(r"^\s*HYPOTHESIS\s*:\s*.+", last_turn_text, re.IGNORECASE | re.MULTILINE):
                    return last_turn_text

                # Append non-finalizing turn to transcript (attribute to speaker)
                if last_turn_text:
                    transcript.append(f"Expert {speaker}: {last_turn_text}")

            # Fallback: return the last turn even if not finalized
            safe_append_error(s, "Debate mode reached max turns without finalization; using last turn output.")
            return last_turn_text

        for _ in range(qty):
            try:
                if generation_mode == "debate":
                    hypothesis_text = await run_debate_once()
                else:
                    prompt = literature_strategy_prompt(
                        goal=goal,
                        summary=summary,
                        preferences=preferences,
                        constraints=constraints,
                        instructions=user_instructions,
                        source_hypotheses_context=source_context,
                    )
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



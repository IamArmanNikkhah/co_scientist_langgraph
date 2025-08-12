from __future__ import annotations

import json
from typing import Dict, List

from langchain_openai import ChatOpenAI

from ..state import GraphState, safe_append_error
from ..tools import extract_json_from_text, infer_domain_from_goal


def make_observation_aggregator_node(llm: ChatOpenAI):
    async def observation_aggregator_node(state: GraphState) -> GraphState:
        s = state
        goal = s.get("research_goal", "")
        literature = (s.get("literature_content") or "").strip()
        params = s.get("parameters", {}) or {}
        priority_ids = params.get("priority_hypothesis_ids", [])

        # Build hypotheses context if priorities exist
        hyp_ctx = ""
        if priority_ids:
            all_h = s.get("hypotheses", [])
            by_id = {h.get("id"): h for h in all_h}
            chosen = [by_id.get(hid) for hid in priority_ids if by_id.get(hid)]
            if chosen:
                parts = [f"Hypothesis {h.get('id')[:8]}: {h.get('content','')}" for h in chosen]
                hyp_ctx = "\n".join(parts)

        if not literature:
            msg = (
                "No scientific literature provided. Reflection may proceed using general knowledge. "
                "Provide --literature-file to improve grounding."
            )
            return {"scientific_observations": msg}

        # Domain inference (LLM optional; fallback tool)
        try:
            dom_resp = await llm.ainvoke(
                f"Analyze the research goal and return ONLY one domain label from: biology, chemistry, physics, medicine, materials_science, computer_science, general.\nGoal: {goal}\nDomain:"
            )
            domain = getattr(dom_resp, "content", "general").strip().lower()
            if domain not in {"biology", "chemistry", "physics", "medicine", "materials_science", "computer_science", "general"}:
                domain = infer_domain_from_goal.invoke({"research_goal": goal})
        except Exception:
            domain = infer_domain_from_goal.invoke({"research_goal": goal})

        extraction_prompt = f"""You extract structured factual observations from literature.

Research Goal: {goal}
Domain: {domain}

Priority Hypotheses (if any):
{hyp_ctx or "None"}

Instructions:
- Extract ONLY factual observations, measurements, and findings
- Organize a JSON with 'extracted_observations' (list of items with source, observation_type, description, context, quantitative_data, relevance_keywords)
- Include 'observation_summary'

Literature Content:
{literature}

Return ONLY the JSON.
"""
        try:
            resp = await llm.ainvoke(extraction_prompt)
            content = getattr(resp, "content", str(resp))
            json_str = extract_json_from_text.invoke({"text": content}) or content
            parsed: Dict = json.loads(json_str)
            observations: List[Dict] = parsed.get("extracted_observations", [])
            summary: str = parsed.get("observation_summary", "")
            if not isinstance(observations, list):
                raise ValueError("Missing observations list.")
            formatted = f"Scientific Observations Summary:\n{summary}\n\nDetailed Observations:\n"
            for i, obs in enumerate(observations, 1):
                desc = str(obs.get("description", "")).strip()
                src = str(obs.get("source", "Unknown")).strip()
                formatted += f"- Observation {i}: {desc} (Source: {src})\n"
            return {"scientific_observations": formatted}
        except Exception as e:
            safe_append_error(s, f"Observation extraction failed: {e}")
            return {"scientific_observations": "Observation extraction failed; proceeding with general knowledge."}

    return observation_aggregator_node



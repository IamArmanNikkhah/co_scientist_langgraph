from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from ..prompts.ranking import build_ranking_comparison_prompt
from ..state import GraphState, safe_append_error
from ..tools import calculate_elo


def make_ranking_node(llm: ChatOpenAI, initial_elo: int = 1200, k_factor: int = 32, num_matches: int = 3):
    async def ranking_node(state: GraphState) -> GraphState:
        s = state
        decision = s.get("decision", {}) or {}
        goal = decision.get("research_goal", s.get("research_goal", ""))
        params = decision.get("parameters", {}) or {}
        new_ids: List[str] = params.get("newly_reviewed_ids", [])
        hyps = s.get("hypotheses", [])
        if not new_ids:
            new_ids = [h["id"] for h in hyps if h.get("is_reviewed") and not h.get("is_ranked")]

        if not new_ids:
            s["hypotheses"] = sorted(hyps, key=lambda x: x.get("elo_rating", initial_elo), reverse=True)
            return {"hypotheses": s["hypotheses"]}

        config = s.get("research_plan_config", {})
        criteria_text = config.get("preferences", "")

        hyp_map: Dict[str, Dict[str, Any]] = {h["id"]: h for h in hyps}
        for new_id in new_ids:
            if new_id not in hyp_map:
                continue
            opponents = [h for hid, h in hyp_map.items() if hid != new_id]
            if not opponents:
                hyp_map[new_id]["is_ranked"] = True
                continue
            opponents.sort(key=lambda x: x.get("elo_rating", initial_elo))
            n = len(opponents)
            top = opponents[int(n * 0.66):]
            mid = opponents[int(n * 0.33): int(n * 0.66)]
            low = opponents[: int(n * 0.33)]
            selected: List[Dict[str, Any]] = []
            if top:
                selected.append(random.choice(top))
            if mid:
                selected.append(random.choice(mid))
            if low:
                selected.append(random.choice(low))
            while len(selected) < min(num_matches, n):
                selected.append(random.choice(opponents))

            for opp in selected:
                h1 = hyp_map[new_id]
                h2 = opp
                try:
                    is_debate = abs(h1.get("elo_rating", initial_elo) - h2.get("elo_rating", initial_elo)) < 75
                    prompt = build_ranking_comparison_prompt(goal, h1, h2, criteria_text, is_debate)
                    resp = await llm.ainvoke(prompt)
                    text = getattr(resp, "content", str(resp))
                    m = re.search(r"\{.*\}", text, re.DOTALL)
                    if not m:
                        continue
                    result = json.loads(m.group(0))
                    if "winner" not in result:
                        continue
                    conf = str(result.get("confidence", "medium")).lower()
                    conf_map = {"high": 1.0, "medium": 0.66, "low": 0.33}
                    eff_k = k_factor * conf_map.get(conf, 0.5)
                    elo = calculate_elo.invoke({
                        "r1": h1["elo_rating"],
                        "r2": h2["elo_rating"],
                        "winner": int(result["winner"]),
                        "k_factor": float(eff_k),
                    })
                    h1["elo_rating"], h2["elo_rating"] = elo["new_r1"], elo["new_r2"]
                except Exception as e:
                    safe_append_error(s, f"Ranking match error: {e}")
                    continue
            hyp_map[new_id]["is_ranked"] = True

        s["hypotheses"] = sorted(hyp_map.values(), key=lambda x: x.get("elo_rating", initial_elo), reverse=True)
        return {"hypotheses": s["hypotheses"]}

    return ranking_node



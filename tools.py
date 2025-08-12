from __future__ import annotations

import re
from typing import Dict, Optional

from langchain_core.tools import tool


def _extract_first_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text.
    - Skips content before the first '{'
    - Tracks nested braces
    - Ignores braces inside quoted strings (handles escapes)
    - Works even if surrounded by code fences or extra prose
    """
    i = 0
    n = len(text)
    in_string = False
    escape = False
    depth = 0
    start = -1

    while i < n:
        ch = text[i]
        if depth == 0:
            if ch == '{':
                depth = 1
                start = i
                i += 1
                continue
            i += 1
            continue

        # depth > 0: inside JSON
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
        i += 1
    return None


@tool("extract_json_from_text")
def extract_json_from_text(text: str) -> Optional[str]:
    """Extract a single JSON object from arbitrary text.
    Supports fenced blocks (```...```) and inline JSON, returning the first balanced object.
    """
    if not text:
        return None

    # If there is a fenced block, focus on it first
    fence = re.search(r"```[a-zA-Z]*\n([\s\S]*?)```", text)
    if fence:
        candidate = _extract_first_balanced_json(fence.group(1))
        if candidate:
            return candidate

    # Fallback to full text scan
    return _extract_first_balanced_json(text)


@tool("calculate_elo")
def calculate_elo(r1: float, r2: float, winner: int, k_factor: float = 32.0) -> Dict[str, int]:
    """Compute Elo updates for two ratings given a winner. Returns new_r1 and new_r2 (rounded ints)."""
    e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
    e2 = 1.0 / (1.0 + 10 ** ((r1 - r2) / 400.0))
    s1 = 1 if winner == 1 else 0
    s2 = 1 if winner == 2 else 0
    new_r1 = int(round(r1 + k_factor * (s1 - e1)))
    new_r2 = int(round(r2 + k_factor * (s2 - e2)))
    return {"new_r1": new_r1, "new_r2": new_r2}


@tool("infer_domain_from_goal")
def infer_domain_from_goal(research_goal: str) -> str:
    """Heuristic domain inference to guide observation extraction when LLM-based inference is unavailable."""
    text = research_goal.lower()
    if any(k in text for k in ["gene", "protein", "cell", "biolog", "enzyme"]):
        return "biology"
    if any(k in text for k in ["clinic", "patient", "trial", "therapy", "disease"]):
        return "medicine"
    if any(k in text for k in ["synthesis", "reaction", "molecule", "compound"]):
        return "chemistry"
    if any(k in text for k in ["quantum", "optics", "relativity", "particle", "thermo"]):
        return "physics"
    if any(k in text for k in ["alloy", "polymer", "crystal", "material", "microstructure"]):
        return "materials_science"
    if any(k in text for k in ["algorithm", "neural", "network", "model", "data", "compute"]):
        return "computer_science"
    return "general"



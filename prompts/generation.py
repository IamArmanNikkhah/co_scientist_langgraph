from __future__ import annotations

import os
from typing import List


def literature_strategy_prompt(
    goal: str,
    summary: str,
    preferences: str,
    constraints: List[str],
    instructions: str,
    source_hypotheses_context: str,
) -> str:
    return f"""You are an expert scientist tasked with formulating a novel and robust hypothesis based on existing literature.

Goal: {goal}
Criteria: {preferences or "Generate a novel, testable, and scientifically rigorous hypothesis"}
Constraints:
{os.linesep.join(f"- {c}" for c in (constraints or [])) or "None specified"}

Context from existing top hypotheses and reviews:
{source_hypotheses_context or "No existing hypotheses to build upon."}

User Instructions:
{instructions or "None provided."}

Literature Review & Analytical Rationale:
{summary}

Return a single hypothesis with the following headings:
### Proposed Hypothesis: [...]
#### Hypothesis Statement
#### Detailed Description for Domain Experts
#### Falsifiability Statement
#### Key Assumptions
"""



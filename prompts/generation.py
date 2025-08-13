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


def debate_generation_prompt(
    goal: str,
    summary: str,
    preferences: str,
    constraints: List[str],
    source_hypotheses_context: str,
    transcript: List[str],
    speaker_label: str,
    focus_area: str,
) -> str:
    """Build a prompt for a single debate turn.

    The assistant simulates a debate between two experts (Expert A and Expert B).
    Termination rule: when consensus is reached, the model must output a final
    hypothesis starting with a single line:

    HYPOTHESIS: <concise title>

    followed immediately by a fully formatted hypothesis with the same headings
    used by literature_strategy_prompt.
    """
    constraints_text = os.linesep.join(f"- {c}" for c in (constraints or [])) or "None specified"
    transcript_text = os.linesep.join(transcript) if transcript else "<no prior turns>"
    return f"""You are orchestrating a scientific debate to produce a high-quality hypothesis.

Goal: {goal}
Focus Area: {focus_area or "general"}
Criteria: {preferences or "Generate a novel, testable, and scientifically rigorous hypothesis"}
Constraints:
{constraints_text}

Context from existing top hypotheses and reviews:
{source_hypotheses_context or "No existing hypotheses to build upon."}

Literature Review & Analytical Rationale:
{summary}

Debate Transcript So Far:
{transcript_text}

Instructions:
- You are Expert {speaker_label}. Contribute one concise turn that advances toward consensus.
- If consensus is reached, FINALIZE immediately by outputting ONLY:
  1) A single line beginning exactly with: HYPOTHESIS: <concise title>
  2) Then the full hypothesis with these headings:
     ### Proposed Hypothesis: [...]
     #### Hypothesis Statement
     #### Detailed Description for Domain Experts
     #### Falsifiability Statement
     #### Key Assumptions
- If not finalizing, DO NOT include the HYPOTHESIS line; provide only the next debate turn content.
"""



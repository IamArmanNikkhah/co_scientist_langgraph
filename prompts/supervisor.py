from __future__ import annotations

import json


def build_supervisor_prompt(
    statistics: dict,
    enhanced_statistics: dict,
    meta_review_content: str,
    research_goal: str,
    preferences: str,
    hypotheses_summary: list,
) -> str:
    """Paper-compliant Supervisor prompt adapted from the LangFlow component."""
    return f"""You are the Expert Supervisor Agent for an advanced AI co-scientist system. Your role is to analyze the current research state and determine the single most strategic next action, adhering to "Towards an AI co-scientist".

<context>
## Research Goal
`{research_goal}`

## Evaluation Preferences
`{preferences}`

## Current System Statistics
```json
{json.dumps(statistics, indent=2)}
```

## Enhanced Strategic Metrics
```json
{json.dumps(enhanced_statistics, indent=2)}
```

## Hypotheses Summary
```json
{json.dumps(hypotheses_summary, indent=2)}
```

## Meta-Review Insights
`{meta_review_content if meta_review_content else "No meta-review available yet."}`
</context>

# Instructions

- Enforce absolute precedence: reflect before meta_review if unreviewed > 0; rank before meta_review if newly_reviewed > 0; meta_review only when both are 0 and sufficient new reviews.
- Then score remaining actions and pick the best.
- When selecting generate, choose the generation method:
  - Use generation_mode = "debate" during exploratory phases or when hypothesis_diversity is low or stagnation_risk is high; otherwise use "standard".
  - debate_max_turns defaults to 6 if not specified.

Output a single JSON with keys: next_task, parameters, rationale, strategic_context.

JSON schemas:
- If generate: {{"quantity": <int>, "focus_area": "<string>", "generation_mode": "<standard|debate>", "debate_max_turns": <int>}}
- If reflect: {{"priority_hypothesis_ids": [<ids>], "review_depth": "<standard|deep>"}}
- If rank: {{"newly_reviewed_ids": [<ids>]}}
- If evolve: {{"target_hypothesis_ids": [<ids>], "strategy": "<refine|combine|analogize>"}}
- If meta_review: {{"scope": "<full_history|last_3_iterations>", "focus": "<identify_patterns|suggest_new_directions>"}}
- If terminate: {{"reason": "<iteration_limit|breakthrough_achieved|stagnation_limit>", "final_hypothesis_id": "<id>"}}
"""



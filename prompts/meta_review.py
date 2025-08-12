from __future__ import annotations

import json
from typing import Any, Dict


def build_meta_review_prompt(goal: str, preferences: str, focus: str, quant_summary: Dict[str, Any], reviews_text: str) -> str:
    return f"""You are an expert in scientific meta-analysis.

Research Goal: {goal}
Guiding Preferences: {preferences}

Quantitative Summary:
```json
{json.dumps(quant_summary, indent=2)}
```

Full Text of Recent Reviews:
{reviews_text}

Focus: {focus}

Return one JSON with:
- quantitative_insights: summary, elo_distribution_comment
- qualitative_analysis: emerging_themes, common_strengths, critical_weaknesses
- strategic_recommendations: next_evolution_strategy, new_generation_focus, stop_exploring
"""



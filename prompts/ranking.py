from __future__ import annotations


def build_ranking_comparison_prompt(goal: str, h1: dict, h2: dict, criteria_text: str, is_debate: bool) -> str:
    debate = (
        "\nDebate Procedure:\n3-5 turn debate between two experts; conclude with a JSON object only.\n"
        if is_debate
        else "\nReasoning Procedure:\nEvaluate both hypotheses against criteria; conclude with a JSON object only.\n"
    )
    return f"""You are an expert evaluator comparing two scientific hypotheses.

Goal: {goal}
Evaluation Criteria:
{criteria_text or "novelty, correctness, potential impact, feasibility"}

Hypothesis 1:
{h1.get('content','')}
Review of H1:
{(h1.get('reviews') or [{}])[-1].get('paper_analysis',{}).get('full_analysis', '') if h1.get('reviews') else ''}

Hypothesis 2:
{h2.get('content','')}
Review of H2:
{(h2.get('reviews') or [{}])[-1].get('paper_analysis',{}).get('full_analysis', '') if h2.get('reviews') else ''}

{debate}

Final Output:
A single JSON:
{{"rationale": "...", "confidence": "<high|medium|low>", "winner": <1|2>}}
"""



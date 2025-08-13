from __future__ import annotations

from typing import Optional


def build_reflection_prompt(
    hypothesis: str,
    goal: str,
    observations: str,
    review_depth: str,
    strategic_context: str,
    good_example: Optional[str],
    bad_example: Optional[str],
) -> str:
    deep = ""
    if review_depth == "deep":
        deep = (
            "\nDeep Review Instructions:\n"
            "Adopt two personas in your full_analysis: Proponent and Skeptic, then synthesize.\n"
        )
    elif review_depth == "deep_verification":
        deep = (
            "\nDeep Verification Review Instructions:\n"
            "- Decompose the hypothesis into top-level assumptions (2–6).\n"
            "- For each assumption, identify sub-assumptions (facts or relationships) and evaluate them.\n"
            "- For each sub-assumption, provide:\n"
            "  - check_method: how you would verify it (experiment, literature, calculation)\n"
            "  - evidence_needed: minimal evidence/data to verify\n"
            "  - correctness: one of [\"likely_correct\",\"uncertain\",\"likely_incorrect\"]\n"
            "  - invalidation_reasons: a list of concrete reasons it could fail\n"
            "  - notes: brief commentary\n"
            "- Provide a summary of critical failure points and an overall decision.\n"
            "Output JSON MUST include a 'deep_verification' object with this schema:\n"
            "{\n"
            "  \"assumptions\": [\n"
            "    {\"assumption\": str,\n"
            "     \"sub_assumptions\": [\n"
            "        {\"text\": str, \"check_method\": str, \"evidence_needed\": str,\n"
            "         \"correctness\": \"likely_correct\"|\"uncertain\"|\"likely_incorrect\",\n"
            "         \"invalidation_reasons\": [str], \"notes\": str}\n"
            "     ]}\n"
            "  ],\n"
            "  \"critical_failure_points\": [str],\n"
            "  \"overall_decision\": \"keep\"|\"revise\"|\"reject\"\n"
            "}\n"
        )
    calibration = ""
    if good_example and bad_example:
        calibration = f"\nCalibration Examples:\nGood Example:\n{good_example}\n\nPoor Example:\n{bad_example}\n"
    return f"""You are an expert in scientific hypothesis evaluation.

Strategic Context:
"{strategic_context}"

{calibration}

Your task is to output a single JSON object with:
- full_analysis
- classification in ["missing piece","already explained","other explanations more likely","neutral","disproved"]
- scores: overall, novelty, validity, testability, specificity (1..10)
    - strengths, weaknesses, suggestions (arrays)

    {deep}

Article for Analysis:
{observations or "No specific observations provided."}

Hypothesis to Evaluate:
{hypothesis}

Your JSON:
"""



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



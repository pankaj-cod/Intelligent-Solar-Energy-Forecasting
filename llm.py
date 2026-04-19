"""
LLM-powered recommendation engine for solar grid optimization.

Uses Groq (Llama 3.3 70B) to synthesize forecast data, risk analysis,
and retrieved guidelines into a single structured recommendation.
"""

import json
from groq import Groq


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a solar grid optimization assistant.

Your job is to analyze solar generation forecasts, risk assessments, and
operational guidelines, then produce a concise, actionable recommendation
for grid operators.

RULES:
1. Base every claim strictly on the data provided — never invent numbers.
2. Be concise: each field should be 1-3 sentences maximum.
3. Respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
4. Use the exact schema shown below, with no extra keys.

REQUIRED OUTPUT SCHEMA:
{
  "risk_interpretation": "<plain-English summary of the current risk level and what it means for grid operations>",
  "strategy": "<high-level strategy to follow given the forecast and risk>",
  "actions": "<specific, numbered steps the grid operator should take right now>",
  "justification": "<why these actions are appropriate, referencing the retrieved guidelines>"
}
"""


def generate_recommendation(summary, risk, retrieved_docs, api_key):
    """Generate a structured grid-optimization recommendation via Groq LLM.

    Parameters
    ----------
    summary : dict
        Output of summarize_forecast() — contains average_generation,
        max_generation, min_generation, variability.
    risk : dict
        Output of analyze_risk() — contains risk_level, details.
    retrieved_docs : list[dict]
        Output of retrieve_guidelines() — each dict has 'guideline' and 'score'.
    api_key : str
        Groq API key.

    Returns
    -------
    dict
        Structured recommendation with keys: risk_interpretation, strategy,
        actions, justification.
    """
    # ── Build the user prompt with all context ────────────────────────────────
    guidelines_text = "\n".join(
        f"  {i+1}. (relevance {g['score']:.0%}) {g['guideline']}"
        for i, g in enumerate(retrieved_docs)
    )

    user_prompt = f"""\
=== FORECAST SUMMARY ===
Average Generation : {summary['average_generation']} kW
Peak Generation    : {summary['max_generation']} kW
Minimum Generation : {summary['min_generation']} kW
Variability        : {summary['variability']}

=== RISK ANALYSIS ===
Risk Level : {risk['risk_level']}
Details    : {risk['details']}

=== RETRIEVED OPERATIONAL GUIDELINES ===
{guidelines_text}

Based on the above data, produce the JSON recommendation now."""

    # ── Call Groq ─────────────────────────────────────────────────────────────
    client = Groq(api_key=api_key)

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,          # deterministic
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = chat.choices[0].message.content

    # ── Parse and validate ────────────────────────────────────────────────────
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "risk_interpretation": raw,
            "strategy": "Unable to parse structured response.",
            "actions": "Please retry.",
            "justification": "LLM returned non-JSON output.",
        }

    # Ensure all expected keys are present
    for key in ("risk_interpretation", "strategy", "actions", "justification"):
        result.setdefault(key, "—")

    return result

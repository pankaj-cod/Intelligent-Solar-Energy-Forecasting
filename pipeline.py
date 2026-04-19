"""
Unified agent pipeline for solar grid optimization.

Chains all Phase 2 steps into a single callable:
  predictions → summary → risk → RAG retrieval → LLM recommendation

No complex frameworks — just clean function composition.
"""

from analysis import summarize_forecast, analyze_risk
from rag import retrieve_guidelines
from llm import generate_recommendation


def run_ai_optimization(predictions, api_key=None):
    """Run the full AI grid-optimization pipeline.

    Parameters
    ----------
    predictions : array-like
        Sequence of predicted AC power values (kW), e.g. 24 hourly values.
    api_key : str, optional
        Groq API key.  When provided the pipeline includes an LLM-generated
        strategy recommendation; otherwise that field is None.

    Returns
    -------
    dict
        {
            "summary":        { average_generation, max_generation,
                                min_generation, variability },
            "risk":           { risk_level, details },
            "guidelines":     [ { guideline, score }, ... ],
            "recommendation": { risk_interpretation, strategy,
                                actions, justification } | None
        }
    """
    # Step 1 — Forecast summary
    summary = summarize_forecast(predictions)

    # Step 2 — Risk analysis
    risk = analyze_risk(predictions)

    # Step 3 — Build a natural-language query from the risk assessment
    query = f"{risk['risk_level']}. {risk['details']}"

    # Step 4 — Retrieve relevant grid-management guidelines (RAG)
    guidelines = retrieve_guidelines(query, top_k=3)

    # Step 5 — LLM recommendation (optional, needs API key)
    recommendation = None
    if api_key:
        recommendation = generate_recommendation(
            summary=summary,
            risk=risk,
            retrieved_docs=guidelines,
            api_key=api_key,
        )

    return {
        "summary":        summary,
        "risk":           risk,
        "guidelines":     guidelines,
        "recommendation": recommendation,
    }

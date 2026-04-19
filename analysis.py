"""
Statistical analysis functions for solar generation forecasts.

All logic is rule-based (no ML models).  Functions accept arrays / Series
of predicted AC power values and return clean dictionaries.
"""

import numpy as np


def summarize_forecast(predictions):
    """Summarize forecast predictions into key generation metrics.

    Parameters
    ----------
    predictions : array-like or pandas Series
        Sequence of predicted AC power values (kW).

    Returns
    -------
    dict
        average_generation, max_generation, min_generation, variability.
    """
    predictions = np.array(predictions, dtype=float)
    avg = float(predictions.mean())
    std = float(predictions.std())

    # Classify variability using coefficient of variation
    if avg == 0:
        variability = "low"
    else:
        cv = std / abs(avg)
        if cv < 0.3:
            variability = "low"
        elif cv < 0.6:
            variability = "medium"
        else:
            variability = "high"

    return {
        "average_generation": round(avg, 2),
        "max_generation":     round(float(predictions.max()), 2),
        "min_generation":     round(float(predictions.min()), 2),
        "variability":        variability,
    }


def analyze_risk(predictions):
    """Rule-based risk assessment for a sequence of power predictions.

    Parameters
    ----------
    predictions : array-like or pandas Series
        Sequence of predicted AC power values (kW), e.g. 24 hourly values.

    Returns
    -------
    dict
        risk_level  – 'High variability risk' | 'Low generation risk' | 'Stable generation'
        details     – human-readable explanation including any sudden-drop alerts.
    """
    predictions = np.array(predictions, dtype=float)
    avg = float(predictions.mean())
    std = float(predictions.std())

    # ── 1. Classify overall risk ──────────────────────────────────────────────
    #   coefficient of variation > 0.5 → high variability
    #   mean below 50 kW              → low generation
    #   otherwise                      → stable
    if avg != 0 and (std / abs(avg)) > 0.5:
        risk_level = "High variability risk"
        details = (f"Standard deviation ({std:,.1f} kW) is large relative to "
                   f"the mean ({avg:,.1f} kW), indicating unpredictable output.")
    elif avg < 50:
        risk_level = "Low generation risk"
        details = (f"Average predicted generation is only {avg:,.1f} kW, "
                   f"which may be insufficient to meet demand.")
    else:
        risk_level = "Stable generation"
        details = (f"Generation is healthy (avg {avg:,.1f} kW) with "
                   f"moderate spread (std {std:,.1f} kW).")

    # ── 2. Detect sudden drops between consecutive values ─────────────────────
    drops = []
    for i in range(1, len(predictions)):
        prev_val = predictions[i - 1]
        curr_val = predictions[i]
        if prev_val > 0 and (prev_val - curr_val) / prev_val > 0.3:
            drops.append({
                "from_index": i - 1,
                "to_index":   i,
                "drop_pct":   round((prev_val - curr_val) / prev_val * 100, 1),
                "from_kw":    round(prev_val, 1),
                "to_kw":      round(curr_val, 1),
            })

    if drops:
        drop_msgs = [f"Hour {d['from_index']}\u2192{d['to_index']}: "
                     f"{d['from_kw']} \u2192 {d['to_kw']} kW (\u2212{d['drop_pct']}%)"
                     for d in drops]
        details += f" \u26a0 Sudden drops detected: {'; '.join(drop_msgs)}."
        # Escalate risk if drops are present and risk was stable
        if risk_level == "Stable generation":
            risk_level = "High variability risk"

    return {
        "risk_level": risk_level,
        "details":    details,
    }

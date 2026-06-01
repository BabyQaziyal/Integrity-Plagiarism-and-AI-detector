"""aggregator.py — combine sub-signals into AI % and the integrity score.

Combined AI score = weighted blend of:
  * DistilBERT classifier  P(AI)        (primary)
  * GPT-2 perplexity        likelihood   (secondary)
  * burstiness              likelihood   (light)

Missing components are dropped and the remaining weights renormalised, so the
system degrades gracefully (e.g. before the classifier is trained).

Integrity score = 100 - weighted(plagiarism %, AI %).
"""
from __future__ import annotations

from src.config import (AI_CONF_FLOOR, AI_WEIGHTS, INTEGRITY_AI_WEIGHT,
                        INTEGRITY_PLAGIARISM_WEIGHT)


def combine_ai_score(classifier_p: float | None,
                     perplexity_ai: float | None,
                     burstiness_ai: float | None,
                     weights: dict | None = None) -> dict:
    """Blend available AI sub-signals (each 0..1) into an AI percentage.

    Two robustness rules keep weak/missing signals from manufacturing an AI
    verdict (the old failure mode for short or casual text):

    * **Abstention** — a signal of ``None`` means "no evidence" and is dropped,
      with the remaining weights renormalised. (Previously perplexity/burstiness
      returned 0.5 when they had nothing to say, which read as 50% AI.)
    * **Confidence weighting** — each signal's weight is scaled by how far it is
      from 0.5 (its confidence). An unsure signal (≈0.5) barely moves the score;
      a confident one dominates. ``overall_confidence`` summarises this so the
      analyzer can fall back to "insufficient evidence" when everything is unsure.
    """
    weights = weights or AI_WEIGHTS
    components = {
        "classifier": classifier_p,
        "perplexity": perplexity_ai,
        "burstiness": burstiness_ai,
    }
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return {"ai_percent": 0.0, "components": components, "used_weights": {},
                "overall_confidence": 0.0}

    # confidence(signal) = |p - 0.5| * 2, floored so a signal is never fully muted
    conf = {k: max(AI_CONF_FLOOR, abs(v - 0.5) * 2) for k, v in available.items()}
    eff = {k: weights[k] * conf[k] for k in available}
    wsum = sum(eff.values())
    used = {k: eff[k] / wsum for k in available}
    blended = sum(available[k] * used[k] for k in available)

    # Overall confidence = config-weighted mean of the *true* per-signal
    # confidences (no floor), so a lone barely-past-50% signal stays "unsure".
    raw_conf = {k: abs(available[k] - 0.5) * 2 for k in available}
    cw = sum(weights[k] for k in available)
    overall_conf = sum(weights[k] * raw_conf[k] for k in available) / cw if cw else 0.0

    return {
        "ai_percent": round(100.0 * blended, 2),
        "components": {k: (round(v, 4) if v is not None else None)
                       for k, v in components.items()},
        "used_weights": {k: round(w, 3) for k, w in used.items()},
        "overall_confidence": round(overall_conf, 4),
    }


def integrity_score(plagiarism_percent: float, ai_percent: float) -> float:
    """100 = clean; lower = more concerning. Weighted penalty of both axes."""
    penalty = (INTEGRITY_PLAGIARISM_WEIGHT * plagiarism_percent
               + INTEGRITY_AI_WEIGHT * ai_percent)
    return round(max(0.0, 100.0 - penalty), 2)


def verdict(integrity: float) -> str:
    if integrity >= 85:
        return "Low risk"
    if integrity >= 60:
        return "Review recommended"
    return "High risk"

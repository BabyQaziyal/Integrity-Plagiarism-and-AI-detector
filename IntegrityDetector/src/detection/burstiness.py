"""burstiness.py — sentence-variation analysis (a lightweight AI signal).

Human prose is typically "bursty": it mixes short and long sentences, giving a
high variance in sentence length. LLM prose tends to be more uniform. We measure
sentence-length statistics and map low variation to a higher AI likelihood.

Pure-stdlib + regex (no model). Contributes a small weight to the combined AI
score; it is interpretable evidence rather than a strong classifier on its own.
"""
from __future__ import annotations

import math

from src.data.text_cleaning import split_sentences, word_count


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def analyze_burstiness(text: str) -> dict:
    sentences = split_sentences(text)
    lengths = [word_count(s) for s in sentences if word_count(s) > 0]
    n = len(lengths)
    if n < 3:
        # Abstain (None) rather than 0.5 — too few sentences is "no signal", not
        # "50% AI". Returning 0.5 here used to push short text toward an AI verdict.
        return {"sentence_count": n, "mean_sentence_len": 0.0, "std_sentence_len": 0.0,
                "cv": 0.0, "burstiness": 0.0, "ai_likelihood": None,
                "note": "too few sentences for a reliable signal"}

    mean = sum(lengths) / n
    var = sum((x - mean) ** 2 for x in lengths) / n
    std = math.sqrt(var)
    cv = std / mean if mean else 0.0
    # Goh-Barabási burstiness in [-1, 1]; higher => burstier (more human).
    burstiness = (std - mean) / (std + mean) if (std + mean) else 0.0

    # Low coefficient of variation => uniform => more AI-like.
    # Centre ~0.55 (typical human essay cv); slope tuned for a soft signal.
    ai_likelihood = round(_sigmoid(6.0 * (0.55 - cv)), 4)

    return {
        "sentence_count": n,
        "mean_sentence_len": round(mean, 2),
        "std_sentence_len": round(std, 2),
        "cv": round(cv, 4),
        "burstiness": round(burstiness, 4),
        "ai_likelihood": ai_likelihood,
    }

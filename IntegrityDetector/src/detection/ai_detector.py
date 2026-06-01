"""ai_detector.py — DistilBERT AI-vs-human classifier (inference).

Loads the fine-tuned model from ``AI_DETECTOR_DIR`` when available; otherwise it
falls back to the base DistilBERT with an untrained head and reports
``is_trained=False`` so the aggregator can ignore an unreliable signal.

Long documents are split into overlapping ≤512-token windows; per-window P(AI)
is averaged (weighted by window length).

Train the model with:  python -m src.training.train_ai_detector
"""
from __future__ import annotations

from src import config
from src.config import (AI_DETECTOR_DIR, MAX_SEQ_LEN, MIN_CLASSIFIER_WORDS,
                        MIN_RELIABLE_WORDS)
from src.data.text_cleaning import normalize_text, word_count
from src.logging_utils import get_logger

log = get_logger("ai_detector")

_MODEL = None
_TOK = None
_DEVICE = None
_IS_TRAINED = False


def _ensure_model():
    global _MODEL, _TOK, _DEVICE, _IS_TRAINED
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _DEVICE = config.get_device(require_cuda=False)
    log.info("Loading fine-tuned AI detector from %s on %s", AI_DETECTOR_DIR, _DEVICE)
    _TOK = AutoTokenizer.from_pretrained(str(AI_DETECTOR_DIR), cache_dir=str(config.CACHE_DIR))
    _MODEL = AutoModelForSequenceClassification.from_pretrained(
        str(AI_DETECTOR_DIR), num_labels=2, cache_dir=str(config.CACHE_DIR)).to(_DEVICE).eval()
    _IS_TRAINED = True


def is_available() -> bool:
    """True only when a fine-tuned model exists. The classifier signal is ignored
    until then (no pointless base-model download for a discarded prediction)."""
    return (AI_DETECTOR_DIR / "config.json").exists()


def predict_proba(text: str, max_windows: int = 16) -> dict:
    """Return P(AI) for a document by averaging over ≤512-token windows.

    If no fine-tuned model is present, returns is_trained=False without loading
    anything (the aggregator then blends only perplexity + burstiness)."""
    if not is_available():
        return {"p_ai": None, "is_trained": False, "windows": 0,
                "note": f"fine-tuned model not found at {AI_DETECTOR_DIR}; "
                        f"run: python scripts/train.py"}
    import torch
    _ensure_model()
    text = normalize_text(text)
    wc = word_count(text)
    # A handful of words carries almost no stylistic signal; abstain instead of
    # guessing (the aggregator then drops the classifier rather than reading the
    # guess as evidence). The model now sees short text in training, so anything
    # at/above the floor gets a real prediction.
    if wc < MIN_CLASSIFIER_WORDS:
        return {"p_ai": None, "is_trained": _IS_TRAINED, "windows": 0,
                "confidence": 0.0, "low_confidence": True,
                "note": f"only {wc} word(s) — too short to classify"}

    enc = _TOK(text, truncation=True, max_length=MAX_SEQ_LEN, stride=64,
               return_overflowing_tokens=True, padding="max_length",
               return_tensors="pt")
    input_ids = enc["input_ids"][:max_windows].to(_DEVICE)
    attn = enc["attention_mask"][:max_windows].to(_DEVICE)

    use_amp = _DEVICE.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=_DEVICE.type, enabled=use_amp):
        logits = _MODEL(input_ids=input_ids, attention_mask=attn).logits.float()
    probs = torch.softmax(logits, dim=-1)[:, 1]            # P(label==1==AI) per window
    weights = attn.sum(dim=1).float()
    p_ai = float((probs * weights).sum() / weights.sum())
    return {"p_ai": round(p_ai, 4), "is_trained": _IS_TRAINED,
            "windows": int(input_ids.size(0)),
            "confidence": round(abs(p_ai - 0.5) * 2, 4),   # 0 = unsure, 1 = certain
            "low_confidence": wc < MIN_RELIABLE_WORDS}


def analyze_ai_classifier(text: str) -> dict:
    res = predict_proba(text)
    res["ai_likelihood"] = res["p_ai"]
    return res


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "A short sentence for the classifier."
    print(analyze_ai_classifier(s))

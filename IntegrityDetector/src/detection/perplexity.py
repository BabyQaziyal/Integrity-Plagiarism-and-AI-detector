"""perplexity.py — GPT-2 perplexity as an AI-content signal.

AI-generated text is usually *more predictable* under a language model, so it has
**lower** perplexity than spontaneous human writing. We compute GPT-2 perplexity
with a sliding window (so long documents are handled) and map it to a 0..1 AI
likelihood. The model is loaded once and cached.

Run:  python -m src.detection.perplexity "some text to score"
"""
from __future__ import annotations

from src import config
from src.config import PERPLEXITY_MODEL
from src.data.text_cleaning import normalize_text, word_count
from src.logging_utils import get_logger

log = get_logger("perplexity")

# Map perplexity -> AI likelihood. Below PPL_AI ~ machine-like; above PPL_HUMAN ~ human.
PPL_AI = 25.0
PPL_HUMAN = 90.0

_MODEL = None
_TOK = None
_DEVICE = None


def _ensure_model():
    global _MODEL, _TOK, _DEVICE
    if _MODEL is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _DEVICE = config.get_device(require_cuda=False)
    log.info("Loading perplexity model '%s' on %s", PERPLEXITY_MODEL, _DEVICE)
    _TOK = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL, cache_dir=str(config.CACHE_DIR))
    _MODEL = AutoModelForCausalLM.from_pretrained(
        PERPLEXITY_MODEL, cache_dir=str(config.CACHE_DIR)).to(_DEVICE).eval()
    if _TOK.pad_token is None:
        _TOK.pad_token = _TOK.eos_token


def compute_perplexity(text: str, max_length: int = 1024, stride: int = 512) -> float:
    """Sliding-window perplexity (standard HF recipe). Returns +inf-safe float."""
    import torch
    _ensure_model()
    text = normalize_text(text)
    if word_count(text) < 10:
        return float("nan")

    enc = _TOK(text, return_tensors="pt")
    input_ids = enc.input_ids.to(_DEVICE)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("nan")

    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end                  # tokens actually scored this window
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100               # ignore overlap region
        with torch.no_grad():
            out = _MODEL(ids, labels=target)
        # out.loss is mean NLL over scored tokens; multiply back to a sum
        valid = max(1, trg_len - 1)
        nll_sum += out.loss.item() * valid
        n_tokens += valid
        prev_end = end
        if end == seq_len:
            break
    import math
    mean_nll = nll_sum / max(1, n_tokens)
    return float(math.exp(mean_nll)) if mean_nll < 20 else float("inf")


def perplexity_to_ai(ppl: float):
    """Map perplexity to P(AI). Returns None (abstain) when there is no signal —
    a missing signal must NOT be read as 0.5 (= '50% AI'), which previously
    inflated the score for short text."""
    if ppl != ppl:                                # NaN -> too short to score
        return None
    if ppl == float("inf"):
        return 0.0
    score = (PPL_HUMAN - ppl) / (PPL_HUMAN - PPL_AI)
    return round(min(1.0, max(0.0, score)), 4)


def analyze_perplexity(text: str) -> dict:
    ppl = compute_perplexity(text)
    return {
        "perplexity": None if ppl != ppl else round(ppl, 2) if ppl != float("inf") else None,
        "ai_likelihood": perplexity_to_ai(ppl),    # may be None (abstain)
        "model": PERPLEXITY_MODEL,
    }


if __name__ == "__main__":
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else "This is a short test sentence to score."
    print(analyze_perplexity(sample))

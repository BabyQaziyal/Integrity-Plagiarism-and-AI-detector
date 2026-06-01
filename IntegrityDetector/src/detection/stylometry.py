"""stylometry.py — per-student writing fingerprint + history comparison.

Extracts a numeric "writing fingerprint" from a document:
  * average / std sentence length
  * vocabulary richness (type-token & hapax ratios)
  * punctuation style (per-100-word rates)
  * readability (Flesch reading ease / grade — textstat if present, else stdlib)
  * POS-tag distribution (spaCy if installed, else skipped gracefully)

``compare_to_history`` z-scores a new submission's fingerprint against a
student's previous submissions to flag a sudden style shift (a classic signal of
ghost-writing or AI assistance).

spaCy + textstat are OPTIONAL: the fingerprint degrades gracefully without them.
For full POS features:  pip install spacy && python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import math
import re

from src.config import SPACY_MODEL
from src.data.text_cleaning import normalize_text, split_sentences
from src.logging_utils import get_logger

log = get_logger("stylometry")

_WORD = re.compile(r"[A-Za-z']+")
_PUNCT = [",", ";", ":", "!", "?", '"', "-", "(", ")", "..."]
_FUNCTION_WORDS = {
    "the", "of", "and", "a", "to", "in", "is", "it", "you", "that", "he", "was",
    "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this",
    "have", "from", "or", "had", "by", "but", "not", "what", "all", "were", "we",
    "when", "your", "can", "said", "there", "an", "which", "their", "if", "do",
    "will", "each", "about", "how", "up", "out", "them", "then", "she", "many",
}

# numeric keys used for fingerprint comparison (must be stable across versions)
FEATURE_KEYS = [
    "avg_sentence_len", "std_sentence_len", "avg_word_len", "type_token_ratio",
    "hapax_ratio", "function_word_ratio", "flesch_reading_ease",
    "flesch_kincaid_grade", "punct_comma", "punct_semicolon", "punct_colon",
    "punct_exclaim", "punct_question", "punct_dash",
    "pos_noun", "pos_verb", "pos_adj", "pos_adv", "pos_pron", "pos_propn",
]

_NLP = None
_SPACY_TRIED = False


def _spacy():
    """Lazily load spaCy; return None (once) if unavailable."""
    global _NLP, _SPACY_TRIED
    if _SPACY_TRIED:
        return _NLP
    _SPACY_TRIED = True
    try:
        import spacy
        _NLP = spacy.load(SPACY_MODEL, disable=["ner", "lemmatizer", "parser"])
        log.info("spaCy model '%s' loaded for POS features", SPACY_MODEL)
    except Exception as exc:  # noqa: BLE001
        log.warning("spaCy unavailable (%s) — POS features skipped. "
                    "Install: pip install spacy && python -m spacy download %s",
                    type(exc).__name__, SPACY_MODEL)
        _NLP = None
    return _NLP


def _count_syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def _readability(words: list[str], n_sentences: int) -> tuple[float, float]:
    """Flesch reading ease + Flesch-Kincaid grade (textstat if available)."""
    try:
        import textstat
        text = " ".join(words)
        return (round(float(textstat.flesch_reading_ease(text)), 2),
                round(float(textstat.flesch_kincaid_grade(text)), 2))
    except Exception:  # noqa: BLE001
        nw, ns = max(1, len(words)), max(1, n_sentences)
        syl = sum(_count_syllables(w) for w in words)
        wps, spw = nw / ns, syl / nw
        fre = 206.835 - 1.015 * wps - 84.6 * spw
        fkg = 0.39 * wps + 11.8 * spw - 15.59
        return round(fre, 2), round(fkg, 2)


def extract_fingerprint(text: str) -> dict:
    text = normalize_text(text)
    sentences = split_sentences(text)
    words = _WORD.findall(text.lower())
    n_words = max(1, len(words))
    n_sent = max(1, len(sentences))

    sent_lens = [len(_WORD.findall(s)) for s in sentences] or [0]
    mean_sl = sum(sent_lens) / len(sent_lens)
    std_sl = math.sqrt(sum((x - mean_sl) ** 2 for x in sent_lens) / len(sent_lens))

    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    hapax = sum(1 for c in vocab.values() if c == 1)
    func = sum(1 for w in words if w in _FUNCTION_WORDS)

    fre, fkg = _readability(words, n_sent)

    fp = {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_len": round(mean_sl, 3),
        "std_sentence_len": round(std_sl, 3),
        "avg_word_len": round(sum(len(w) for w in words) / n_words, 3),
        "type_token_ratio": round(len(vocab) / n_words, 4),
        "hapax_ratio": round(hapax / n_words, 4),
        "function_word_ratio": round(func / n_words, 4),
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkg,
    }

    # punctuation per 100 words
    per100 = 100.0 / n_words
    fp["punct_comma"] = round(text.count(",") * per100, 3)
    fp["punct_semicolon"] = round(text.count(";") * per100, 3)
    fp["punct_colon"] = round(text.count(":") * per100, 3)
    fp["punct_exclaim"] = round(text.count("!") * per100, 3)
    fp["punct_question"] = round(text.count("?") * per100, 3)
    fp["punct_dash"] = round(text.count("-") * per100, 3)

    # POS distribution (spaCy, optional)
    nlp = _spacy()
    pos_keys = {"pos_noun": "NOUN", "pos_verb": "VERB", "pos_adj": "ADJ",
                "pos_adv": "ADV", "pos_pron": "PRON", "pos_propn": "PROPN"}
    if nlp is not None and text:
        doc = nlp(text[:100000])                       # cap for very long docs
        counts: dict[str, int] = {}
        total = 0
        for tok in doc:
            if tok.is_alpha:
                counts[tok.pos_] = counts.get(tok.pos_, 0) + 1
                total += 1
        total = max(1, total)
        for key, pos in pos_keys.items():
            fp[key] = round(counts.get(pos, 0) / total, 4)
        fp["pos_available"] = True
    else:
        for key in pos_keys:
            fp[key] = None
        fp["pos_available"] = False

    return fp


def compare_to_history(new_fp: dict, history_fps: list[dict]) -> dict:
    """z-score the new fingerprint against the student's prior submissions."""
    history_fps = [h for h in (history_fps or []) if h]
    if len(history_fps) < 2:
        return {"consistency": None, "inconsistency_score": None,
                "n_history": len(history_fps), "flags": [], "deltas": {},
                "note": "Need >=2 prior submissions to assess style consistency."}

    abs_z, deltas, flags = [], {}, []
    for key in FEATURE_KEYS:
        vals = [h[key] for h in history_fps if h.get(key) is not None]
        new_val = new_fp.get(key)
        if new_val is None or len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        z = (new_val - mean) / std if std > 1e-9 else 0.0
        deltas[key] = {"new": round(new_val, 3), "mean": round(mean, 3),
                       "std": round(std, 3), "z": round(z, 2)}
        abs_z.append(abs(z))
        if abs(z) >= 2.0:
            flags.append({"feature": key, "z": round(z, 2)})

    if not abs_z:
        return {"consistency": None, "inconsistency_score": None,
                "n_history": len(history_fps), "flags": [], "deltas": deltas,
                "note": "Insufficient overlapping features."}

    mean_abs_z = sum(abs_z) / len(abs_z)
    consistency = round(100.0 * math.exp(-mean_abs_z / 2.0), 2)   # 0=high z, 100=identical
    return {
        "consistency": consistency,
        "inconsistency_score": round(100.0 - consistency, 2),
        "mean_abs_z": round(mean_abs_z, 3),
        "n_history": len(history_fps),
        "flags": sorted(flags, key=lambda f: -abs(f["z"])),
        "deltas": deltas,
    }


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else (
        "This is a test. It has a few short sentences! And one that is, perhaps, "
        "a little longer than the others; just to add some variation to the mix.")
    from pprint import pprint
    pprint(extract_fingerprint(s))

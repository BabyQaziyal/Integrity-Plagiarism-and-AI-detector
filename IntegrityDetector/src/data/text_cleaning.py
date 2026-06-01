"""Text normalisation, sentence/chunk splitting, and hashing.

Shared by the data pipeline (cleaning + dedup) and the detection core
(chunking for plagiarism, sentence splitting for burstiness). Deliberately
dependency-free (stdlib only) so it is fast and import-cheap everywhere.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WS    = re.compile(r"\s+")
_CTRL  = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_WORD  = re.compile(r"\b\w+\b")
_TOKEN = re.compile(r"\S+")
# Sentence boundary: terminal punctuation + whitespace + an opener (capital/quote/digit).
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def normalize_text(text) -> str:
    """NFKC-normalise, strip control chars, collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def split_sentences(text: str) -> list[str]:
    """Lightweight, offline sentence splitter (no nltk download required)."""
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def split_chunks(text: str, chunk_words: int, stride: int) -> list[str]:
    """Sliding-window word chunks for plagiarism / source comparison."""
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, stride)
    chunks = [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words) - chunk_words + 1, step)
    ]
    # Always include the tail window so the document end isn't dropped.
    if (len(words) - chunk_words) % step != 0:
        chunks.append(" ".join(words[-chunk_words:]))
    return chunks


def word_spans(text: str):
    """List of (word, start_char, end_char) over the ORIGINAL text."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text or "")]


def chunk_spans(text: str, chunk_words: int, stride: int):
    """Sliding-window chunks as (chunk_text, start_char, end_char) on the original
    text — used to map plagiarism matches back to highlightable character ranges."""
    spans = word_spans(text)
    if not spans:
        return []
    if len(spans) <= chunk_words:
        return [(text[spans[0][1]:spans[-1][2]], spans[0][1], spans[-1][2])]
    step = max(1, stride)
    out = []
    for i in range(0, len(spans) - chunk_words + 1, step):
        win = spans[i:i + chunk_words]
        out.append((text[win[0][1]:win[-1][2]], win[0][1], win[-1][2]))
    if (len(spans) - chunk_words) % step != 0:
        win = spans[-chunk_words:]
        out.append((text[win[0][1]:win[-1][2]], win[0][1], win[-1][2]))
    return out


def text_hash(text: str) -> str:
    """Stable hash of normalised, case-folded text — used for dedup."""
    norm = normalize_text(text).casefold()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

"""plagiarism.py — TF-IDF + cosine, chunk-based plagiarism detection.

The detector compares a student submission against a reference ("web-like")
corpus using a character/word TF-IDF model and cosine similarity over sliding
text chunks. Because TF-IDF vectors are L2-normalised, cosine similarity is a
single sparse matrix product, so scoring a document is fast even against a few
thousand corpus chunks.

Pipeline:
  fit:    corpus docs -> sliding chunks -> TfidfVectorizer -> sparse matrix (cached)
  query:  submission  -> sliding chunks -> transform -> cosine vs corpus matrix
          -> best match per chunk -> flag >= threshold
  score:  plagiarism % = share of the document's words covered by flagged chunks
          (union of word ranges, so overlapping windows aren't double-counted)

Build the index once:  python -m src.detection.plagiarism --build
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.config import (CHUNK_STRIDE, CHUNK_WORDS, CHECKPOINTS_DIR, FINAL_DATASET,
                        PLAGIARISM_SIM_THRESHOLD, SOURCE_CORPUS_FILES,
                        TFIDF_MAX_FEATURES, TFIDF_NGRAM)
from src.data.text_cleaning import chunk_spans, normalize_text, word_count
from src.logging_utils import get_logger

log = get_logger("plagiarism")

INDEX_PATH = CHECKPOINTS_DIR / "plagiarism_index.joblib"


# --------------------------------------------------------------------------- #
# Result data structures
# --------------------------------------------------------------------------- #
@dataclass
class ChunkMatch:
    chunk_index: int
    start_char: int
    end_char: int
    score: float
    source_id: str
    source_title: str
    source_snippet: str
    text: str = ""


@dataclass
class PlagiarismResult:
    plagiarism_percent: float
    matched_chunks: list = field(default_factory=list)   # list[ChunkMatch] (flagged)
    matched_sources: list = field(default_factory=list)  # aggregated per source
    highlights: list = field(default_factory=list)       # [{start,end,score}] for UI
    chunk_count: int = 0
    flagged_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------- #
# Corpus index
# --------------------------------------------------------------------------- #
class CorpusIndex:
    """Holds the fitted TF-IDF vectorizer, the corpus chunk matrix, and metadata."""

    def __init__(self, vectorizer=None, matrix=None, chunk_meta=None):
        self.vectorizer = vectorizer
        self.matrix = matrix                 # scipy sparse (n_chunks x vocab), L2-normalised
        self.chunk_meta = chunk_meta or []   # list[dict(doc_id,title,text)]

    # ---- build -------------------------------------------------------------
    @classmethod
    def build(cls, corpus_docs: list[tuple], chunk_words: int = CHUNK_WORDS,
              stride: int = CHUNK_STRIDE) -> "CorpusIndex":
        """corpus_docs: list of (doc_id, title, text)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        chunk_texts, chunk_meta = [], []
        for doc_id, title, text in corpus_docs:
            for ctext, _s, _e in chunk_spans(text, chunk_words, stride):
                norm = normalize_text(ctext)
                if word_count(norm) < 8:     # skip trivially short corpus chunks
                    continue
                chunk_texts.append(norm)
                chunk_meta.append({"doc_id": str(doc_id), "title": str(title),
                                   "text": norm})
        if not chunk_texts:
            raise ValueError("Corpus produced no usable chunks.")

        vec = TfidfVectorizer(lowercase=True, ngram_range=TFIDF_NGRAM,
                              max_features=TFIDF_MAX_FEATURES, sublinear_tf=True,
                              norm="l2", strip_accents="unicode")
        matrix = vec.fit_transform(chunk_texts)
        log.info("Built corpus index: %d chunks, vocab=%d", matrix.shape[0], matrix.shape[1])
        return cls(vec, matrix, chunk_meta)

    # ---- persistence -------------------------------------------------------
    def save(self, path: Path = INDEX_PATH) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix,
                     "chunk_meta": self.chunk_meta}, path)
        log.info("Saved plagiarism index -> %s", path)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "CorpusIndex":
        import joblib
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Plagiarism index not found at {path}. Build it first: "
                f"python -m src.detection.plagiarism --build")
        d = joblib.load(path)
        return cls(d["vectorizer"], d["matrix"], d["chunk_meta"])

    # ---- query -------------------------------------------------------------
    def topk_for_text(self, text: str, k: int = 3):
        """Return up to k (score, meta) matches for a single text chunk."""
        q = self.vectorizer.transform([normalize_text(text)])
        sims = (q @ self.matrix.T).toarray().ravel()      # cosine (L2-normalised)
        if sims.size == 0:
            return []
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunk_meta[i]) for i in idx]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
_INDEX_CACHE: CorpusIndex | None = None


def get_index() -> CorpusIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = CorpusIndex.load()
    return _INDEX_CACHE


def detect_plagiarism(text: str, index: CorpusIndex | None = None,
                      threshold: float = PLAGIARISM_SIM_THRESHOLD,
                      chunk_words: int = CHUNK_WORDS,
                      stride: int = CHUNK_STRIDE) -> PlagiarismResult:
    """Score a submission against the reference corpus."""
    index = index or get_index()
    text = text or ""
    spans = chunk_spans(text, chunk_words, stride)
    total_words = max(1, word_count(text))

    if not spans:
        return PlagiarismResult(0.0)

    chunk_norms = [normalize_text(c) for c, _, _ in spans]
    q = index.vectorizer.transform(chunk_norms)            # (n_chunks x vocab)
    sims = (q @ index.matrix.T)                            # (n_chunks x n_corpus)
    sims = sims.toarray() if hasattr(sims, "toarray") else np.asarray(sims)
    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    matched, highlights = [], []
    source_hits: dict[str, dict] = {}
    flagged_word_ranges: list[tuple[int, int]] = []

    for ci, ((ctext, s_char, e_char), bidx, score) in enumerate(zip(spans, best_idx, best_score)):
        if score < threshold:
            continue
        meta = index.chunk_meta[int(bidx)]
        matched.append(ChunkMatch(
            chunk_index=ci, start_char=int(s_char), end_char=int(e_char),
            score=round(float(score), 4), source_id=meta["doc_id"],
            source_title=meta["title"], source_snippet=meta["text"][:240],
            text=ctext[:240]))
        highlights.append({"start": int(s_char), "end": int(e_char),
                           "score": round(float(score), 4)})
        # track word range (approx by char-proportional words) for coverage
        flagged_word_ranges.append((s_char, e_char))
        agg = source_hits.setdefault(meta["doc_id"], {
            "source_id": meta["doc_id"], "title": meta["title"],
            "hits": 0, "max_score": 0.0})
        agg["hits"] += 1
        agg["max_score"] = max(agg["max_score"], round(float(score), 4))

    # coverage % by union of flagged CHARACTER ranges (no double counting)
    covered_chars = _union_length(flagged_word_ranges)
    total_chars = max(1, len(text))
    plagiarism_percent = round(100.0 * covered_chars / total_chars, 2)

    matched_sources = sorted(source_hits.values(), key=lambda x: -x["max_score"])

    return PlagiarismResult(
        plagiarism_percent=min(plagiarism_percent, 100.0),
        matched_chunks=[asdict(m) for m in matched],
        matched_sources=matched_sources,
        highlights=_merge_intervals(highlights),
        chunk_count=len(spans),
        flagged_count=len(matched),
    )


def _union_length(ranges: list[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    ranges = sorted(ranges)
    total, cur_s, cur_e = 0, ranges[0][0], ranges[0][1]
    for s, e in ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _merge_intervals(highlights: list[dict]) -> list[dict]:
    """Merge overlapping highlight spans, keeping the max score."""
    if not highlights:
        return []
    hs = sorted(highlights, key=lambda h: h["start"])
    merged = [dict(hs[0])]
    for h in hs[1:]:
        last = merged[-1]
        if h["start"] <= last["end"]:
            last["end"] = max(last["end"], h["end"])
            last["score"] = max(last["score"], h["score"])
        else:
            merged.append(dict(h))
    return merged


# --------------------------------------------------------------------------- #
# Default corpus assembly + build entry point
# --------------------------------------------------------------------------- #
def default_corpus(include_human_samples: bool = True,
                   max_human: int = 4000) -> list[tuple]:
    """Assemble a web-like reference corpus.

    Primary: the long ``source_text`` passages from train_prompts.csv (the
    original material students read). Optional: a capped sample of human-labeled
    essays from final_dataset.csv as additional reference documents.
    """
    docs: list[tuple] = []
    for f in SOURCE_CORPUS_FILES:
        f = Path(f)
        if not f.exists():
            continue
        df = pd.read_csv(f)
        text_col = "source_text" if "source_text" in df.columns else None
        title_col = "prompt_name" if "prompt_name" in df.columns else None
        id_col = "prompt_id" if "prompt_id" in df.columns else None
        if not text_col:
            continue
        for i, row in df.iterrows():
            txt = normalize_text(row.get(text_col))
            if word_count(txt) >= 40:
                docs.append((row.get(id_col, f"src_{i}"),
                             row.get(title_col, f"source_{i}"), txt))
    log.info("Source corpus passages: %d", len(docs))

    if include_human_samples and FINAL_DATASET.exists():
        df = pd.read_csv(FINAL_DATASET, usecols=["text", "label", "source"])
        human = df[df["label"] == 0]
        if len(human) > max_human:
            human = human.sample(n=max_human, random_state=config.SEED)
        for i, row in human.reset_index(drop=True).iterrows():
            docs.append((f"corpus_human_{i}", f"reference_essay_{row['source']}",
                         normalize_text(row["text"])))
        log.info("Added %d human reference essays (web-like corpus)", len(human))
    return docs


def build_index(include_human_samples: bool = True) -> CorpusIndex:
    config.ensure_dirs()
    docs = default_corpus(include_human_samples=include_human_samples)
    if not docs:
        raise ValueError("No corpus documents found. Ensure train_prompts.csv "
                         "exists in models/ or build final_dataset.csv first.")
    index = CorpusIndex.build(docs)
    index.save()
    return index


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plagiarism corpus index tools.")
    p.add_argument("--build", action="store_true", help="build + save the corpus index")
    p.add_argument("--no-human", action="store_true",
                   help="exclude human essays from final_dataset from the corpus")
    p.add_argument("--demo", type=str, help="score a string against the index")
    a = p.parse_args()
    if a.build:
        build_index(include_human_samples=not a.no_human)
    if a.demo:
        res = detect_plagiarism(a.demo)
        log.info("Plagiarism: %.2f%% | flagged %d/%d chunks | sources=%s",
                 res.plagiarism_percent, res.flagged_count, res.chunk_count,
                 [s["title"] for s in res.matched_sources[:3]])

"""source_attribution.py — match suspicious chunks to their closest sources.

Builds on the plagiarism CorpusIndex: for each (optionally only flagged) chunk of
a submission, returns the top-k closest reference passages with similarity scores
and snippet evidence. Also rolls up a document-level "matched sources" table for
the integrity report.
"""
from __future__ import annotations

from src.config import CHUNK_STRIDE, CHUNK_WORDS, PLAGIARISM_SIM_THRESHOLD
from src.data.text_cleaning import chunk_spans, normalize_text
from src.detection.plagiarism import CorpusIndex, get_index
from src.logging_utils import get_logger

log = get_logger("attribution")


def attribute_chunks(text: str, index: CorpusIndex | None = None, top_k: int = 3,
                     only_above: float = PLAGIARISM_SIM_THRESHOLD,
                     chunk_words: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE) -> list[dict]:
    """For each chunk whose best match >= ``only_above``, list its top-k sources."""
    index = index or get_index()
    spans = chunk_spans(text or "", chunk_words, stride)
    results = []
    for ci, (ctext, s_char, e_char) in enumerate(spans):
        matches = index.topk_for_text(ctext, k=top_k)
        if not matches or matches[0][0] < only_above:
            continue
        results.append({
            "chunk_index": ci,
            "start_char": int(s_char),
            "end_char": int(e_char),
            "excerpt": ctext[:240],
            "matches": [
                {"source_id": m["doc_id"], "title": m["title"],
                 "score": round(score, 4), "snippet": m["text"][:240]}
                for score, m in matches
            ],
        })
    return results


def aggregate_sources(attributions: list[dict], limit: int = 10) -> list[dict]:
    """Roll chunk-level attributions into a per-source evidence table."""
    by_source: dict[str, dict] = {}
    for attr in attributions:
        top = attr["matches"][0]
        sid = top["source_id"]
        agg = by_source.setdefault(sid, {
            "source_id": sid, "title": top["title"],
            "max_score": 0.0, "hits": 0, "example": top["snippet"]})
        agg["hits"] += 1
        if top["score"] > agg["max_score"]:
            agg["max_score"] = top["score"]
            agg["example"] = top["snippet"]
    ranked = sorted(by_source.values(), key=lambda x: (-x["max_score"], -x["hits"]))
    return ranked[:limit]


def attribute(text: str, index: CorpusIndex | None = None, top_k: int = 3) -> dict:
    """Convenience: per-chunk attributions + aggregated source table."""
    index = index or get_index()
    chunks = attribute_chunks(text, index=index, top_k=top_k)
    return {"chunk_attributions": chunks, "sources": aggregate_sources(chunks)}

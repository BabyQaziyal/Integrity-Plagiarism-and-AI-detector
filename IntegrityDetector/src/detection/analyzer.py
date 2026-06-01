"""analyzer.py — orchestrate every detector into one integrity result.

This is the single entry point the Flask service calls. It runs plagiarism,
source attribution, the AI-content detectors (classifier + perplexity +
burstiness), and stylometry, then aggregates an AI % and an integrity score.

Every component is wrapped so one failure (e.g. the plagiarism index not built
yet, or the GPU model unavailable) degrades gracefully instead of failing the
whole analysis — failures are reported under ``meta.errors``.
"""
from __future__ import annotations

from src.config import AI_WEIGHTS, MIN_RELIABLE_WORDS
from src.data.text_cleaning import normalize_text, word_count
from src.detection.aggregator import combine_ai_score, integrity_score, verdict
from src.logging_utils import get_logger

log = get_logger("analyzer")


def analyze_submission(text: str,
                       student_history_fingerprints: list | None = None,
                       run_classifier: bool = True,
                       run_perplexity: bool = True,
                       run_plagiarism: bool = True,
                       run_stylometry: bool = True) -> dict:
    text = normalize_text(text)
    errors: dict[str, str] = {}
    result: dict = {"meta": {"word_count": word_count(text), "char_count": len(text)}}

    # ---- Plagiarism + source attribution ---------------------------------
    plag = {"plagiarism_percent": 0.0, "matched_sources": [], "highlights": [],
            "matched_chunks": [], "flagged_count": 0, "chunk_count": 0}
    attribution = {"sources": [], "chunk_attributions": []}
    if run_plagiarism:
        try:
            from src.detection.plagiarism import detect_plagiarism, get_index
            from src.detection.source_attribution import attribute
            index = get_index()
            plag = detect_plagiarism(text, index=index).to_dict()
            attribution = attribute(text, index=index)
        except FileNotFoundError as exc:
            errors["plagiarism"] = str(exc)
            log.warning("Plagiarism index missing: %s", exc)
        except Exception as exc:  # noqa: BLE001
            errors["plagiarism"] = f"{type(exc).__name__}: {exc}"
            log.exception("Plagiarism detection failed")
    result["plagiarism"] = plag
    result["source_attribution"] = attribution

    # ---- AI content: classifier + perplexity + burstiness -----------------
    classifier_p = perplexity_ai = burstiness_ai = None
    classifier_info = perplexity_info = burstiness_info = None

    try:
        from src.detection.burstiness import analyze_burstiness
        burstiness_info = analyze_burstiness(text)
        burstiness_ai = burstiness_info["ai_likelihood"]
    except Exception as exc:  # noqa: BLE001
        errors["burstiness"] = f"{type(exc).__name__}: {exc}"

    if run_classifier:
        try:
            from src.detection.ai_detector import analyze_ai_classifier
            classifier_info = analyze_ai_classifier(text)
            if classifier_info.get("is_trained"):
                classifier_p = classifier_info["p_ai"]
            else:
                errors["classifier"] = "model not fine-tuned (signal ignored)"
        except Exception as exc:  # noqa: BLE001
            errors["classifier"] = f"{type(exc).__name__}: {exc}"

    if run_perplexity:
        try:
            from src.detection.perplexity import analyze_perplexity
            perplexity_info = analyze_perplexity(text)
            perplexity_ai = perplexity_info["ai_likelihood"]
        except Exception as exc:  # noqa: BLE001
            errors["perplexity"] = f"{type(exc).__name__}: {exc}"

    ai = combine_ai_score(classifier_p, perplexity_ai, burstiness_ai, AI_WEIGHTS)
    ai["classifier"] = classifier_info
    ai["perplexity"] = perplexity_info
    ai["burstiness"] = burstiness_info

    # Reliability gate: very short input, or no/low-confidence signals, must not
    # produce a confident AI accusation. We keep the computed numbers for
    # transparency but flag low confidence so the verdict can abstain.
    wc = result["meta"]["word_count"]
    overall_conf = ai.get("overall_confidence", 0.0)
    ai["low_confidence"] = bool(wc < MIN_RELIABLE_WORDS or classifier_p is None
                                or overall_conf < 0.2)
    if ai["low_confidence"]:
        ai["note"] = ("Not enough text (or signal) to assess AI authorship "
                      "reliably — treat the AI score as indicative only.")
    result["ai_content"] = ai

    # ---- Stylometry + history --------------------------------------------
    if run_stylometry:
        try:
            from src.detection.stylometry import compare_to_history, extract_fingerprint
            fp = extract_fingerprint(text)
            hist = compare_to_history(fp, student_history_fingerprints or [])
            result["stylometry"] = {"fingerprint": fp, "history": hist}
        except Exception as exc:  # noqa: BLE001
            errors["stylometry"] = f"{type(exc).__name__}: {exc}"
            result["stylometry"] = {"fingerprint": {}, "history": {}}
    else:
        result["stylometry"] = {"fingerprint": {}, "history": {}}

    # ---- Aggregate integrity score ----------------------------------------
    plag_pct = result["plagiarism"].get("plagiarism_percent", 0.0)
    ai_pct = result["ai_content"].get("ai_percent", 0.0)
    score = integrity_score(plag_pct, ai_pct)
    result["integrity_score"] = score
    # If the AI signal is low-confidence and there's no plagiarism evidence,
    # don't escalate to a risk verdict on the strength of a shaky AI guess.
    if result["ai_content"].get("low_confidence") and plag_pct < 15:
        result["verdict"] = "Insufficient text to assess"
    else:
        result["verdict"] = verdict(score)
    result["meta"]["errors"] = errors
    return result


if __name__ == "__main__":
    import json
    import sys
    sample = sys.argv[1] if len(sys.argv) > 1 else (
        "It is important to note that artificial intelligence is very important. "
        "It helps many people. It is used in many areas. It is a useful tool.")
    out = analyze_submission(sample, run_classifier=True, run_perplexity=False)
    print(json.dumps({k: v for k, v in out.items() if k != "source_attribution"},
                     indent=2, default=str)[:2000])

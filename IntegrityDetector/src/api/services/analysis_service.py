"""analysis_service.py — run the detector stack and persist results.

Bridges the detection core (src.detection.analyzer) and the database: builds the
student's writing-history fingerprints, runs the analysis with the configured
detectors, and stores a Submission + AnalysisResult.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src import config
from src.api.models import AnalysisResult, Student, Submission
from src.data.text_cleaning import word_count
from src.detection.analyzer import analyze_submission
from src.logging_utils import get_logger

log = get_logger("analysis_service")


def student_history_fingerprints(db: Session, student_id: int,
                                 exclude_submission_id: int | None = None) -> list[dict]:
    """Collect stylometry fingerprints from a student's prior analyses."""
    stmt = (select(AnalysisResult.fingerprint_json)
            .join(Submission, Submission.id == AnalysisResult.submission_id)
            .where(Submission.student_id == student_id))
    if exclude_submission_id is not None:
        stmt = stmt.where(Submission.id != exclude_submission_id)
    return [fp for (fp,) in db.execute(stmt).all() if fp]


def analyze_and_store(db: Session, student: Student, text: str, title: str = "Untitled",
                      filename: str | None = None, stored_path: str | None = None,
                      course: str | None = None, assignment: str | None = None) -> Submission:
    """Create a submission, analyse it against the student's history, persist."""
    submission = Submission(
        student_id=student.id, title=title or "Untitled", filename=filename,
        stored_path=stored_path, course=course, assignment=assignment,
        text=text, word_count=word_count(text))
    db.add(submission)
    db.flush()                      # assign submission.id before history lookup

    history = student_history_fingerprints(db, student.id,
                                           exclude_submission_id=submission.id)
    log.info("Analysing submission %s (student=%s) with %d history fingerprints",
             submission.id, student.id, len(history))

    result = analyze_submission(
        text,
        student_history_fingerprints=history,
        run_classifier=config.API_RUN_CLASSIFIER,
        run_perplexity=config.API_RUN_PERPLEXITY,
        run_plagiarism=config.API_RUN_PLAGIARISM,
        run_stylometry=config.API_RUN_STYLOMETRY,
    )

    fingerprint = result.get("stylometry", {}).get("fingerprint", {})
    analysis = AnalysisResult(
        submission_id=submission.id,
        plagiarism_percent=result["plagiarism"].get("plagiarism_percent", 0.0),
        ai_percent=result["ai_content"].get("ai_percent", 0.0),
        integrity_score=result.get("integrity_score", 100.0),
        verdict=result.get("verdict", "Low risk"),
        result_json=result,
        fingerprint_json=fingerprint,
    )
    db.add(analysis)
    db.flush()
    submission.analysis = analysis
    return submission

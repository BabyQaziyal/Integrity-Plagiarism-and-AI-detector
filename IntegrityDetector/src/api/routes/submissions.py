"""/api/submissions — upload, analyse, fetch results, original file, PDF report."""
from __future__ import annotations

import uuid
from pathlib import Path

from flask import Blueprint, g, jsonify, request, send_file
from sqlalchemy import select
from werkzeug.utils import secure_filename

from src.config import ALLOWED_UPLOAD_EXT, UPLOAD_DIR
from src.api.models import Student, Submission
from src.api.services.analysis_service import analyze_and_store
from src.api.services.extract import extract_text

bp = Blueprint("submissions", __name__, url_prefix="/api/submissions")


def _get_student_or_404(student_id):
    if student_id is None:
        return None, (jsonify({"error": "student_id is required"}), 400)
    student = g.db.get(Student, int(student_id))
    if not student:
        return None, (jsonify({"error": "student not found"}), 404)
    return student, None


@bp.post("")
def create_submission():
    """Accept either a multipart file upload or a raw-text JSON body."""
    if request.files.get("file"):
        return _create_from_file()
    return _create_from_json()


def _create_from_json():
    data = request.get_json(silent=True) or {}
    student, err = _get_student_or_404(data.get("student_id"))
    if err:
        return err
    text = (data.get("text") or "").strip()
    if len(text) < 20:
        return jsonify({"error": "text is too short to analyse (min 20 chars)"}), 400
    sub = analyze_and_store(g.db, student, text=text,
                            title=data.get("title", "Untitled"),
                            course=data.get("course"),
                            assignment=data.get("assignment"))
    return jsonify(_full(sub)), 201


def _create_from_file():
    f = request.files["file"]
    student, err = _get_student_or_404(request.form.get("student_id"))
    if err:
        return err
    filename = secure_filename(f.filename or "upload")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXT:
        return jsonify({"error": f"unsupported type '{ext}'",
                        "allowed": sorted(ALLOWED_UPLOAD_EXT)}), 400
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stored = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    f.save(str(stored))
    try:
        text = extract_text(stored, filename)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"could not extract text: {exc}"}), 422
    if len((text or "").strip()) < 20:
        return jsonify({"error": "no extractable text found in file"}), 422
    sub = analyze_and_store(g.db, student, text=text,
                            title=request.form.get("title", filename),
                            filename=filename, stored_path=str(stored),
                            course=request.form.get("course"),
                            assignment=request.form.get("assignment"))
    return jsonify(_full(sub)), 201


@bp.get("")
def list_submissions():
    stmt = select(Submission).order_by(Submission.created_at.desc())
    sid = request.args.get("student_id")
    if sid:
        stmt = stmt.where(Submission.student_id == int(sid))
    rows = g.db.execute(stmt).scalars().all()
    return jsonify([s.to_dict() for s in rows])


@bp.get("/<int:submission_id>")
def get_submission(submission_id: int):
    sub = g.db.get(Submission, submission_id)
    if not sub:
        return jsonify({"error": "submission not found"}), 404
    return jsonify(_full(sub))


@bp.get("/<int:submission_id>/result")
def get_result(submission_id: int):
    sub = g.db.get(Submission, submission_id)
    if not sub or not sub.analysis:
        return jsonify({"error": "analysis not found"}), 404
    return jsonify(sub.analysis.to_dict())


@bp.get("/<int:submission_id>/file")
def get_file(submission_id: int):
    sub = g.db.get(Submission, submission_id)
    if not sub or not sub.stored_path or not Path(sub.stored_path).exists():
        return jsonify({"error": "original file not available"}), 404
    return send_file(sub.stored_path, download_name=sub.filename or "submission",
                     as_attachment=False)


@bp.get("/<int:submission_id>/report")
def get_report(submission_id: int):
    sub = g.db.get(Submission, submission_id)
    if not sub or not sub.analysis:
        return jsonify({"error": "analysis not found"}), 404
    from src.reports.pdf_report import build_report  # lazy (reportlab)
    pdf_path = build_report(sub)
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"integrity_report_submission_{sub.id}.pdf")


def _full(sub: Submission) -> dict:
    d = sub.to_dict(include_text=True)
    d["student"] = sub.student.to_dict(with_counts=False) if sub.student else None
    d["analysis"] = sub.analysis.to_dict() if sub.analysis else None
    return d

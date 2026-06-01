"""/api/students — manage students and view their submission history."""
from __future__ import annotations

from flask import Blueprint, g, jsonify, request
from sqlalchemy import select

from src.api.models import Student, Submission

bp = Blueprint("students", __name__, url_prefix="/api/students")


@bp.post("")
def create_student():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    student = Student(name=name, email=data.get("email"),
                      external_id=data.get("external_id"))
    g.db.add(student)
    g.db.flush()
    return jsonify(student.to_dict()), 201


@bp.get("")
def list_students():
    rows = g.db.execute(select(Student).order_by(Student.name)).scalars().all()
    return jsonify([s.to_dict() for s in rows])


@bp.get("/<int:student_id>")
def get_student(student_id: int):
    student = g.db.get(Student, student_id)
    if not student:
        return jsonify({"error": "student not found"}), 404
    subs = (g.db.execute(select(Submission)
                         .where(Submission.student_id == student_id)
                         .order_by(Submission.created_at.desc()))
            .scalars().all())
    return jsonify({**student.to_dict(),
                    "submissions": [s.to_dict() for s in subs]})

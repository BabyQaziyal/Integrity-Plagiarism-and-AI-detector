"""SQLAlchemy ORM models: Student, Submission, AnalysisResult."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from src.api.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="student", cascade="all, delete-orphan")

    def to_dict(self, with_counts: bool = True) -> dict:
        d = {"id": self.id, "name": self.name, "email": self.email,
             "external_id": self.external_id,
             "created_at": self.created_at.isoformat() if self.created_at else None}
        if with_counts:
            d["submission_count"] = len(self.submissions)
        return d


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="Untitled")
    course: Mapped[str | None] = mapped_column(String(200), nullable=True)
    assignment: Mapped[str | None] = mapped_column(String(200), nullable=True)
    filename: Mapped[str | None] = mapped_column(String(400), nullable=True)
    stored_path: Mapped[str | None] = mapped_column(String(600), nullable=True)
    text: Mapped[str] = mapped_column(Text, default="")
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    student: Mapped["Student"] = relationship(back_populates="submissions")
    analysis: Mapped["AnalysisResult | None"] = relationship(
        back_populates="submission", uselist=False, cascade="all, delete-orphan")

    def to_dict(self, include_text: bool = False) -> dict:
        d = {"id": self.id, "student_id": self.student_id, "title": self.title,
             "course": self.course, "assignment": self.assignment,
             "filename": self.filename, "word_count": self.word_count,
             "created_at": self.created_at.isoformat() if self.created_at else None,
             "has_analysis": self.analysis is not None}
        if include_text:
            d["text"] = self.text
        if self.analysis is not None:
            d["summary"] = self.analysis.summary()
        return d


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id"), unique=True, index=True)
    plagiarism_percent: Mapped[float] = mapped_column(Float, default=0.0)
    ai_percent: Mapped[float] = mapped_column(Float, default=0.0)
    integrity_score: Mapped[float] = mapped_column(Float, default=100.0)
    verdict: Mapped[str] = mapped_column(String(40), default="Low risk")
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)        # full analyzer output
    fingerprint_json: Mapped[dict] = mapped_column(JSON, default=dict)   # stylometry fingerprint
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    submission: Mapped["Submission"] = relationship(back_populates="analysis")

    def summary(self) -> dict:
        return {"plagiarism_percent": self.plagiarism_percent,
                "ai_percent": self.ai_percent,
                "integrity_score": self.integrity_score,
                "verdict": self.verdict}

    def to_dict(self) -> dict:
        return {"id": self.id, "submission_id": self.submission_id,
                **self.summary(), "result": self.result_json,
                "created_at": self.created_at.isoformat() if self.created_at else None}

"""Extract plain text from uploaded files (PDF / DOCX / TXT / MD)."""
from __future__ import annotations

from pathlib import Path

from src.config import ALLOWED_UPLOAD_EXT
from src.data.text_cleaning import normalize_text
from src.logging_utils import get_logger

log = get_logger("extract")


def extract_text(path: str | Path, filename: str | None = None) -> str:
    path = Path(path)
    ext = Path(filename or path).suffix.lower()

    if ext == ".pdf":
        return _from_pdf(path)
    if ext == ".docx":
        return _from_docx(path)
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_UPLOAD_EXT)}")


def _from_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return normalize_text("\n".join(pages))


def _from_docx(path: Path) -> str:
    import docx
    document = docx.Document(str(path))
    return normalize_text("\n".join(p.text for p in document.paragraphs))

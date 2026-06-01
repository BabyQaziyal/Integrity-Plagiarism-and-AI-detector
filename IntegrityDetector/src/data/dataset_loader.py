"""dataset_loader.py — load + normalise every raw labeled dataset.

Scans ``RAW_DATA_DIRS`` for CSV / JSON / JSONL, maps each heterogeneous schema to
the canonical ``[text, label, source]`` form (0 = human, 1 = AI), drops empty /
short / unlabeled rows, removes duplicates, shuffles, and logs statistics.

Memory is bounded with *per-file reservoir sampling*: the big Kaggle files
(DAIGT v2 ~460k, Training_Essay ~255k) are also SORTED BY LABEL on disk, so a
uniform random reservoir both caps RAM and mixes classes — never head-truncate.

Run standalone to inspect:  python -m src.data.dataset_loader
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from src import config
from src.config import (
    LABEL_COLUMN_CANDIDATES, MAX_SAMPLES_PER_SOURCE, MAX_TEXT_CHARS,
    MIN_TEXT_CHARS, MIN_TEXT_WORDS, RAW_DATA_DIRS, RAW_SCAN_SKIP, SEED,
    SOURCE_COLUMN_CANDIDATES, TEXT_COLUMN_CANDIDATES,
)
from src.data.text_cleaning import normalize_text, text_hash
from src.logging_utils import get_logger

log = get_logger("loader")

_CHUNK = 20_000          # rows per CSV read chunk


# --------------------------------------------------------------------------- #
# Schema detection / label normalisation
# --------------------------------------------------------------------------- #
def _pick(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalize_label(value):
    """Map heterogeneous label values to {0, 1}; return None if unparseable."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "1.0", "ai", "ai-generated", "generated", "fake", "machine",
             "gpt", "llm", "true", "yes"}:
        return 1
    if s in {"0", "0.0", "human", "real", "original", "student", "false", "no"}:
        return 0
    try:
        f = float(s)
        if f == 1.0:
            return 1
        if f == 0.0:
            return 0
    except (TypeError, ValueError):
        pass
    return None


def discover_files() -> list[Path]:
    """All candidate raw dataset files across RAW_DATA_DIRS (deduped by path)."""
    files, seen = [], set()
    for d in RAW_DATA_DIRS:
        d = Path(d)
        if not d.exists():
            continue
        for pattern in ("*.csv", "*.json", "*.jsonl"):
            for p in sorted(d.glob(pattern)):
                rp = p.resolve()
                if p.name in RAW_SCAN_SKIP or rp in seen:
                    continue
                seen.add(rp)
                files.append(p)
    return files


# --------------------------------------------------------------------------- #
# Per-file loading (vectorised filter + reservoir sampling)
# --------------------------------------------------------------------------- #
def _clean_frame(frame: pd.DataFrame, text_col, label_col, source_col,
                 default_source: str) -> pd.DataFrame:
    """Vectorised normalise + filter of a raw chunk → [text, label, source]."""
    out = pd.DataFrame()
    out["text"] = frame[text_col].map(normalize_text)
    out["label"] = frame[label_col].map(normalize_label)
    out["source"] = (frame[source_col].astype(str)
                     if source_col and source_col in frame
                     else default_source)
    out = out[out["label"].notna()]
    if out.empty:
        return out
    lengths = out["text"].str.len()
    words = out["text"].str.count(r"\b\w+\b")
    out = out[(lengths >= MIN_TEXT_CHARS) & (words >= MIN_TEXT_WORDS)]
    out["text"] = out["text"].str.slice(0, MAX_TEXT_CHARS)
    out["label"] = out["label"].astype(int)
    out["source"] = out["source"].fillna(default_source).replace("", default_source)
    return out


class _Reservoir:
    """Uniform random sample of at most ``cap`` items (Algorithm R)."""

    def __init__(self, cap: int, rng: random.Random):
        self.cap, self.rng = cap, rng
        self.items: list[tuple] = []
        self.seen = 0

    def add(self, item) -> None:
        if len(self.items) < self.cap:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.seen)
            if j < self.cap:
                self.items[j] = item
        self.seen += 1


def _iter_frames(path: Path):
    """Yield (frame, text_col, label_col, source_col). CSV is chunked; JSON whole."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        head = pd.read_csv(path, nrows=0)
        tc = _pick(head.columns, TEXT_COLUMN_CANDIDATES)
        lc = _pick(head.columns, LABEL_COLUMN_CANDIDATES)
        sc = _pick(head.columns, SOURCE_COLUMN_CANDIDATES)
        if not tc or not lc:
            log.warning("  skip %-28s (need text+label, got %s)", path.name, list(head.columns))
            return
        usecols = [c for c in {tc, lc, sc} if c]
        reader = pd.read_csv(path, usecols=usecols, chunksize=_CHUNK, dtype=str,
                             keep_default_na=False, on_bad_lines="skip")
        for chunk in reader:
            yield chunk, tc, lc, sc
    else:  # .json / .jsonl
        lines = suffix == ".jsonl"
        df = pd.read_json(path, lines=lines, dtype=str)
        tc = _pick(df.columns, TEXT_COLUMN_CANDIDATES)
        lc = _pick(df.columns, LABEL_COLUMN_CANDIDATES)
        sc = _pick(df.columns, SOURCE_COLUMN_CANDIDATES)
        if not tc or not lc:
            log.warning("  skip %-28s (need text+label, got %s)", path.name, list(df.columns))
            return
        yield df, tc, lc, sc


def load_file(path: Path, cap: int = MAX_SAMPLES_PER_SOURCE,
              rng: random.Random | None = None) -> list[tuple]:
    """Load one raw file → list of (text, label, source), capped via reservoir."""
    rng = rng or random.Random(SEED)
    reservoir = _Reservoir(cap, rng)
    pos = neg = 0
    for frame, tc, lc, sc in _iter_frames(path):
        cleaned = _clean_frame(frame, tc, lc, sc, default_source=path.stem)
        for text, label, source in zip(cleaned["text"], cleaned["label"], cleaned["source"]):
            reservoir.add((text, label, source))
            pos += label == 1
            neg += label == 0
    log.info("  %-28s valid=%-7d (human=%d ai=%d) -> sampled %d",
             path.name, reservoir.seen, neg, pos, len(reservoir.items))
    return reservoir.items


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def _log_stats(df: pd.DataFrame, tag: str) -> None:
    counts = df["label"].value_counts().to_dict()
    by_src = df["source"].value_counts().head(12).to_dict()
    lens = df["text"].str.len()
    log.info("[%s] n=%d | human=%d ai=%d", tag, len(df),
             counts.get(0, 0), counts.get(1, 0))
    log.info("[%s] text chars: min=%d median=%d max=%d",
             tag, int(lens.min()), int(lens.median()), int(lens.max()))
    log.info("[%s] by source: %s", tag, by_src)


def load_all(dedup: bool = True) -> pd.DataFrame:
    """Load + normalise every raw labeled dataset into one shuffled DataFrame."""
    config.ensure_dirs()
    config.set_seed()
    rng = random.Random(SEED)

    files = discover_files()
    log.info("Discovered %d raw dataset file(s): %s",
             len(files), [f.name for f in files])

    rows: list[tuple] = []
    for p in files:
        rows.extend(load_file(p, rng=rng))

    if not rows:
        log.warning("No labeled samples found. Place CSVs in data/external/ or models/.")
        return pd.DataFrame(columns=["text", "label", "source"])

    df = pd.DataFrame(rows, columns=["text", "label", "source"])
    log.info("Merged valid samples (pre-dedup): %d", len(df))

    if dedup:
        before = len(df)
        df["_h"] = df["text"].map(text_hash)
        df = df.drop_duplicates("_h").drop(columns="_h")
        log.info("Removed %d exact duplicates", before - len(df))

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    _log_stats(df, "raw-merged")
    return df


if __name__ == "__main__":
    load_all()

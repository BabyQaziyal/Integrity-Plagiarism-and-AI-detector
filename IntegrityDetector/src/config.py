"""Central configuration — the single source of truth.

Import this everywhere instead of hard-coding paths, seeds, the device, dataset
rules, or hyper-parameters. Values can be overridden via environment variables
where it makes sense (DB URL, host/port, REQUIRE_CUDA).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # repo root  (c:\plagiarism)

DATA_DIR      = ROOT / "data"
EXTERNAL_DIR  = DATA_DIR / "external"               # spec drop-zone for Kaggle CSVs
GENERATED_DIR = DATA_DIR / "generated"
INTERIM_DIR   = DATA_DIR / "interim"
SPLITS_DIR    = DATA_DIR / "splits"
FINAL_DATASET = DATA_DIR / "final_dataset.csv"
GENERATED_CSV = GENERATED_DIR / "dataset_generated.csv"

MODELS_DIR      = ROOT / "models"                   # raw Kaggle CSVs already live here
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
CACHE_DIR       = ROOT / ".cache"                   # HF / model cache

# Directories scanned for raw *labeled* datasets (must have text + label/generated).
# Honours the spec's data/external while also reading the CSVs already in models/.
RAW_DATA_DIRS = [EXTERNAL_DIR, MODELS_DIR]

# Source-attribution corpus (web-like). train_prompts.csv carries long source_text.
SOURCE_CORPUS_FILES = [MODELS_DIR / "train_prompts.csv"]

# Files skipped when scanning for labeled training data (wrong schema / our outputs).
RAW_SCAN_SKIP = {"train_prompts.csv", "final_dataset.csv", "dataset_generated.csv"}

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch so the whole pipeline is reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Schema / dataset normalisation rules
# --------------------------------------------------------------------------- #
TEXT_COLUMN_CANDIDATES   = ["text", "essay", "content", "body", "document"]
LABEL_COLUMN_CANDIDATES  = ["label", "generated", "is_ai", "ai_generated", "target", "class"]
SOURCE_COLUMN_CANDIDATES = ["source", "model", "prompt_name", "dataset"]

NORMALISED_COLUMNS = ["text", "label", "source"]
LABEL_HUMAN = 0
LABEL_AI    = 1

# Short text MUST survive into training: the detector has to learn that casual,
# terse, slangy, or goofy human writing is *not* AI. We only drop truly empty
# rows here; length/quality diversity is supplied by the diverse corpus instead
# of a hard length floor (the old 200-char / 40-word floor made every sample a
# long formal essay, so anything shorter was out-of-distribution → flagged AI).
MIN_TEXT_CHARS = 1
MIN_TEXT_WORDS = 1
MAX_TEXT_CHARS = 20000     # truncate absurdly long rows before hashing/storage

# Per-source cap keeps the merge balanced + memory-friendly: DAIGT v2 (~460k) and
# Training_Essay (~255k) dwarf the others, and both are SORTED BY LABEL on disk,
# so we reservoir-sample (uniform random) rather than head-truncate.
MAX_SAMPLES_PER_SOURCE = 60000
TARGET_TOTAL           = 220000    # final balanced size target (pre-split)
BALANCE_CLASSES        = True

# Train / val / test split
SPLIT_TRAIN = 0.80
SPLIT_VAL   = 0.10
SPLIT_TEST  = 0.10

# --------------------------------------------------------------------------- #
# Synthetic generation
# --------------------------------------------------------------------------- #
GEN_HUMAN_MIN  = 3000
GEN_AI_MIN     = 3000
GEN_USE_GPT2   = True              # GPT-2 local generation preferred; template fallback
GPT2_GEN_MODEL = "gpt2"
GEN_MAX_NEW_TOKENS = 220

# --------------------------------------------------------------------------- #
# Models / hyper-parameters
# --------------------------------------------------------------------------- #
AI_DETECTOR_BASE_MODEL = "distilbert-base-uncased"
AI_DETECTOR_DIR        = CHECKPOINTS_DIR / "ai_detector"
MAX_SEQ_LEN  = 512
TRAIN_BATCH  = 16
EVAL_BATCH   = 32
EPOCHS       = 3
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
USE_AMP      = True                # mixed-precision training (RTX 4060 Ti)

PERPLEXITY_MODEL = "gpt2"
SPACY_MODEL      = "en_core_web_sm"

# Plagiarism / chunking
CHUNK_WORDS        = 60            # sliding-window size (words)
CHUNK_STRIDE       = 30
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM        = (1, 3)
PLAGIARISM_SIM_THRESHOLD = 0.55    # chunk flagged as matched at/above this cosine

# Aggregation weights for the combined AI score (must sum to 1.0)
AI_WEIGHTS = {"classifier": 0.60, "perplexity": 0.25, "burstiness": 0.15}

# Confidence / abstention behaviour ------------------------------------------
# Below this many words the signals are unreliable: the classifier abstains for
# extremely short input, perplexity/burstiness already lack data, and the
# analyzer reports "insufficient text" instead of confidently accusing. This is
# what stops short, goofy, throwaway text from being branded AI.
MIN_RELIABLE_WORDS = 8
MIN_CLASSIFIER_WORDS = 3          # below this the classifier itself abstains
# Each sub-signal is down-weighted by its own confidence (|p-0.5|*2). A small
# floor keeps a near-50/50 signal from being zeroed out entirely.
AI_CONF_FLOOR = 0.2

# Integrity score = 100 - weighted(plagiarism%, ai%)
INTEGRITY_PLAGIARISM_WEIGHT = 0.5
INTEGRITY_AI_WEIGHT         = 0.5

# --------------------------------------------------------------------------- #
# Device (NVIDIA CUDA)
# --------------------------------------------------------------------------- #
REQUIRE_CUDA = os.environ.get("REQUIRE_CUDA", "1") == "1"


def get_device(require_cuda: bool | None = None):
    """Return the torch device. Raises if CUDA is required but unavailable."""
    import torch
    require = REQUIRE_CUDA if require_cuda is None else require_cuda
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require:
        raise RuntimeError(
            "CUDA is required but not available. This project trains on an NVIDIA "
            "GPU (e.g. RTX 4060 Ti). Set REQUIRE_CUDA=0 to allow CPU fallback for "
            "light inference/testing."
        )
    return torch.device("cpu")


def describe_device() -> str:
    import torch
    if torch.cuda.is_available():
        return f"cuda | GPU Detected: {torch.cuda.get_device_name(0)}"
    return "cpu (no CUDA device)"


# --------------------------------------------------------------------------- #
# Database / API / uploads
# --------------------------------------------------------------------------- #
DB_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(ROOT / 'app.db').as_posix()}")
UPLOAD_DIR    = ROOT / "uploads"
MAX_UPLOAD_MB = 25
ALLOWED_UPLOAD_EXT = {".pdf", ".docx", ".txt"}
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

# Which detectors the API runs per submission (toggle heavy ones via env).
API_RUN_CLASSIFIER = os.environ.get("API_RUN_CLASSIFIER", "1") == "1"
API_RUN_PERPLEXITY = os.environ.get("API_RUN_PERPLEXITY", "1") == "1"
API_RUN_PLAGIARISM = os.environ.get("API_RUN_PLAGIARISM", "1") == "1"
API_RUN_STYLOMETRY = os.environ.get("API_RUN_STYLOMETRY", "1") == "1"

# --------------------------------------------------------------------------- #
# Theme (watermelon / lemon chiffon) — used by the PDF report + shared with UI
# --------------------------------------------------------------------------- #
THEME = {
    "watermelon":    "#f0485f",
    "lemon_chiffon": "#fdf7c3",
    "ink":           "#2b2b2b",
    "muted":         "#7a7a7a",
    "good":          "#3aa76d",
    "warn":          "#e8a13a",
}


def ensure_dirs() -> None:
    """Create every directory the pipeline/app writes to."""
    for d in (DATA_DIR, EXTERNAL_DIR, GENERATED_DIR, INTERIM_DIR, SPLITS_DIR,
              MODELS_DIR, CHECKPOINTS_DIR, CACHE_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)

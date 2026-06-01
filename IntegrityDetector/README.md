# AI Plagiarism & AI-Content Detector for Universities

**Is this student’s work — or the internet’s?**

A university-grade, fully-offline full-stack system that analyses student submissions and reports:

- **Plagiarism %** — TF-IDF + cosine similarity over sliding chunks vs a web-like corpus
- **AI-generated %** — fine-tuned **DistilBERT** + **GPT-2 perplexity** + **burstiness**
- **Integrity score** + verdict (Low risk / Review recommended / High risk)
- **Highlighted suspicious text**, **matched sources**, and a per-student **writing fingerprint** compared against the student’s history
- **Exportable PDF integrity report** (themed watermelon `#f0485f` / lemon chiffon `#fdf7c3`)

Stack: **Python · Flask · SQLAlchemy · PyTorch (CUDA) · HuggingFace Transformers · spaCy** + **React · Tailwind · PDF.js**.

---

## Architecture

```
React + Tailwind UI  ──REST──►  Flask API  ──►  Detection core (PyTorch/CUDA)  ──►  Data pipeline
(upload, PDF.js,                (students,       plagiarism · AI · stylometry ·       (loader →
 report, history)                submissions,    source attribution · aggregator)     generator →
                                 analysis)        SQLAlchemy ↕ SQLite/PostgreSQL        builder)
```

See the file tree and per-module docstrings; every component is importable from `src/`.

---

## 1. Setup

### Prerequisites
- **NVIDIA GPU + CUDA** (target: RTX 4060 Ti). Training is GPU-only by design.
- Python 3.11, Node 18+ (for the UI), and **a few GB of free disk** for model caches.

### Install (backend)
```powershell
# 1) Install the CUDA build of PyTorch FIRST (cu121 shown):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2) Then the rest:
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # POS features for stylometry

# 3) Verify the GPU:
python scripts/check_gpu.py
#   Using device: cuda
#   GPU Detected: NVIDIA GeForce RTX 4060 Ti
```

> HuggingFace downloads DistilBERT/GPT-2 on first use. To go fully offline afterwards,
> set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (see `.env.example`).

### Datasets (offline, user-provided)
Kaggle CSVs may live in **`models/`** (where they already are) or in **`data/external/`** —
the loader scans both. Approved datasets: DAIGT v2, AIDE, LLM-Detect AI Generated Text.
No APIs, no runtime internet dataset fetching.

---

## 2. Build the dataset (reproducible, fixed seed = 42)

```powershell
python scripts/generate_dataset.py            # optional synthetic augmentation (≥3000 + ≥3000)
#   --no-gpt2  → template-only (fast)
python scripts/build_dataset.py               # loader → merge → dedup → balance → split 80/10/10
#   --no-generated   → real Kaggle data only
#   --target 80000   → cap balanced size
```

Outputs `data/final_dataset.csv` (+ `data/splits/{train,val,test}.csv`). The pipeline
auto-detects each schema (`generated`/`label` → 0=human, 1=AI), reservoir-samples the
label-sorted big files, hash-dedups across all sources, and splits with **no leakage**.

> Verified run on the provided data: 76,835 valid → 22,595 dups removed → **49,400 balanced**
> rows → 39,520 / 4,940 / 4,940 train/val/test.

---

## 3. Train the AI detector (CUDA + mixed precision)

```powershell
python scripts/train.py                        # fine-tunes DistilBERT on data/splits
#   --epochs 3 --batch 16 --max-train 40000
```

Strict CUDA + AMP (`autocast` + `GradScaler`). Saves to `models/checkpoints/ai_detector/`.
**Until this runs, the classifier signal is ignored** and AI detection uses perplexity +
burstiness (graceful degradation — the system still works).

## 4. Build the plagiarism index

```powershell
python -m src.detection.plagiarism --build     # TF-IDF index from train_prompts + human essays
```
Saves `models/checkpoints/plagiarism_index.joblib`.

---

## 5. Run the app

```powershell
# Backend (terminal 1)
python scripts/run_api.py                       # http://127.0.0.1:5000

# Frontend (terminal 2)
cd frontend
npm install
npm run dev                                     # http://localhost:5173  (proxies /api → backend)
```

Open the UI → pick/create a student → paste text or upload a `.pdf/.docx/.txt` →
**Analyze** → view scores, highlighted text, matched sources, stylometry, and download the PDF.

### Detector toggles (env)
`API_RUN_CLASSIFIER`, `API_RUN_PERPLEXITY`, `API_RUN_PLAGIARISM`, `API_RUN_STYLOMETRY` (all `1` by default).
`DATABASE_URL` switches SQLite↔PostgreSQL. See `.env.example`.

---

## API quick reference

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | status, device, enabled detectors |
| POST | `/api/students` | create student `{name,email?}` |
| GET  | `/api/students` · `/api/students/:id` | list / detail + submissions |
| POST | `/api/submissions` | analyze (multipart file **or** JSON `{student_id,text,title}`) |
| GET  | `/api/submissions?student_id=` · `/:id` | list / full result |
| GET  | `/api/submissions/:id/result` | analysis JSON |
| GET  | `/api/submissions/:id/file` | original upload (for PDF.js) |
| GET  | `/api/submissions/:id/report` | PDF integrity report |

---

## How scoring works

- **Plagiarism %** = share of document characters covered by chunks whose best cosine
  similarity to the corpus ≥ threshold (overlapping windows merged, not double-counted).
- **AI %** = weighted blend of classifier `P(AI)` (0.60), perplexity likelihood (0.25),
  burstiness likelihood (0.15); missing signals drop out and weights renormalise.
- **Integrity** = `100 − 0.5·plagiarism% − 0.5·AI%`.
- **Stylometry** z-scores a submission’s fingerprint against the student’s prior
  submissions (needs ≥2) to flag sudden style shifts.

> Reports are **decision-support evidence**, not a verdict. Scores are probabilistic —
> corroborate before any academic-integrity action.

---

## Project layout
```
src/
  config.py · logging_utils.py
  data/      text_cleaning · dataset_loader · dataset_generator · dataset_builder
  detection/ plagiarism · source_attribution · ai_detector · perplexity · burstiness ·
             stylometry · aggregator · analyzer
  training/  train_ai_detector            reports/ pdf_report
  api/       app · db · models · routes/ · services/
scripts/     check_gpu · generate_dataset · build_dataset · train · run_api
frontend/    Vite + React + Tailwind + PDF.js   (watermelon / lemon chiffon)
data/  models/  (raw CSVs + checkpoints)
```

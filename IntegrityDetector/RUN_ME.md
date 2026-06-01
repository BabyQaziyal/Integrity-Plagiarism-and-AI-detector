# Quick Start — AI Plagiarism & AI-Content Detector

The AI model is **already trained and included** (`models/checkpoints/`).
You only need to install dependencies and run. (Full details in `README.md`.)

## 1) Backend  (Python 3.11)

If your laptop has **no NVIDIA GPU**, allow CPU fallback first (PowerShell):

    $env:REQUIRE_CUDA = "0"

Install PyTorch, then the rest:

    # NVIDIA GPU:  pip install torch --index-url https://download.pytorch.org/whl/cu121
    # CPU only:    pip install torch
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

Run the API (first run downloads GPT-2 once, needs internet):

    python scripts/run_api.py          # http://127.0.0.1:5000

## 2) Frontend  (Node 18+, second terminal)

    cd frontend
    npm install
    npm run dev                        # open http://localhost:5173

Pick/create a student, paste text or upload a PDF/Word/TXT, click **Analyze**.

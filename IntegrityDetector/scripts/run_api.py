"""CLI: run the Flask API.

    python scripts/run_api.py

Environment knobs (see .env.example): FLASK_HOST, FLASK_PORT, API_RUN_PERPLEXITY, ...
Set API_WARMUP=0 to skip pre-loading models at startup.

Tip: to run the API *and* the web UI together, use ./dev.ps1 from the repo root.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.api.app import app, warmup  # noqa: E402


def main() -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    url = f"http://{config.FLASK_HOST}:{config.FLASK_PORT}"

    print("=" * 64)
    print(f"  Integrity Detector API  |  device: {device}")
    print(f"  API:        {url}")
    print(f"  Health:     {url}/api/health")
    print(f"  Web UI:     http://localhost:5173   <-- open THIS in your browser")
    print("=" * 64)

    # Warm the models in the background so startup is instant but the first
    # real submission doesn't pay the full model-load cost.
    if os.environ.get("API_WARMUP", "1") == "1":
        print("  Warming up models in the background (first submission will be fast)…")
        threading.Thread(target=warmup, daemon=True).start()

    # threaded=True: health/students calls don't block behind a slow analysis.
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT,
            debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

"""Flask application factory."""
from __future__ import annotations

from flask import Flask, Response, g, jsonify

from src import config
from src.api.db import SessionLocal, init_db
from src.logging_utils import get_logger

log = get_logger("api")

# A tiny landing page so hitting the API origin (e.g. http://127.0.0.1:5000)
# directly in a browser shows guidance instead of a bare 404.
_INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Integrity Detector API</title>
<style>
 body{{margin:0;font-family:Inter,Segoe UI,system-ui,sans-serif;color:#2b2b2b;
   background:#fdf7c3;background-image:radial-gradient(40rem 40rem at 110% -10%,#ffe1e6 0,transparent 60%);
   min-height:100vh;display:grid;place-items:center}}
 .card{{background:#fff;max-width:34rem;padding:2rem 2.25rem;border-radius:1.25rem;
   box-shadow:0 10px 40px -12px rgba(43,43,43,.25);border:1px solid rgba(0,0,0,.05)}}
 .dot{{display:inline-block;width:.6rem;height:.6rem;border-radius:50%;background:#3aa76d;margin-right:.4rem}}
 h1{{margin:.2rem 0 .25rem;font-size:1.4rem}} code{{background:#fff1f3;color:#d83452;
   padding:.15rem .4rem;border-radius:.4rem;font-size:.85em}}
 a{{color:#d83452;font-weight:600}} .muted{{color:#7a7a7a;font-size:.9rem;line-height:1.5}}
</style></head><body><div class="card">
 <div class="muted"><span class="dot"></span>API is running</div>
 <h1>Integrity Detector API</h1>
 <p class="muted">This is the <b>backend</b>. The app you want is the web UI:
 <a href="http://localhost:5173/">http://localhost:5173/</a> (Vite dev server).</p>
 <p class="muted">Health check: <a href="/api/health">/api/health</a> &middot;
 Endpoints live under <code>/api/*</code>.</p>
 <p class="muted">Tip: run the API and the UI together with
 <code>./dev.ps1</code> from the project root.</p>
</div></body></html>"""


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False

    # CORS (allow the React dev server / configured origins)
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": config.CORS_ORIGINS}})
    except ImportError:
        log.warning("flask-cors not installed; cross-origin requests may fail")

    config.ensure_dirs()
    init_db()

    # ---- per-request DB session -------------------------------------------
    @app.before_request
    def _open_session():
        g.db = SessionLocal()

    @app.teardown_appcontext
    def _close_session(exc):
        db = g.pop("db", None)
        if db is None:
            return
        try:
            if exc is None:
                db.commit()
            else:
                db.rollback()
        finally:
            db.close()

    # ---- blueprints --------------------------------------------------------
    from src.api.routes import ALL_BLUEPRINTS
    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)

    # ---- misc routes / errors ---------------------------------------------
    @app.get("/")
    def index():
        return Response(_INDEX_HTML, mimetype="text/html")

    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/api/health")
    def health():
        import torch
        return jsonify({
            "status": "ok",
            "device": config.describe_device(),
            "cuda": torch.cuda.is_available(),
            "detectors": {
                "classifier": config.API_RUN_CLASSIFIER,
                "perplexity": config.API_RUN_PERPLEXITY,
                "plagiarism": config.API_RUN_PLAGIARISM,
                "stylometry": config.API_RUN_STYLOMETRY,
            },
        })

    @app.errorhandler(404)
    def _404(_e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(413)
    def _413(_e):
        return jsonify({"error": f"file too large (max {config.MAX_UPLOAD_MB} MB)"}), 413

    @app.errorhandler(ValueError)
    def _value_error(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(500)
    def _500(e):
        log.exception("Unhandled server error")
        return jsonify({"error": "internal server error"}), 500

    return app


def warmup() -> None:
    """Pre-load the heavy detector models so the first submission isn't slow.

    Safe to call in a background thread; failures are logged, never raised."""
    sample = ("The quick brown fox jumps over the lazy dog. This warmup paragraph "
              "primes the language models so the first real submission is fast.")
    try:
        from src.detection.analyzer import analyze_submission
        analyze_submission(sample,
                           run_classifier=config.API_RUN_CLASSIFIER,
                           run_perplexity=config.API_RUN_PERPLEXITY,
                           run_plagiarism=config.API_RUN_PLAGIARISM,
                           run_stylometry=config.API_RUN_STYLOMETRY)
        log.info("Model warmup complete — detectors are ready.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Model warmup skipped (%s): %s", type(exc).__name__, exc)


# WSGI entrypoint:  gunicorn/waitress -> "src.api.app:app"
app = create_app()


if __name__ == "__main__":
    print(f"Using device: {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False, threaded=True)

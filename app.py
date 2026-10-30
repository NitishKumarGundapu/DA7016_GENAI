"""Local web UI: run with ``python app.py``."""
import json
import logging
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from auto_assessment.config import Settings
from auto_assessment.ingest import IMAGE_MIME_TYPES, TEXT_EXT
from auto_assessment.pipeline import stream_assessment, tracing_enabled

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
logger = logging.getLogger(__name__)
SUPPORTED = {".pdf", *IMAGE_MIME_TYPES, *TEXT_EXT}


@app.after_request
def protect_response(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    )
    return response


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description), error.code


@app.get("/")
def index():
    return render_template("index.html", settings=Settings.from_env())


def _settings(form):
    settings = Settings.from_env()
    settings.provider = form.get("provider", settings.provider)
    if settings.provider not in {"ollama", "openrouter"}:
        raise ValueError("Choose Ollama or OpenRouter.")
    for field in ("openrouter_model", "openrouter_vision_model", "ollama_model", "ollama_base_url"):
        value = form.get(field, getattr(settings, field)).strip()
        if not value:
            raise ValueError(f"{field.replace('_', ' ').capitalize()} must not be blank.")
        setattr(settings, field, value)
    url = urlsplit(settings.ollama_base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("Ollama URL must be an HTTP(S) URL without credentials.")
    settings.max_grade_retries = int(form.get("max_grade_retries", settings.max_grade_retries))
    if not 0 <= settings.max_grade_retries <= 4:
        raise ValueError("Re-grade retries must be between 0 and 4.")
    settings.use_critic = form.get("use_critic") == "on"
    return settings


@app.post("/api/assess")
def assess():
    origin = request.headers.get("Origin")
    if (origin and origin != request.host_url.rstrip("/")) or request.headers.get(
        "Sec-Fetch-Site"
    ) == "cross-site":
        return jsonify(error="Cross-origin assessment requests are not allowed."), 403
    try:
        settings = _settings(request.form)
        uploads = {}
        for role in ("question_paper", "rubric", "model_answers", "answer_files"):
            uploads[role] = [file for file in request.files.getlist(role) if file.filename]
            if role != "answer_files" and len(uploads[role]) > 1:
                raise ValueError(f"Provide only one {role.replace('_', ' ')} file.")
            for file in uploads[role]:
                if Path(file.filename).suffix.lower() not in SUPPORTED:
                    raise ValueError("Unsupported file. Use PDF, TXT, MD, PNG, JPEG, WebP, or GIF.")
        inline = request.form.get("model_answers_text", "").strip()
        if not uploads["question_paper"] or not uploads["answer_files"]:
            raise ValueError("Upload a question paper and at least one student answer sheet.")
        if bool(uploads["model_answers"]) == bool(inline):
            raise ValueError("Provide either a model-answer file or pasted model answers, not both.")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    temp_dir = tempfile.TemporaryDirectory(prefix="auto-assessment-")
    try:
        inputs = {}
        for role, files in uploads.items():
            paths = []
            for index, file in enumerate(files):
                path = Path(temp_dir.name) / f"{role}-{index}{Path(file.filename).suffix.lower()}"
                file.save(path)
                if not path.stat().st_size:
                    raise ValueError(f"{role.replace('_', ' ').capitalize()} file is empty.")
                paths.append(str(path))
            inputs[role] = paths if role == "answer_files" else next(iter(paths), None)
    except ValueError as exc:
        temp_dir.cleanup()
        return jsonify(error=str(exc)), 400
    except OSError:
        temp_dir.cleanup()
        logger.exception("Could not save uploaded documents")
        return jsonify(error="Could not save uploaded documents. Check disk space and permissions."), 500

    def generate():
        try:
            for event in stream_assessment(
                **inputs, model_answers_text=inline or None, settings=settings, tags=["web"],
            ):
                if event["type"] == "progress":
                    payload = {"type": "progress", "node": event["node"]}
                else:
                    state = event["state"]
                    payload = {
                        "type": "final", "result": state["result"],
                        "report_md": state["report_md"], "transcript": state["transcript"],
                        "items": state["items"], "run_id": event["run_id"],
                        "tracing": tracing_enabled(),
                    }
                yield json.dumps(payload, ensure_ascii=False) + "\n"
        except Exception as exc:
            logger.exception("Assessment failed")
            yield json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n"
        finally:
            temp_dir.cleanup()

    # ponytail: request-scoped streaming, no job queue; use a worker queue for durable jobs.
    response = Response(
        generate(), mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no"},
    )
    response.call_on_close(temp_dir.cleanup)
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

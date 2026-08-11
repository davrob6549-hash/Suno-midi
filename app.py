"""
Flask web server for Suno → MIDI Converter (production build).
Handles concurrent jobs via threading, auto-cleans old files.
"""

import os
import uuid
import zipfile
import threading
import traceback
import time
import logging
from pathlib import Path

from flask import (
    Flask, request, jsonify, send_file,
    render_template, abort
)

import processor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "outputs"
ALLOWED_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
MAX_FILE_MB = 150
JOB_TTL_SEC = 3600        # clean up jobs older than 1 hour
MAX_CONCURRENT = 3        # max simultaneous demucs jobs (CPU-bound)

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder='.')
app.config["MAX_CONTENT_LENGTH"] = (MAX_FILE_MB + 10) * 1024 * 1024

# In-memory job registry
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
active_jobs = threading.Semaphore(MAX_CONCURRENT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_job(job_id: str) -> dict | None:
    with jobs_lock:
        return jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def stem_callback(job_id: str):
    def cb(stem: str, status: str, message: str):
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["stems"][stem] = {"status": status, "message": message}
    return cb


def cleanup_old_jobs():
    """Remove jobs and files older than JOB_TTL_SEC."""
    now = time.time()
    with jobs_lock:
        expired = [jid for jid, j in jobs.items()
                   if now - j.get("created_at", now) > JOB_TTL_SEC]
    for jid in expired:
        job = get_job(jid)
        if job:
            # Remove output directory
            job_dir = OUTPUT_DIR / jid
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir, ignore_errors=True)
            # Remove uploaded audio
            audio = job.get("audio_path")
            if audio and Path(audio).exists():
                Path(audio).unlink(missing_ok=True)
        with jobs_lock:
            jobs.pop(jid, None)
        log.info(f"Cleaned up job {jid}")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def run_job(job_id: str, audio_path: str, original_name: str):
    with active_jobs:
        job_dir = OUTPUT_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        update_job(job_id, status="processing", message="Starting stem separation…")

        try:
            midi_files = processor.process_audio(
                audio_path=str(audio_path),
                job_dir=str(job_dir),
                update_cb=stem_callback(job_id),
            )

            zip_name = Path(original_name).stem[:40] + "_midi.zip"
            zip_path = job_dir / zip_name
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for stem_name, midi_path in midi_files.items():
                    zf.write(midi_path, arcname=f"{stem_name}.mid")

            update_job(
                job_id,
                status="done",
                message=f"Converted {len(midi_files)} stems successfully",
                zip_path=str(zip_path),
                zip_name=zip_name,
                stem_count=len(midi_files),
            )
            log.info(f"Job {job_id} complete: {len(midi_files)} stems")

        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Job {job_id} failed: {e}\n{tb}")
            update_job(job_id, status="error", message=str(e))

        finally:
            # Remove uploaded audio once processed
            if Path(audio_path).exists():
                Path(audio_path).unlink(missing_ok=True)

    # Cleanup old jobs opportunistically
    threading.Thread(target=cleanup_old_jobs, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify(error="No file provided"), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error=f"Unsupported format '{ext}'. Accepted: MP3, WAV, FLAC, M4A, OGG, AAC"), 400

    # Check active job count
    active_count = MAX_CONCURRENT - active_jobs._value
    if active_count >= MAX_CONCURRENT:
        return jsonify(error="Server is busy — please try again in a minute"), 503

    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}{ext}"
    audio_path = UPLOAD_DIR / safe_name
    f.save(str(audio_path))

    # Enforce file size
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        audio_path.unlink(missing_ok=True)
        return jsonify(error=f"File too large ({file_size_mb:.0f} MB). Max is {MAX_FILE_MB} MB."), 413

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": f.filename,
            "status": "queued",
            "message": "Job queued",
            "stems": {},
            "zip_path": None,
            "zip_name": None,
            "audio_path": str(audio_path),
            "created_at": time.time(),
        }

    t = threading.Thread(target=run_job, args=(job_id, str(audio_path), f.filename), daemon=True)
    t.start()

    return jsonify(job_id=job_id, filename=f.filename)


@app.route("/status/<job_id>")
def status(job_id: str):
    job = get_job(job_id)
    if job is None:
        return jsonify(error="Job not found"), 404
    safe = {k: v for k, v in job.items() if k not in ("zip_path", "audio_path")}
    return jsonify(safe)


@app.route("/download/<job_id>")
def download(job_id: str):
    job = get_job(job_id)
    if job is None:
        abort(404)
    if job["status"] != "done" or not job.get("zip_path"):
        abort(400)
    return send_file(
        job["zip_path"],
        as_attachment=True,
        download_name=job["zip_name"],
        mimetype="application/zip",
    )


@app.route("/health")
def health():
    return jsonify(status="ok", active_jobs=MAX_CONCURRENT - active_jobs._value)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n🎵 Suno → MIDI Converter running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from omr_system.config import settings
from omr_system.logging_config import configure_logging
from omr_system.models import DocumentResult, JobCreateRequest, JobCreateResponse, JobResultEnvelope, JobStatusResponse
from omr_system.pipeline.processor import DocumentProcessor
from omr_system.queue.fs_queue import FileSystemJobQueue
from omr_system.utils import ensure_dir

configure_logging(settings.log_level)
queue = FileSystemJobQueue(settings.queue_dir)
app = FastAPI(title="AI-Driven OMR API", version="0.1.0")
ui_file = Path(__file__).resolve().parent / "web" / "index.html"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "omr-api"}


@app.get("/ready")
def ready() -> dict:
    return {"status": "ready"}


@app.get("/", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    return HTMLResponse(ui_file.read_text(encoding="utf-8"))


@app.get("/ui", response_class=HTMLResponse)
def web_ui_alias() -> HTMLResponse:
    return HTMLResponse(ui_file.read_text(encoding="utf-8"))


@app.post("/scan/sync", response_model=DocumentResult)
async def scan_sync(
    image: UploadFile = File(...),
    template_path: str = Form("config/template.example.yaml"),
    answer_key_json: str = Form("{}"),
    save_artifacts: bool = Form(True),
) -> DocumentResult:
    try:
        answer_key_raw = json.loads(answer_key_json)
        if not isinstance(answer_key_raw, dict):
            raise ValueError("answer_key_json must be a JSON object")
        answer_key = {str(key): str(value) for key, value in answer_key_raw.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads_dir = ensure_dir(settings.runtime_dir / "uploads")
    suffix = Path(image.filename or "capture.jpg").suffix or ".jpg"
    upload_path = uploads_dir / f"{uuid4().hex}{suffix}"
    upload_path.write_bytes(await image.read())

    processor = DocumentProcessor()
    return processor.process_file(
        input_path=upload_path,
        template_path=template_path,
        answer_key=answer_key,
        save_artifacts=save_artifacts,
    )


@app.post("/jobs", response_model=JobCreateResponse)
def create_job(request: JobCreateRequest) -> JobCreateResponse:
    payload = queue.enqueue(
        inputs=request.inputs,
        template_path=request.template_path,
        answer_key=request.answer_key,
        save_artifacts=request.save_artifacts,
    )
    return JobCreateResponse(job_id=payload.job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    record = queue.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=record.get("status", "unknown"),
        submitted_at=record.get("submitted_at"),
        started_at=record.get("started_at"),
        finished_at=record.get("finished_at"),
        error=record.get("error"),
    )


@app.get("/jobs/{job_id}/results", response_model=JobResultEnvelope)
def get_job_result(job_id: str) -> JobResultEnvelope:
    record = queue.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResultEnvelope(
        job_id=job_id,
        status=record.get("status", "unknown"),
        result=record.get("result"),
        error=record.get("error"),
    )

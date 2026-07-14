from __future__ import annotations

from fastapi import FastAPI, HTTPException

from omr_system.config import settings
from omr_system.logging_config import configure_logging
from omr_system.models import JobCreateRequest, JobCreateResponse, JobResultEnvelope, JobStatusResponse
from omr_system.queue.fs_queue import FileSystemJobQueue

configure_logging(settings.log_level)
queue = FileSystemJobQueue(settings.queue_dir)
app = FastAPI(title="AI-Driven OMR API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "omr-api"}


@app.get("/ready")
def ready() -> dict:
    return {"status": "ready"}


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


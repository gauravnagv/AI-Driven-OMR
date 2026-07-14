from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from omr_system.models import JobPayload, WorkerClaim
from omr_system.utils import ensure_dir, now_iso, read_json, write_json


class FileSystemJobQueue:
    def __init__(self, queue_root: Path) -> None:
        self.queue_root = queue_root
        self.pending = ensure_dir(queue_root / "pending")
        self.processing = ensure_dir(queue_root / "processing")
        self.done = ensure_dir(queue_root / "done")
        self.failed = ensure_dir(queue_root / "failed")

    def enqueue(
        self,
        inputs: list[str],
        template_path: str,
        answer_key: dict[str, str],
        save_artifacts: bool,
    ) -> JobPayload:
        job_id = str(uuid4())
        payload = JobPayload(
            job_id=job_id,
            submitted_at=now_iso(),
            inputs=inputs,
            template_path=template_path,
            answer_key=answer_key,
            save_artifacts=save_artifacts,
        )
        write_json(self.pending / f"{job_id}.json", payload.model_dump())
        return payload

    def claim(self, worker_id: str) -> WorkerClaim | None:
        for pending_file in sorted(self.pending.glob("*.json")):
            job_id = pending_file.stem
            target = self.processing / f"{job_id}.{worker_id}.json"
            try:
                pending_file.replace(target)
            except FileNotFoundError:
                continue
            payload = JobPayload.model_validate(read_json(target))
            return WorkerClaim(job_id=job_id, processing_path=target, payload=payload)
        return None

    def complete(self, claim: WorkerClaim, result: list[dict]) -> None:
        write_json(
            self.done / f"{claim.job_id}.json",
            {
                "job_id": claim.job_id,
                "status": "done",
                "submitted_at": claim.payload.submitted_at,
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "result": result,
            },
        )
        if claim.processing_path.exists():
            claim.processing_path.unlink()

    def fail(self, claim: WorkerClaim, error: str) -> None:
        write_json(
            self.failed / f"{claim.job_id}.json",
            {
                "job_id": claim.job_id,
                "status": "failed",
                "submitted_at": claim.payload.submitted_at,
                "started_at": now_iso(),
                "finished_at": now_iso(),
                "error": error,
            },
        )
        if claim.processing_path.exists():
            claim.processing_path.unlink()

    def get(self, job_id: str) -> dict | None:
        done_path = self.done / f"{job_id}.json"
        failed_path = self.failed / f"{job_id}.json"
        pending_path = self.pending / f"{job_id}.json"
        processing_files = list(self.processing.glob(f"{job_id}.*.json"))

        if done_path.exists():
            return read_json(done_path)
        if failed_path.exists():
            return read_json(failed_path)
        if pending_path.exists():
            pending_payload = JobPayload.model_validate(read_json(pending_path))
            return {
                "job_id": job_id,
                "status": "queued",
                "submitted_at": pending_payload.submitted_at,
            }
        if processing_files:
            proc_payload = JobPayload.model_validate(read_json(processing_files[0]))
            return {
                "job_id": job_id,
                "status": "processing",
                "submitted_at": proc_payload.submitted_at,
            }
        return None


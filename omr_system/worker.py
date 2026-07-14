from __future__ import annotations

import logging
import time
from pathlib import Path
from uuid import uuid4

from omr_system.config import settings
from omr_system.logging_config import configure_logging
from omr_system.pipeline.batch import BatchProcessor
from omr_system.queue.fs_queue import FileSystemJobQueue
from omr_system.utils import ensure_dir

logger = logging.getLogger(__name__)


def run_worker() -> None:
    configure_logging(settings.log_level)
    worker_id = f"worker-{uuid4().hex[:8]}"
    queue = FileSystemJobQueue(settings.queue_dir)
    processor = BatchProcessor(max_workers=settings.max_batch_workers)

    heartbeat_dir = ensure_dir(settings.runtime_dir / "health")
    heartbeat_file = heartbeat_dir / "worker_heartbeat.txt"
    logger.info("Starting worker %s", worker_id)

    while True:
        heartbeat_file.write_text(str(time.time()), encoding="utf-8")
        claim = queue.claim(worker_id)
        if claim is None:
            time.sleep(settings.worker_poll_seconds)
            continue
        try:
            result = processor.process(
                inputs=claim.payload.inputs,
                template_path=claim.payload.template_path,
                answer_key=claim.payload.answer_key,
                save_artifacts=claim.payload.save_artifacts,
            )
            queue.complete(claim, [doc.model_dump() for doc in result])
            logger.info("Job %s completed", claim.job_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", claim.job_id)
            queue.fail(claim, str(exc))


if __name__ == "__main__":
    run_worker()


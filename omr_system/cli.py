from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn

from omr_system.config import settings
from omr_system.pipeline.batch import BatchProcessor
from omr_system.pipeline.processor import DocumentProcessor
from omr_system.worker import run_worker

app = typer.Typer(help="AI-driven OMR processing CLI")


@app.command("process-one")
def process_one(
    input_path: Path,
    template_path: Path,
    answer_key_path: Path | None = None,
    save_artifacts: bool = True,
) -> None:
    answer_key = {}
    if answer_key_path:
        answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    processor = DocumentProcessor()
    result = processor.process_file(
        input_path=input_path,
        template_path=template_path,
        answer_key=answer_key,
        save_artifacts=save_artifacts,
    )
    typer.echo(json.dumps(result.model_dump(), indent=2))


@app.command("process-batch")
def process_batch(
    inputs_file: Path,
    template_path: Path,
    answer_key_path: Path | None = None,
    workers: int = settings.max_batch_workers,
    save_artifacts: bool = True,
) -> None:
    inputs = json.loads(inputs_file.read_text(encoding="utf-8"))
    answer_key = {}
    if answer_key_path:
        answer_key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    processor = BatchProcessor(max_workers=workers)
    result = processor.process(
        inputs=inputs,
        template_path=str(template_path),
        answer_key=answer_key,
        save_artifacts=save_artifacts,
    )
    typer.echo(json.dumps([x.model_dump() for x in result], indent=2))


@app.command("serve-api")
def serve_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run("omr_system.api.app:app", host=host, port=port, reload=False)


@app.command("run-worker")
def worker_cmd() -> None:
    run_worker()


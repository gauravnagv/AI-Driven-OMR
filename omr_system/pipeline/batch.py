from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from omr_system.models import DocumentResult
from omr_system.pipeline.processor import DocumentProcessor


def _process_single(
    input_path: str,
    template_path: str,
    answer_key: dict[str, str],
    save_artifacts: bool,
) -> dict:
    processor = DocumentProcessor()
    result = processor.process_file(
        input_path=Path(input_path),
        template_path=Path(template_path),
        answer_key=answer_key,
        save_artifacts=save_artifacts,
    )
    return result.model_dump()


class BatchProcessor:
    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def process(
        self,
        inputs: list[str],
        template_path: str,
        answer_key: dict[str, str] | None = None,
        save_artifacts: bool = True,
    ) -> list[DocumentResult]:
        answer_key = answer_key or {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(_process_single, path, template_path, answer_key, save_artifacts) for path in inputs
            ]
            return [DocumentResult.model_validate(f.result()) for f in futures]


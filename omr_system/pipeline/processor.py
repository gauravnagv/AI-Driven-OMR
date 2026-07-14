from __future__ import annotations

import logging
from pathlib import Path

import cv2

from omr_system.config import settings
from omr_system.dewarp.dewarper import Dewarper
from omr_system.detection.page_detector import PageDetector
from omr_system.models import DocumentResult
from omr_system.omr.extractor import OMRExtractor
from omr_system.omr.template import load_template
from omr_system.utils import ensure_dir, write_json

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self) -> None:
        self.page_detector = PageDetector(settings.page_detector_weights, settings.detector_confidence)
        self.dewarper = Dewarper()
        self.extractor = OMRExtractor()

    def process_file(
        self,
        input_path: str | Path,
        template_path: str | Path,
        answer_key: dict[str, str] | None = None,
        save_artifacts: bool = True,
    ) -> DocumentResult:
        input_path = Path(input_path)
        image = cv2.imread(str(input_path))
        if image is None:
            return DocumentResult(input_path=str(input_path), success=False, error="Unable to read input image")

        try:
            detection = self.page_detector.detect(image)
            dewarped = self.dewarper.dewarp(image, detection.quadrilateral_xy)
            template = load_template(template_path)
            omr_result = self.extractor.extract(dewarped, template, answer_key=answer_key)

            artifacts: dict[str, str] = {}
            if save_artifacts:
                root = ensure_dir(settings.artifacts_dir / input_path.stem)
                dewarped_path = root / "dewarped.png"
                cv2.imwrite(str(dewarped_path), dewarped)
                write_json(root / "result.json", omr_result.model_dump())
                artifacts = {
                    "dewarped": str(dewarped_path),
                    "result": str(root / "result.json"),
                }

            return DocumentResult(
                input_path=str(input_path),
                success=True,
                detection=detection,
                omr=omr_result,
                artifacts=artifacts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed processing %s", input_path)
            return DocumentResult(input_path=str(input_path), success=False, error=str(exc))


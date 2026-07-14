from __future__ import annotations

import logging
from pathlib import Path

import cv2

from omr_system.config import settings
from omr_system.dewarp.dewarper import Dewarper
from omr_system.detection.page_detector import PageDetector
from omr_system.models import DocumentResult, ProcessingStep
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
            return DocumentResult(
                input_path=str(input_path),
                success=False,
                sheet_present=False,
                error="Unable to read input image",
                processing_steps=[
                    ProcessingStep(name="load-image", status="failed", detail="Input image file is unreadable.")
                ],
            )

        try:
            steps: list[ProcessingStep] = [
                ProcessingStep(
                    name="load-image",
                    status="success",
                    detail=f"Loaded image with resolution {image.shape[1]}x{image.shape[0]}",
                )
            ]
            detection = self.page_detector.detect(image)
            if detection is None:
                steps.append(
                    ProcessingStep(
                        name="detect-sheet",
                        status="failed",
                        detail="No OMR sheet detected in frame. Hold the full sheet inside camera view and retry.",
                    )
                )
                return DocumentResult(
                    input_path=str(input_path),
                    success=False,
                    sheet_present=False,
                    error="No OMR sheet detected in the provided image.",
                    processing_steps=steps,
                )

            steps.append(
                ProcessingStep(
                    name="detect-sheet",
                    status="success",
                    detail=f"Detected sheet via {detection.method} (confidence={detection.confidence:.2f}, area={detection.area_ratio:.2f})",
                )
            )
            dewarped = self.dewarper.dewarp(image, detection.quadrilateral_xy)
            steps.append(
                ProcessingStep(
                    name="dewarp-sheet",
                    status="success",
                    detail=f"Dewarped and flattened sheet to {dewarped.shape[1]}x{dewarped.shape[0]}",
                )
            )
            template = load_template(template_path)
            steps.append(
                ProcessingStep(
                    name="load-template",
                    status="success",
                    detail=f"Loaded template with {len(template.questions)} questions",
                )
            )
            omr_result = self.extractor.extract(
                dewarped,
                template,
                answer_key=answer_key,
                detection_confidence=detection.confidence,
            )
            steps.append(
                ProcessingStep(
                    name="extract-omr",
                    status="success",
                    detail=omr_result.summary,
                )
            )

            artifacts: dict[str, str] = {}
            if save_artifacts:
                root = ensure_dir(settings.artifacts_dir / input_path.stem)
                source_path = root / "source.png"
                dewarped_path = root / "dewarped.png"
                cv2.imwrite(str(source_path), image)
                cv2.imwrite(str(dewarped_path), dewarped)
                write_json(root / "result.json", omr_result.model_dump())
                artifacts = {
                    "source": str(source_path),
                    "dewarped": str(dewarped_path),
                    "result": str(root / "result.json"),
                }
                steps.append(
                    ProcessingStep(
                        name="save-artifacts",
                        status="success",
                        detail="Saved source image, dewarped image, and result JSON.",
                        artifact_path=str(root),
                    )
                )
            else:
                steps.append(
                    ProcessingStep(
                        name="save-artifacts",
                        status="skipped",
                        detail="Artifacts saving disabled for this request.",
                    )
                )

            return DocumentResult(
                input_path=str(input_path),
                success=True,
                sheet_present=True,
                detection=detection,
                omr=omr_result,
                processing_steps=steps,
                human_readable_summary=omr_result.summary,
                artifacts=artifacts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed processing %s", input_path)
            return DocumentResult(
                input_path=str(input_path),
                success=False,
                sheet_present=False,
                error=str(exc),
                processing_steps=[
                    ProcessingStep(
                        name="pipeline-error",
                        status="failed",
                        detail=f"Pipeline failed with error: {exc}",
                    )
                ],
            )

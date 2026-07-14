from pathlib import Path

import cv2
import numpy as np
import yaml

from omr_system.pipeline.processor import DocumentProcessor


def _build_synthetic_sheet(path: Path) -> None:
    canvas = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (60, 60), (740, 940), (0, 0, 0), 3)

    centers = [(220, 260), (280, 260), (340, 260), (400, 260)]
    for idx, (x, y) in enumerate(centers):
        cv2.circle(canvas, (x, y), 18, (0, 0, 0), 2)
        if idx == 3:  # mark D
            cv2.circle(canvas, (x, y), 14, (0, 0, 0), -1)
    cv2.imwrite(str(path), canvas)


def test_document_pipeline_end_to_end(tmp_path: Path) -> None:
    image_path = tmp_path / "sheet.png"
    template_path = tmp_path / "template.yaml"

    _build_synthetic_sheet(image_path)
    template = {
        "page_width": 1800,
        "page_height": 2400,
        "fill_threshold": 0.12,
        "questions": [
            {
                "question_id": "q1",
                "marks": 1,
                "negative_marks": 0.25,
                "choices": [
                    {"label": "A", "x": 500, "y": 610, "radius": 35},
                    {"label": "B", "x": 640, "y": 610, "radius": 35},
                    {"label": "C", "x": 770, "y": 610, "radius": 35},
                    {"label": "D", "x": 900, "y": 610, "radius": 35},
                ],
            }
        ],
    }
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")

    processor = DocumentProcessor()
    result = processor.process_file(
        input_path=image_path,
        template_path=template_path,
        answer_key={"q1": "D"},
        save_artifacts=False,
    )

    assert result.success
    assert result.omr is not None
    assert result.omr.responses[0].selected == "D"
    assert result.omr.score.correct == 1

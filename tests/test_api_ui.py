from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi.testclient import TestClient

from omr_system.api.app import app


def _create_sheet(path: Path) -> None:
    canvas = np.full((900, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (660, 860), (0, 0, 0), 3)
    centers = [(220, 260), (280, 260), (340, 260), (400, 260)]
    for idx, (x, y) in enumerate(centers):
        cv2.circle(canvas, (x, y), 16, (0, 0, 0), 2)
        if idx == 3:
            cv2.circle(canvas, (x, y), 12, (0, 0, 0), -1)
    cv2.imwrite(str(path), canvas)


def test_web_ui_and_sync_scan(tmp_path: Path) -> None:
    image_path = tmp_path / "scan.jpg"
    template_path = tmp_path / "template.yaml"
    _create_sheet(image_path)

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
                    {"label": "A", "x": 500, "y": 620, "radius": 34},
                    {"label": "B", "x": 640, "y": 620, "radius": 34},
                    {"label": "C", "x": 770, "y": 620, "radius": 34},
                    {"label": "D", "x": 900, "y": 620, "radius": 34},
                ],
            }
        ],
    }
    template_path.write_text(yaml.safe_dump(template), encoding="utf-8")

    client = TestClient(app)
    ui_response = client.get("/")
    assert ui_response.status_code == 200
    assert "AI-Driven OMR Web Scanner" in ui_response.text

    with image_path.open("rb") as image_stream:
        response = client.post(
            "/scan/sync",
            files={"image": ("scan.jpg", image_stream, "image/jpeg")},
            data={
                "template_path": str(template_path),
                "answer_key_json": "{\"q1\":\"D\"}",
                "save_artifacts": "false",
            },
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["omr"] is not None
    assert payload["omr"]["responses"][0]["question_id"] == "q1"

import cv2
import numpy as np

from omr_system.omr.extractor import OMRExtractor
from omr_system.omr.template import OMRTemplate, QuestionTemplate, BubbleChoice


def _draw_bubble(image: np.ndarray, x: int, y: int, r: int, filled: bool) -> None:
    cv2.circle(image, (x, y), r, (0, 0, 0), 2)
    if filled:
        cv2.circle(image, (x, y), r - 2, (0, 0, 0), -1)


def test_extract_single_mark_and_score() -> None:
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    _draw_bubble(image, 200, 150, 20, True)   # A
    _draw_bubble(image, 260, 150, 20, False)  # B
    _draw_bubble(image, 320, 150, 20, False)  # C
    _draw_bubble(image, 380, 150, 20, False)  # D

    template = OMRTemplate(
        page_width=800,
        page_height=600,
        fill_threshold=0.15,
        questions=[
            QuestionTemplate(
                question_id="q1",
                choices=[
                    BubbleChoice(label="A", x=200, y=150, radius=20),
                    BubbleChoice(label="B", x=260, y=150, radius=20),
                    BubbleChoice(label="C", x=320, y=150, radius=20),
                    BubbleChoice(label="D", x=380, y=150, radius=20),
                ],
            )
        ],
    )

    extractor = OMRExtractor()
    result = extractor.extract(image, template, answer_key={"q1": "A"})

    assert result.responses[0].selected == "A"
    assert result.responses[0].status == "single"
    assert result.score.correct == 1
    assert result.score.obtained == 1.0

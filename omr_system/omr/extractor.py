from __future__ import annotations

import cv2
import numpy as np

from omr_system.models import OMRResult, OMRScore, QuestionResponse
from omr_system.omr.template import OMRTemplate


class OMRExtractor:
    def _bubble_fill_ratio(
        self,
        binary: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> float:
        mask = np.zeros(binary.shape, dtype=np.uint8)
        # Use an inner disc so hollow bubble outlines do not look marked.
        inner_radius = max(1, int(radius * 0.6))
        cv2.circle(mask, (cx, cy), inner_radius, 255, -1)
        selected = cv2.bitwise_and(binary, binary, mask=mask)
        total = cv2.countNonZero(mask)
        if total <= 0:
            return 0.0
        return cv2.countNonZero(selected) / total

    def extract(
        self,
        image: np.ndarray,
        template: OMRTemplate,
        answer_key: dict[str, str] | None = None,
    ) -> OMRResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )

        img_h, img_w = binary.shape
        sx = img_w / template.page_width
        sy = img_h / template.page_height

        responses: list[QuestionResponse] = []
        obtained = 0.0
        maximum = 0.0
        correct = incorrect = blank = 0
        answer_key = answer_key or {}

        for q in template.questions:
            fill = {}
            for choice in q.choices:
                cx = int(choice.x * sx)
                cy = int(choice.y * sy)
                radius = int(choice.radius * min(sx, sy))
                fill[choice.label] = self._bubble_fill_ratio(binary, cx, cy, radius)

            sorted_fill = sorted(fill.items(), key=lambda item: item[1], reverse=True)
            marked = [label for label, ratio in fill.items() if ratio >= template.fill_threshold]
            if not marked:
                status = "blank"
                selected = None
                confidence = sorted_fill[0][1] if sorted_fill else 0.0
                blank += 1
            elif len(marked) == 1:
                status = "single"
                selected = marked[0]
                confidence = fill[selected]
            else:
                status = "multiple"
                selected = None
                confidence = max(fill.values())

            if q.question_id in answer_key:
                maximum += q.marks
                if status == "single" and selected == answer_key[q.question_id]:
                    obtained += q.marks
                    correct += 1
                elif status == "blank":
                    pass
                else:
                    obtained -= q.negative_marks
                    incorrect += 1

            responses.append(
                QuestionResponse(
                    question_id=q.question_id,
                    selected=selected,
                    status=status,
                    confidence=float(confidence),
                    options_fill_ratio={k: float(v) for k, v in fill.items()},
                )
            )

        return OMRResult(
            responses=responses,
            score=OMRScore(
                obtained=float(obtained),
                maximum=float(maximum),
                correct=correct,
                incorrect=incorrect,
                blank=blank,
            ),
            answer_key_used=bool(answer_key),
        )

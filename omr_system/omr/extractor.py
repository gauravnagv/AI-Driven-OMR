from __future__ import annotations

import cv2
import numpy as np

from omr_system.models import OMRResult, OMRScore, QualityMetrics, QuestionResponse, SectionScore
from omr_system.omr.template import OMRTemplate


class OMRExtractor:
    def _bubble_fill_ratio(
        self,
        binary: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> float:
        inner_radius = max(1, int(radius * 0.6))
        outer_radius = max(inner_radius + 1, int(radius * 1.15))
        inner_mask = np.zeros(binary.shape, dtype=np.uint8)
        outer_mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.circle(inner_mask, (cx, cy), inner_radius, 255, -1)
        cv2.circle(outer_mask, (cx, cy), outer_radius, 255, -1)
        ring_mask = cv2.subtract(outer_mask, inner_mask)

        inner_selected = cv2.bitwise_and(binary, binary, mask=inner_mask)
        ring_selected = cv2.bitwise_and(binary, binary, mask=ring_mask)
        inner_total = cv2.countNonZero(inner_mask)
        ring_total = cv2.countNonZero(ring_mask)
        if inner_total <= 0:
            return 0.0
        inner_ratio = cv2.countNonZero(inner_selected) / inner_total
        ring_ratio = cv2.countNonZero(ring_selected) / max(1, ring_total)
        normalized = max(0.0, inner_ratio - (0.55 * ring_ratio))
        robust_score = max(inner_ratio, normalized)
        return float(min(1.0, robust_score))

    def extract(
        self,
        image: np.ndarray,
        template: OMRTemplate,
        answer_key: dict[str, str] | None = None,
        detection_confidence: float | None = None,
    ) -> OMRResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
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
        section_stats: dict[str, dict] = {}

        for q in template.questions:
            fill = {}
            for choice in q.choices:
                cx = int(choice.x * sx)
                cy = int(choice.y * sy)
                radius = int(choice.radius * min(sx, sy))
                fill[choice.label] = self._bubble_fill_ratio(binary, cx, cy, radius)

            sorted_fill = sorted(fill.items(), key=lambda item: item[1], reverse=True)
            top_label, top_ratio = sorted_fill[0] if sorted_fill else ("", 0.0)
            second_ratio = sorted_fill[1][1] if len(sorted_fill) > 1 else 0.0
            margin = max(0.0, top_ratio - second_ratio)
            min_margin = max(0.02, template.fill_threshold * 0.15)
            marked = [label for label, ratio in fill.items() if ratio >= template.fill_threshold]

            relaxed_threshold = template.fill_threshold * 0.55
            if top_ratio < relaxed_threshold:
                status = "blank"
                selected = None
                confidence = top_ratio
                blank += 1
            elif len(marked) == 1 and margin >= min_margin:
                status = "single"
                selected = top_label
                confidence = min(1.0, top_ratio + margin)
            elif margin >= (min_margin * 1.5):
                status = "single"
                selected = top_label
                confidence = min(1.0, top_ratio + margin)
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

            section_entry = section_stats.setdefault(
                q.section_id,
                {
                    "section_title": q.section_title,
                    "obtained": 0.0,
                    "maximum": 0.0,
                    "correct": 0,
                    "incorrect": 0,
                    "blank": 0,
                    "total_questions": 0,
                },
            )
            section_entry["total_questions"] += 1
            if q.question_id in answer_key:
                section_entry["maximum"] += q.marks
                if status == "single" and selected == answer_key[q.question_id]:
                    section_entry["obtained"] += q.marks
                    section_entry["correct"] += 1
                elif status == "blank":
                    section_entry["blank"] += 1
                else:
                    section_entry["obtained"] -= q.negative_marks
                    section_entry["incorrect"] += 1

            responses.append(
                QuestionResponse(
                    question_id=q.question_id,
                    section_id=q.section_id,
                    selected=selected,
                    expected=answer_key.get(q.question_id),
                    is_correct=(status == "single" and selected == answer_key[q.question_id])
                    if q.question_id in answer_key
                    else None,
                    status=status,
                    confidence=float(confidence),
                    selection_margin=float(margin),
                    options_fill_ratio={k: float(v) for k, v in fill.items()},
                )
            )

        sections = [
            SectionScore(
                section_id=section_id,
                section_title=entry["section_title"],
                obtained=float(entry["obtained"]),
                maximum=float(entry["maximum"]),
                correct=entry["correct"],
                incorrect=entry["incorrect"],
                blank=entry["blank"],
                total_questions=entry["total_questions"],
            )
            for section_id, entry in section_stats.items()
        ]
        summary = (
            f"Score {obtained:.2f}/{maximum:.2f} | Correct: {correct}, Incorrect: {incorrect}, Blank: {blank}"
            if maximum > 0
            else f"Processed {len(responses)} questions (no answer key provided)."
        )
        avg_conf = float(np.mean([item.confidence for item in responses])) if responses else 0.0
        avg_margin = float(np.mean([item.selection_margin for item in responses])) if responses else 0.0
        detection_confidence = float(detection_confidence or 0.0)
        estimated_accuracy = max(
            35.0,
            min(99.0, (detection_confidence * 45.0) + (avg_conf * 35.0) + (avg_margin * 20.0)) * 100.0 / 100.0,
        )
        recommendation = (
            "High confidence scan."
            if estimated_accuracy >= 88
            else "Medium confidence: keep sheet flatter and increase lighting."
            if estimated_accuracy >= 72
            else "Low confidence: keep entire sheet visible, remove glare, and rescan."
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
            sections=sections,
            summary=summary,
            quality=QualityMetrics(
                detection_confidence=detection_confidence,
                average_question_confidence=avg_conf,
                average_selection_margin=avg_margin,
                estimated_accuracy_percent=float(round(estimated_accuracy, 2)),
                recommendation=recommendation,
            ),
            answer_key_used=bool(answer_key),
        )

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from omr_system.models import DetectionResult

logger = logging.getLogger(__name__)


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(s)]  # top-left
    ordered[2] = points[np.argmax(s)]  # bottom-right
    ordered[1] = points[np.argmin(diff)]  # top-right
    ordered[3] = points[np.argmax(diff)]  # bottom-left
    return ordered


class PageDetector:
    def __init__(self, weights_path: Path | None, confidence: float = 0.35) -> None:
        self.weights_path = weights_path
        self.confidence = confidence
        self._yolo = None
        self._load_yolo()

    def _load_yolo(self) -> None:
        if self.weights_path is None or not self.weights_path.exists():
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(str(self.weights_path))
            logger.info("Loaded YOLOv8 page detector: %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLOv8 unavailable. Falling back to contour detection: %s", exc)
            self._yolo = None

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self._yolo is not None:
            result = self._detect_with_yolo(image)
            if result is not None:
                return result
        return self._detect_with_contours(image)

    def _detect_with_yolo(self, image: np.ndarray) -> DetectionResult | None:
        prediction = self._yolo.predict(image, conf=self.confidence, verbose=False)
        if not prediction or len(prediction[0].boxes) == 0:
            return None
        boxes = prediction[0].boxes
        best = boxes[boxes.conf.argmax().item()]
        x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].tolist()]
        quad = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return DetectionResult(
            bbox_xyxy=(x1, y1, x2, y2),
            quadrilateral_xy=quad,
            method="yolov8",
            confidence=float(best.conf[0].item()),
        )

    def _detect_with_contours(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            h, w = gray.shape
            return DetectionResult(
                bbox_xyxy=(0, 0, w - 1, h - 1),
                quadrilateral_xy=[(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
                method="full-frame",
                confidence=0.1,
            )

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) >= 4:
            pts = approx.reshape(-1, 2).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)

        quad = _order_quad(box).astype(int)
        x_values = quad[:, 0]
        y_values = quad[:, 1]
        x1, x2 = int(x_values.min()), int(x_values.max())
        y1, y2 = int(y_values.min()), int(y_values.max())

        return DetectionResult(
            bbox_xyxy=(x1, y1, x2, y2),
            quadrilateral_xy=[tuple(map(int, p)) for p in quad.tolist()],
            method="contour-fallback",
            confidence=0.6,
        )


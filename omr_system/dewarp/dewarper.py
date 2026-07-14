from __future__ import annotations

import cv2
import numpy as np


def _order_points(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


class Dewarper:
    def __init__(self, output_width: int = 1800, output_height: int = 2400) -> None:
        self.output_width = output_width
        self.output_height = output_height

    def rectify(self, image: np.ndarray, quadrilateral_xy: list[tuple[int, int]]) -> np.ndarray:
        src = _order_points(np.array(quadrilateral_xy, dtype=np.float32))
        dst = np.array(
            [
                [0, 0],
                [self.output_width - 1, 0],
                [self.output_width - 1, self.output_height - 1],
                [0, self.output_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (self.output_width, self.output_height))

    def flatten_curvature(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            10,
        )
        h, w = binary.shape
        top = np.zeros(w, dtype=np.float32)
        bottom = np.full(w, h - 1, dtype=np.float32)

        for x in range(w):
            ys = np.where(binary[:, x] > 0)[0]
            if ys.size > 0:
                top[x] = float(ys[0])
                bottom[x] = float(ys[-1])
            else:
                top[x] = 0.0
                bottom[x] = float(h - 1)

        kernel = np.ones(51) / 51
        top_smooth = np.convolve(top, kernel, mode="same")
        bottom_smooth = np.convolve(bottom, kernel, mode="same")

        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.zeros((h, w), dtype=np.float32)
        y_coords = np.arange(h, dtype=np.float32).reshape(-1, 1)

        for x in range(w):
            source_height = max(10.0, bottom_smooth[x] - top_smooth[x])
            norm = y_coords[:, 0] / max(1.0, h - 1)
            source_y = top_smooth[x] + norm * source_height
            map_y[:, x] = np.clip(source_y, 0, h - 1)

        return cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def dewarp(self, image: np.ndarray, quadrilateral_xy: list[tuple[int, int]]) -> np.ndarray:
        rectified = self.rectify(image, quadrilateral_xy)
        return self.flatten_curvature(rectified)

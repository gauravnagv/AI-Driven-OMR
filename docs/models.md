# Model Artifact Expectations

## YOLOv8 Boundary Detector

- Optional dependency: `pip install -e .[yolo]`
- Place model at: `runtime/models/page_detector.pt`
- Configure via `.env`:
  - `OMR_PAGE_DETECTOR_WEIGHTS=runtime/models/page_detector.pt`
  - `OMR_DETECTOR_CONFIDENCE=0.35`

If the model is absent or `ultralytics` is not installed, the system automatically uses the contour-based fallback detector.

## OMR Template

Use YAML schema shown in `config/template.example.yaml`. Coordinates are relative to the dewarped page dimensions (`page_width`, `page_height`).


# API Usage Examples

## Web UI

- Open: `http://localhost:8000/`
- Features:
  - Live camera capture via browser camera access
  - Real-time OMR sheet presence detection before scan
  - Image upload scanning
  - AI pipeline invocation through `/scan/sync`
  - Human-readable scoring summary + section-wise score table + pipeline step logs
  - AI quality panel (estimated accuracy %, detection confidence, scan recommendation)

## Presence Detection Endpoint

```bash
curl -X POST http://localhost:8000/detect/presence \
  -F "image=@./samples/sheet1.png"
```

## Sync Scan Endpoint (for UI/camera)

```bash
curl -X POST http://localhost:8000/scan/sync \
  -F "image=@./samples/sheet1.png" \
  -F "template_path=./config/template.example.yaml" \
  -F 'answer_key_json={"q1":"A","q2":"C"}' \
  -F "save_artifacts=true"
```

Response includes:
- `sheet_present`
- `processing_steps`
- `omr.sections` (section-wise scores)
- `omr.quality` with:
  - `estimated_accuracy_percent`
  - `detection_confidence`
  - `average_question_confidence`
  - `recommendation`

## Artifact Download Endpoint

```bash
curl "http://localhost:8000/artifacts?path=<absolute-path-inside-runtime>"
```

## Create Job

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["./samples/sheet1.png", "./samples/sheet2.png"],
    "template_path": "./config/template.example.yaml",
    "answer_key": {"q1":"A","q2":"C"},
    "save_artifacts": true
  }'
```

## Job Status

```bash
curl http://localhost:8000/jobs/<job_id>
```

## Job Result

```bash
curl http://localhost:8000/jobs/<job_id>/results
```

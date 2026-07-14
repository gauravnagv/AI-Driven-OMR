# API Usage Examples

## Web UI

- Open: `http://localhost:8000/`
- Features:
  - Live camera capture via browser camera access
  - Image upload scanning
  - AI pipeline invocation through `/scan/sync`

## Sync Scan Endpoint (for UI/camera)

```bash
curl -X POST http://localhost:8000/scan/sync \
  -F "image=@./samples/sheet1.png" \
  -F "template_path=./config/template.example.yaml" \
  -F 'answer_key_json={"q1":"A","q2":"C"}' \
  -F "save_artifacts=true"
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

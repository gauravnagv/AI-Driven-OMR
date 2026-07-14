# AI-Driven OMR (Production-Ready Baseline)

End-to-end document processing system for large-scale exam digitization with:
- **Page/document detection** (YOLOv8 if weights are provided, robust OpenCV fallback otherwise)
- **Geometric correction + de-warping** (perspective rectification + curvature flattening pass)
- **OMR extraction + scoring**
- **Batch execution**, **API**, and **queue/worker split** for scalable deployment

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Start API:
```bash
omr-system serve-api --host 0.0.0.0 --port 8000
```

Start worker:
```bash
omr-system run-worker
```

Submit a job:
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": ["./samples/sheet1.png"],
    "template_path": "./config/template.example.yaml",
    "answer_key": {"q1":"A","q2":"C"}
  }'
```

## Documentation

- Architecture: `docs/architecture.md`
- Deployment: `docs/deployment.md`
- API usage: `docs/api.md`
- Model artifact wiring: `docs/models.md`


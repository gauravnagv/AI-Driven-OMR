# API Usage Examples

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


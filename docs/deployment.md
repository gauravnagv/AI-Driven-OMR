# Local Run and Deployment Guide

## Local Run (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
mkdir -p runtime/queue runtime/artifacts runtime/models
omr-system serve-api --host 0.0.0.0 --port 8000
```

In another terminal:
```bash
source .venv/bin/activate
omr-system run-worker
```

## Docker Deployment

```bash
cp .env.example .env
mkdir -p runtime/queue runtime/artifacts runtime/models
docker compose up --build -d
docker compose ps
```

## Health Checks

- API: `GET /health`
- API readiness: `GET /ready`
- Worker: heartbeat file-based health check (`scripts/worker-healthcheck.sh`)


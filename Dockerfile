FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY omr_system /app/omr_system
COPY config /app/config
COPY scripts /app/scripts

RUN pip install --no-cache-dir -e .

RUN chmod +x /app/scripts/start-api.sh /app/scripts/start-worker.sh /app/scripts/worker-healthcheck.sh

EXPOSE 8000


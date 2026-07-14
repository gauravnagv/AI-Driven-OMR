# Architecture Overview

The system is split into independent production modules:

1. **API service (`omr_system.api`)**
   - Accepts processing jobs (`POST /jobs`)
   - Exposes status/result APIs
   - Writes jobs to a filesystem-backed queue

2. **Worker service (`omr_system.worker`)**
   - Polls and atomically claims queued jobs
   - Runs document pipeline in parallel (`ProcessPoolExecutor`)
   - Writes completed/failed outcomes for retrieval

3. **Pipeline (`omr_system.pipeline`)**
   - Page detection (`detection/page_detector.py`)
   - Geometric correction/de-warp (`dewarp/dewarper.py`)
   - OMR extraction and scoring (`omr/extractor.py`)

4. **Runtime layout**
   - `runtime/queue/pending|processing|done|failed`
   - `runtime/artifacts/<document-id>/...`
   - `runtime/models/page_detector.pt` (optional YOLOv8 weights)

## Processing Flow

```mermaid
flowchart LR
  A[Client] --> B[FastAPI /jobs]
  B --> C[Queue pending]
  D[Worker] --> C
  D --> E[Detect page: YOLOv8 or contour fallback]
  E --> F[De-warp + flatten curvature]
  F --> G[OMR extraction + scoring]
  G --> H[Queue done/failed]
  A --> I[GET /jobs/{id}]
  A --> J[GET /jobs/{id}/results]
```


from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    bbox_xyxy: tuple[int, int, int, int]
    quadrilateral_xy: list[tuple[int, int]]
    method: str
    confidence: float


class QuestionResponse(BaseModel):
    question_id: str
    selected: str | None
    status: Literal["single", "multiple", "blank"]
    confidence: float = 0.0
    options_fill_ratio: dict[str, float] = Field(default_factory=dict)


class OMRScore(BaseModel):
    obtained: float
    maximum: float
    correct: int
    incorrect: int
    blank: int


class OMRResult(BaseModel):
    responses: list[QuestionResponse]
    score: OMRScore
    answer_key_used: bool


class DocumentResult(BaseModel):
    input_path: str
    success: bool
    detection: DetectionResult | None = None
    omr: OMRResult | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class JobCreateRequest(BaseModel):
    inputs: list[str]
    template_path: str
    answer_key: dict[str, str] = Field(default_factory=dict)
    save_artifacts: bool = True


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    submitted_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class JobResultEnvelope(BaseModel):
    job_id: str
    status: str
    result: list[DocumentResult] | None = None
    error: str | None = None


class JobPayload(BaseModel):
    job_id: str
    submitted_at: str
    inputs: list[str]
    template_path: str
    answer_key: dict[str, str]
    save_artifacts: bool


class WorkerClaim(BaseModel):
    job_id: str
    processing_path: Path
    payload: JobPayload


JSONDict = dict[str, Any]


from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    bbox_xyxy: tuple[int, int, int, int]
    quadrilateral_xy: list[tuple[int, int]]
    method: str
    confidence: float
    area_ratio: float = 0.0


class ProcessingStep(BaseModel):
    name: str
    status: Literal["success", "failed", "skipped"]
    detail: str
    artifact_path: str | None = None


class PresenceCheckResponse(BaseModel):
    present: bool
    confidence: float
    reason: str
    detection: DetectionResult | None = None


class QuestionResponse(BaseModel):
    question_id: str
    section_id: str = "default"
    selected: str | None
    expected: str | None = None
    is_correct: bool | None = None
    status: Literal["single", "multiple", "blank"]
    confidence: float = 0.0
    selection_margin: float = 0.0
    options_fill_ratio: dict[str, float] = Field(default_factory=dict)


class OMRScore(BaseModel):
    obtained: float
    maximum: float
    correct: int
    incorrect: int
    blank: int


class SectionScore(BaseModel):
    section_id: str
    section_title: str
    obtained: float
    maximum: float
    correct: int
    incorrect: int
    blank: int
    total_questions: int


class QualityMetrics(BaseModel):
    detection_confidence: float
    average_question_confidence: float
    average_selection_margin: float
    estimated_accuracy_percent: float
    recommendation: str


class OMRResult(BaseModel):
    responses: list[QuestionResponse]
    score: OMRScore
    sections: list[SectionScore] = Field(default_factory=list)
    summary: str = ""
    quality: QualityMetrics | None = None
    answer_key_used: bool


class DocumentResult(BaseModel):
    input_path: str
    success: bool
    sheet_present: bool = False
    detection: DetectionResult | None = None
    omr: OMRResult | None = None
    processing_steps: list[ProcessingStep] = Field(default_factory=list)
    human_readable_summary: str | None = None
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

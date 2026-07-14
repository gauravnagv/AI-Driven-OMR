from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class BubbleChoice(BaseModel):
    label: str
    x: float
    y: float
    radius: float


class QuestionTemplate(BaseModel):
    question_id: str
    section_id: str = "default"
    section_title: str = "General"
    marks: float = 1.0
    negative_marks: float = 0.0
    choices: list[BubbleChoice]


class OMRTemplate(BaseModel):
    page_width: int
    page_height: int
    fill_threshold: float = 0.45
    questions: list[QuestionTemplate] = Field(default_factory=list)


def load_template(path: str | Path) -> OMRTemplate:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return OMRTemplate.model_validate(raw)

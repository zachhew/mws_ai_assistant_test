from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal["chat", "reasoning", "coding", "multimodal", "embeddings", "unknown"]
InputModality = Literal["text", "text_image", "unknown"]
Priority = Literal["low", "balanced", "high"]


class UserCaseProfile(BaseModel):
    task_type: TaskType = "unknown"
    input_modality: InputModality = "unknown"

    expected_input_tokens: int | None = Field(default=None, ge=0)
    expected_output_tokens: int | None = Field(default=None, ge=0)

    requests_per_day: int | None = Field(default=None, ge=0)
    requests_per_month: int | None = Field(default=None, ge=0)

    quality_priority: Priority = "balanced"
    latency_priority: Priority = "balanced"

    budget_limit_rub: float | None = Field(default=None, ge=0)

    needs_long_context: bool = False
    context_min_tokens: int | None = Field(default=None, ge=0)

    notes: str | None = None
    assumptions: list[str] = Field(default_factory=list)

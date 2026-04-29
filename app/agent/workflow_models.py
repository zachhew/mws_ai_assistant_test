from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.recommendation import CostEstimate
from app.domain.user_case import UserCaseProfile


class RankingOption(BaseModel):
    model_name: str
    model_family: str | None = None
    context_window_tokens: int | None = None
    supports_image_input: bool = False
    fit_summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    cost_estimate: CostEstimate | None = None


class RankingContextPayload(BaseModel):
    profile: UserCaseProfile
    options: list[RankingOption] = Field(default_factory=list)


class RankedModelDecision(BaseModel):
    model_name: str
    rationale: str = Field(min_length=1)


class RankingAgentOutput(BaseModel):
    recommended_models: list[RankedModelDecision] = Field(default_factory=list, max_length=3)

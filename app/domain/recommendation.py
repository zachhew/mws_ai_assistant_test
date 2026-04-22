from typing import Literal

from pydantic import BaseModel, Field

from app.domain.user_case import UserCaseProfile


FitLabel = Literal["best_fit", "budget_option", "premium_option", "alternative"]


class CostEstimate(BaseModel):
    model_name: str

    requests_per_month: int = Field(ge=0)
    input_tokens_per_request: int = Field(ge=0)
    output_tokens_per_request: int = Field(ge=0)

    monthly_input_tokens: int = Field(ge=0)
    monthly_output_tokens: int = Field(ge=0)

    monthly_input_cost_rub: float = Field(ge=0)
    monthly_output_cost_rub: float = Field(ge=0)
    total_monthly_cost_rub: float = Field(ge=0)

    within_budget: bool | None = None
    assumptions: list[str] = Field(default_factory=list)


class RecommendationCandidate(BaseModel):
    model_name: str
    score: float = Field(ge=0)

    fit_label: FitLabel | None = None
    fit_summary: str

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)

    cost_estimate: CostEstimate | None = None


class RecommendationReport(BaseModel):
    input_data: UserCaseProfile
    recommended_models: list[RecommendationCandidate] = Field(default_factory=list)
    calculations: list[CostEstimate] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)

    final_summary: str
    final_answer_text: str | None = None

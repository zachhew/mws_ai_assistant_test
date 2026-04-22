from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.catalog import CatalogEntry
from app.domain.recommendation import (
    CostEstimate,
    RecommendationCandidate,
    RecommendationReport,
)
from app.domain.user_case import UserCaseProfile


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionState(BaseModel):
    session_id: str

    message_history: list[SessionMessage] = Field(default_factory=list)

    last_user_case: UserCaseProfile | None = None
    last_catalog: list[CatalogEntry] | None = None
    last_cost_estimates: list[CostEstimate] | None = None
    last_recommendations: list[RecommendationCandidate] | None = None
    last_report: RecommendationReport | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

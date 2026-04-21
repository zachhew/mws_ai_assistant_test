from datetime import datetime, timedelta, timezone

from app.domain.catalog import CatalogEntry
from app.domain.recommendation import (
    CostEstimate,
    RecommendationCandidate,
    RecommendationReport,
)
from app.domain.session import SessionMessage, SessionState
from app.domain.user_case import UserCaseProfile


class SessionManager:
    def __init__(self, ttl_minutes: int = 60) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._store: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        self.cleanup_expired()
        return self._store.get(session_id)

    def get_or_create(self, session_id: str) -> SessionState:
        self.cleanup_expired()

        state = self._store.get(session_id)
        if state is not None:
            return state

        state = SessionState(session_id=session_id)
        self._store[session_id] = state
        return state

    def update_messages(
        self,
        session_id: str,
        messages: list[SessionMessage],
    ) -> SessionState:
        state = self.get_or_create(session_id)
        state.message_history = messages
        state.updated_at = datetime.now(timezone.utc)
        self._store[session_id] = state
        return state

    def save_artifacts(
        self,
        session_id: str,
        profile: UserCaseProfile | None = None,
        catalog: list[CatalogEntry] | None = None,
        costs: list[CostEstimate] | None = None,
        recommendations: list[RecommendationCandidate] | None = None,
        report: RecommendationReport | None = None,
    ) -> SessionState:
        state = self.get_or_create(session_id)

        if profile is not None:
            state.last_user_case = profile
        if catalog is not None:
            state.last_catalog = catalog
        if costs is not None:
            state.last_cost_estimates = costs
        if recommendations is not None:
            state.last_recommendations = recommendations
        if report is not None:
            state.last_report = report

        state.updated_at = datetime.now(timezone.utc)
        self._store[session_id] = state
        return state

    def cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired_keys = [
            session_id
            for session_id, state in self._store.items()
            if now - state.updated_at > self._ttl
        ]
        for session_id in expired_keys:
            del self._store[session_id]
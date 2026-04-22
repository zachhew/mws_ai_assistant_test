from __future__ import annotations

from app.api.schemas import ChatCompletionRequest
from app.agent.session_manager import SessionManager
from app.core.logging import get_logger
from app.domain.recommendation import RecommendationReport
from app.domain.session import SessionMessage, SessionState
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.explanation_service import ExplanationService
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


class ChatCompletionsCoordinator:
    def __init__(
        self,
        session_manager: SessionManager,
        profile_service: ProfileService,
        catalog_service: CatalogService,
        estimator: CostEstimator,
        recommender: ModelRecommender,
        report_builder: ReportBuilder,
        explanation_service: ExplanationService,
        agent_model: str,
    ) -> None:
        self._session_manager = session_manager
        self._profile_service = profile_service
        self._catalog_service = catalog_service
        self._estimator = estimator
        self._recommender = recommender
        self._report_builder = report_builder
        self._explanation_service = explanation_service
        self._logger = get_logger(self.__class__.__name__)
        self._agent_model = agent_model

    def handle(
        self,
        request: ChatCompletionRequest,
        session_id: str,
    ) -> RecommendationReport:
        self._logger.info("Handling request. session_id=%s", session_id)

        state = self._resolve_session(session_id, request)
        profile = self._build_profile(state)

        full_catalog = self._catalog_service.get_catalog(force_refresh=False)
        filtered_catalog = self._recommender.filter_catalog(profile, full_catalog)

        costs = self._estimator.estimate_for_catalog(profile, filtered_catalog)
        recommendations = self._recommender.recommend(profile, filtered_catalog, costs)
        report = self._report_builder.build(profile, recommendations, costs)

        llm_summary = self._explanation_service.generate_summary(report)
        report.final_summary = llm_summary

        self._session_manager.save_artifacts(
            session_id=session_id,
            profile=profile,
            catalog=filtered_catalog,
            costs=costs,
            recommendations=recommendations,
            report=report,
        )

        self._logger.info(
            "Request handled successfully. session_id=%s recommendations=%s",
            session_id,
            len(recommendations),
        )
        return report

    def _resolve_session(
        self,
        session_id: str,
        request: ChatCompletionRequest,
    ) -> SessionState:
        messages = [
            SessionMessage(role=message.role, content=message.content)
            for message in request.messages
        ]
        return self._session_manager.update_messages(session_id, messages)

    def _build_profile(self, state: SessionState) -> RecommendationReport | object:
        from app.api.schemas import ChatMessage

        messages = [
            ChatMessage(role=message.role, content=message.content)
            for message in state.message_history
        ]
        return self._profile_service.build_profile(
            messages=messages,
            previous_profile=state.last_user_case,
        )
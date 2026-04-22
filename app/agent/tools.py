from __future__ import annotations

from app.api.schemas import ChatMessage
from app.agent.session_manager import SessionManager
from app.domain.catalog import CatalogEntry
from app.domain.recommendation import CostEstimate, RecommendationCandidate, RecommendationReport
from app.domain.user_case import UserCaseProfile
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


class AssistantTools:
    def __init__(
        self,
        session_manager: SessionManager,
        profile_service: ProfileService,
        catalog_service: CatalogService,
        estimator: CostEstimator,
        recommender: ModelRecommender,
        report_builder: ReportBuilder,
    ) -> None:
        self._session_manager = session_manager
        self._profile_service = profile_service
        self._catalog_service = catalog_service
        self._estimator = estimator
        self._recommender = recommender
        self._report_builder = report_builder

    def build_usage_profile(
        self,
        session_id: str,
        messages: list[dict[str, str]],
    ) -> dict:
        state = self._session_manager.get_or_create(session_id)
        chat_messages = [ChatMessage(**message) for message in messages]
        profile = self._profile_service.build_profile(
            messages=chat_messages,
            previous_profile=state.last_user_case,
        )
        self._session_manager.save_artifacts(session_id=session_id, profile=profile)
        return profile.model_dump()

    def load_mws_catalog(
        self,
        session_id: str,
        force_refresh: bool = False,
    ) -> list[dict]:
        catalog = self._catalog_service.get_catalog(force_refresh=force_refresh)
        self._session_manager.save_artifacts(session_id=session_id, catalog=catalog)
        return [entry.model_dump() for entry in catalog]

    def estimate_costs(
        self,
        session_id: str,
        profile: dict,
        catalog: list[dict],
    ) -> list[dict]:
        profile_obj = UserCaseProfile(**profile)
        catalog_objs = [CatalogEntry(**entry) for entry in catalog]
        filtered_catalog = self._recommender.filter_catalog(profile_obj, catalog_objs)
        costs = self._estimator.estimate_for_catalog(profile_obj, filtered_catalog)
        self._session_manager.save_artifacts(session_id=session_id, costs=costs, catalog=filtered_catalog)
        return [item.model_dump() for item in costs]

    def recommend_models(
        self,
        session_id: str,
        profile: dict,
        catalog: list[dict],
        costs: list[dict],
    ) -> list[dict]:
        profile_obj = UserCaseProfile(**profile)
        catalog_objs = [CatalogEntry(**entry) for entry in catalog]
        cost_objs = [CostEstimate(**item) for item in costs]
        recommendations = self._recommender.recommend(profile_obj, catalog_objs, cost_objs)
        self._session_manager.save_artifacts(session_id=session_id, recommendations=recommendations)
        return [item.model_dump() for item in recommendations]

    def build_report(
        self,
        session_id: str,
        profile: dict,
        recommendations: list[dict],
        costs: list[dict],
    ) -> dict:
        profile_obj = UserCaseProfile(**profile)
        recommendation_objs = [RecommendationCandidate(**item) for item in recommendations]
        cost_objs = [CostEstimate(**item) for item in costs]
        report = self._report_builder.build(profile_obj, recommendation_objs, cost_objs)
        self._session_manager.save_artifacts(session_id=session_id, report=report)
        return report.model_dump()

    def get_last_report(self, session_id: str) -> dict | None:
        state = self._session_manager.get(session_id)
        if state is None or state.last_report is None:
            return None
        return state.last_report.model_dump()
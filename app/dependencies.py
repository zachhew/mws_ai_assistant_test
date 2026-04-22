from __future__ import annotations

from functools import lru_cache

from app.agent.coordinator import ChatCompletionsCoordinator
from app.agent.session_manager import SessionManager
from app.core.config import get_settings
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.explanation_service import ExplanationService
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    settings = get_settings()
    return SessionManager(ttl_minutes=settings.default_session_ttl_minutes)


@lru_cache(maxsize=1)
def get_mws_client() -> MWSClient:
    return MWSClient()


@lru_cache(maxsize=1)
def get_mws_parser() -> MWSParser:
    return MWSParser()


@lru_cache(maxsize=1)
def get_catalog_service() -> CatalogService:
    settings = get_settings()
    return CatalogService(
        models_url=settings.mws_models_url,
        pricing_url=settings.mws_pricing_url,
        client=get_mws_client(),
        parser=get_mws_parser(),
    )


@lru_cache(maxsize=1)
def get_profile_service() -> ProfileService:
    return ProfileService()


@lru_cache(maxsize=1)
def get_cost_estimator() -> CostEstimator:
    return CostEstimator()


@lru_cache(maxsize=1)
def get_model_recommender() -> ModelRecommender:
    return ModelRecommender()


@lru_cache(maxsize=1)
def get_report_builder() -> ReportBuilder:
    return ReportBuilder()


@lru_cache(maxsize=1)
def get_explanation_service() -> ExplanationService:
    return ExplanationService()


@lru_cache(maxsize=1)
def get_chat_completions_coordinator() -> ChatCompletionsCoordinator:
    settings = get_settings()
    return ChatCompletionsCoordinator(
        session_manager=get_session_manager(),
        profile_service=get_profile_service(),
        catalog_service=get_catalog_service(),
        estimator=get_cost_estimator(),
        recommender=get_model_recommender(),
        report_builder=get_report_builder(),
        explanation_service=get_explanation_service(),
        agent_model=settings.adk_agent_model,
    )
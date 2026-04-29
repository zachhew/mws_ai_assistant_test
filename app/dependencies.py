from __future__ import annotations

from functools import lru_cache

from app.agent.adk_runtime import AdkRuntimeService
from app.agent.coordinator import ChatCompletionsCoordinator
from app.agent.session_manager import SessionManager
from app.core.config import Settings, get_settings
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.final_answer_streaming_service import FinalAnswerStreamingService
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser
from app.services.mws_recovery_client import MWSRecoveryClient
from app.services.openrouter_client import OpenRouterClient
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
def get_openrouter_client() -> OpenRouterClient:
    return OpenRouterClient()


@lru_cache(maxsize=1)
def get_mws_recovery_client() -> MWSRecoveryClient:
    return MWSRecoveryClient(llm_client=get_openrouter_client())


@lru_cache(maxsize=1)
def get_catalog_service() -> CatalogService:
    settings = get_settings()
    return CatalogService(
        models_url=settings.mws_models_url,
        pricing_url=settings.mws_pricing_url,
        client=get_mws_client(),
        parser=get_mws_parser(),
        recovery_client=get_mws_recovery_client(),
        cache_ttl_seconds=settings.catalog_cache_ttl_seconds,
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
def get_adk_runtime_service() -> AdkRuntimeService:
    settings: Settings = get_settings()
    return AdkRuntimeService(
        settings=settings,
        session_manager=get_session_manager(),
        profile_service=get_profile_service(),
        catalog_service=get_catalog_service(),
        estimator=get_cost_estimator(),
        recommender=get_model_recommender(),
        report_builder=get_report_builder(),
    )


@lru_cache(maxsize=1)
def get_chat_completions_coordinator() -> ChatCompletionsCoordinator:
    return ChatCompletionsCoordinator(
        adk_runtime=get_adk_runtime_service(),
    )


@lru_cache(maxsize=1)
def get_final_answer_streaming_service() -> FinalAnswerStreamingService:
    return FinalAnswerStreamingService()

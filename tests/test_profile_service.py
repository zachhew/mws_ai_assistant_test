from app.agent.session_manager import SessionManager
from app.agent.workflow_agents import RecommendationPreparationAgent
from app.domain.user_case import UserCaseProfile
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender


def test_finalize_profile_merges_previous_profile_and_fills_defaults() -> None:
    service = ProfileService()
    previous = UserCaseProfile(
        task_type="chat",
        input_modality="text",
        requests_per_month=12000,
        expected_input_tokens=700,
        expected_output_tokens=250,
        assumptions=["Previous profile available."],
    )
    current = UserCaseProfile(
        task_type="unknown",
        input_modality="unknown",
        budget_limit_rub=15000,
        assumptions=["Extracted from latest message."],
    )

    profile = service.finalize_profile(current, previous)

    assert profile.task_type == "chat"
    assert profile.input_modality == "text"
    assert profile.budget_limit_rub == 15000
    assert profile.requests_per_month == 12000
    assert "Previous profile available." in profile.assumptions
    assert "Extracted from latest message." in profile.assumptions


def test_profile_payload_normalization_handles_non_list_assumptions_and_aliases() -> None:
    agent = RecommendationPreparationAgent(
        name="test_agent",
        session_manager=SessionManager(),
        profile_service=ProfileService(),
        catalog_service=CatalogService.__new__(CatalogService),
        estimator=CostEstimator(),
        recommender=ModelRecommender(),
    )

    normalized = agent._normalize_profile_payload(
        {
            "task_type": "embeddings",
            "input_modality": "text",
            "monthly_requests": 200000,
            "input_tokens": "350",
            "budget": 10000,
            "needs_long_context": "true",
            "assumptions": {"monthly_requests": 200000, "input_tokens": 350, "budget": 10000},
        }
    )

    assert normalized["requests_per_month"] == 200000
    assert normalized["expected_input_tokens"] == 350
    assert normalized["budget_limit_rub"] == 10000.0
    assert normalized["needs_long_context"] is True
    assert normalized["assumptions"] == [
        "monthly_requests: 200000",
        "input_tokens: 350",
        "budget: 10000",
    ]


def test_extract_profile_from_text_keeps_explicit_embedding_request_numbers() -> None:
    service = ProfileService()

    profile = service.extract_profile_from_text(
        "Нужна модель для построения эмбеддингов для поиска по документам. "
        "Около 200000 запросов в месяц, примерно 350 токенов на вход, бюджет 10000 рублей."
    )

    assert profile.task_type == "embeddings"
    assert profile.input_modality == "text"
    assert profile.requests_per_month == 200000
    assert profile.expected_input_tokens == 350
    assert profile.budget_limit_rub == 10000.0


def test_merge_profiles_uses_deterministic_numbers_when_llm_misses_them() -> None:
    service = ProfileService()
    deterministic = service.extract_profile_from_text(
        "Нужна модель для построения эмбеддингов для поиска по документам. "
        "Около 200000 запросов в месяц, примерно 350 токенов на вход, бюджет 10000 рублей."
    )
    llm_profile = UserCaseProfile(
        task_type="embeddings",
        input_modality="text",
        requests_per_month=200000,
        budget_limit_rub=10000,
        assumptions=["Extracted by LLM."],
    )

    merged = service.merge_profiles(deterministic, llm_profile)
    finalized = service.finalize_profile(merged)

    assert finalized.expected_input_tokens == 350
    assert finalized.expected_output_tokens == 0
    assert "Input tokens per request defaulted to 500." not in finalized.assumptions

from app.domain.catalog import CatalogEntry, ModelSpec, PricingSpec
from app.domain.recommendation import CostEstimate
from app.domain.user_case import UserCaseProfile
from app.services.recommender import ModelRecommender


def make_catalog_entry(
    name: str,
    *,
    supports_image_input: bool = False,
    is_embedding_model: bool = False,
    context_window_tokens: int = 40000,
    family: str = "qwen",
) -> CatalogEntry:
    return CatalogEntry(
        model=ModelSpec(
            name=name,
            input_modalities=["text", "image"] if supports_image_input else ["text"],
            output_modalities=["embedding"] if is_embedding_model else ["text"],
            context_window_tokens=context_window_tokens,
            model_size_label="32",
            family=family,
            supports_text_input=True,
            supports_image_input=supports_image_input,
            supports_text_output=not is_embedding_model,
            is_embedding_model=is_embedding_model,
            source_url="https://example.com/models",
        ),
        pricing=PricingSpec(
            model_name=name,
            input_price_per_1k_tokens_rub=1.0,
            output_price_per_1k_tokens_rub=None if is_embedding_model else 1.0,
            billing_unit_tokens=100,
            source_url="https://example.com/pricing",
        ),
    )


def make_cost_estimate(
    model_name: str,
    total_monthly_cost_rub: float,
    within_budget: bool | None,
) -> CostEstimate:
    return CostEstimate(
        model_name=model_name,
        requests_per_month=1000,
        input_tokens_per_request=500,
        output_tokens_per_request=200,
        monthly_input_tokens=500000,
        monthly_output_tokens=200000,
        monthly_input_cost_rub=total_monthly_cost_rub / 2,
        monthly_output_cost_rub=total_monthly_cost_rub / 2,
        total_monthly_cost_rub=total_monthly_cost_rub,
        within_budget=within_budget,
        assumptions=[],
    )


def test_filter_catalog_excludes_embedding_models_for_chat() -> None:
    recommender = ModelRecommender()
    profile = UserCaseProfile(task_type="chat", input_modality="text")

    catalog = [
        make_catalog_entry("chat-model"),
        make_catalog_entry("embedding-model", is_embedding_model=True),
    ]

    filtered = recommender.filter_catalog(profile, catalog)

    assert len(filtered) == 1
    assert filtered[0].model.name == "chat-model"


def test_filter_catalog_keeps_only_multimodal_models_for_text_image() -> None:
    recommender = ModelRecommender()
    profile = UserCaseProfile(task_type="multimodal", input_modality="text_image")

    catalog = [
        make_catalog_entry("text-only-model", supports_image_input=False),
        make_catalog_entry("multimodal-model", supports_image_input=True),
    ]

    filtered = recommender.filter_catalog(profile, catalog)

    assert len(filtered) == 1
    assert filtered[0].model.name == "multimodal-model"


def test_recommend_prioritizes_in_budget_candidates() -> None:
    recommender = ModelRecommender()
    profile = UserCaseProfile(
        task_type="chat",
        input_modality="text",
        quality_priority="balanced",
        budget_limit_rub=12000,
    )

    catalog = [
        make_catalog_entry("in-budget-model"),
        make_catalog_entry("over-budget-model"),
    ]

    cost_estimates = [
        make_cost_estimate("in-budget-model", 5000.0, True),
        make_cost_estimate("over-budget-model", 50000.0, False),
    ]

    recommendations = recommender.recommend(profile, catalog, cost_estimates)

    assert len(recommendations) >= 2
    assert recommendations[0].model_name == "in-budget-model"
    assert recommendations[0].fit_label == "best_fit"


def test_recommend_uses_alternative_labels_if_nothing_fits_budget() -> None:
    recommender = ModelRecommender()
    profile = UserCaseProfile(
        task_type="chat",
        input_modality="text",
        quality_priority="balanced",
        budget_limit_rub=1000,
    )

    catalog = [
        make_catalog_entry("model-a"),
        make_catalog_entry("model-b"),
        make_catalog_entry("model-c"),
    ]

    cost_estimates = [
        make_cost_estimate("model-a", 10000.0, False),
        make_cost_estimate("model-b", 12000.0, False),
        make_cost_estimate("model-c", 15000.0, False),
    ]

    recommendations = recommender.recommend(profile, catalog, cost_estimates)

    assert recommendations[0].fit_label == "best_fit"
    assert recommendations[1].fit_label == "alternative"
    assert recommendations[2].fit_label == "alternative"

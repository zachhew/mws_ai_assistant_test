from app.domain.catalog import CatalogEntry, ModelSpec, PricingSpec
from app.domain.user_case import UserCaseProfile
from app.services.estimator import CostEstimator


def test_estimate_for_model_calculates_monthly_cost_correctly() -> None:
    estimator = CostEstimator()

    profile = UserCaseProfile(
        task_type="chat",
        input_modality="text",
        expected_input_tokens=700,
        expected_output_tokens=250,
        requests_per_month=50000,
        budget_limit_rub=12000,
    )

    entry = CatalogEntry(
        model=ModelSpec(
            name="qwen3-32b",
            input_modalities=["text"],
            output_modalities=["text"],
            context_window_tokens=40000,
            model_size_label="32",
            family="qwen",
            supports_text_input=True,
            supports_image_input=False,
            supports_text_output=True,
            is_embedding_model=False,
            source_url="https://example.com/models",
        ),
        pricing=PricingSpec(
            model_name="qwen3-32b",
            input_price_per_1k_tokens_rub=1.098,
            output_price_per_1k_tokens_rub=1.098,
            billing_unit_tokens=100,
            source_url="https://example.com/pricing",
        ),
    )

    estimate = estimator.estimate_for_model(profile, entry)

    assert estimate is not None
    assert estimate.monthly_input_tokens == 35_000_000
    assert estimate.monthly_output_tokens == 12_500_000
    assert estimate.monthly_input_cost_rub == 38_430.0
    assert estimate.monthly_output_cost_rub == 13_725.0
    assert estimate.total_monthly_cost_rub == 52_155.0
    assert estimate.within_budget is False


def test_estimate_for_model_returns_none_without_pricing() -> None:
    estimator = CostEstimator()

    profile = UserCaseProfile(
        task_type="chat",
        input_modality="text",
        expected_input_tokens=500,
        expected_output_tokens=200,
        requests_per_month=1000,
    )

    entry = CatalogEntry(
        model=ModelSpec(
            name="tests-model",
            input_modalities=["text"],
            output_modalities=["text"],
            context_window_tokens=8000,
            model_size_label="7",
            family="tests",
            supports_text_input=True,
            supports_image_input=False,
            supports_text_output=True,
            is_embedding_model=False,
            source_url="https://example.com/models",
        ),
        pricing=None,
    )

    estimate = estimator.estimate_for_model(profile, entry)

    assert estimate is None

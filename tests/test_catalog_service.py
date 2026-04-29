from app.domain.catalog import ModelSpec, PricingSpec
from app.services.catalog_service import CatalogService


class MutableClock:
    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class StubMWSClient:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = responses
        self.calls: dict[str, int] = {url: 0 for url in responses}

    def fetch_text(self, url: str) -> str:
        call_index = self.calls[url]
        values = self._responses[url]
        self.calls[url] += 1
        if call_index >= len(values):
            return values[-1]
        return values[call_index]


class StubParser:
    def __init__(self) -> None:
        self.models_parse_calls = 0
        self.pricing_parse_calls = 0

    def parse_models_page(self, html: str, source_url: str) -> list[ModelSpec]:
        self.models_parse_calls += 1
        return [
            ModelSpec(
                name=html.strip(),
                input_modalities=["text"],
                output_modalities=["text"],
                context_window_tokens=32000,
                model_size_label="32",
                family="qwen",
                supports_text_input=True,
                supports_image_input=False,
                supports_text_output=True,
                is_embedding_model=False,
                source_url=source_url,
            )
        ]

    def parse_pricing_page(self, html: str, source_url: str) -> list[PricingSpec]:
        self.pricing_parse_calls += 1
        return [
            PricingSpec(
                model_name="model-v1" if "v1" in html else "model-v2",
                input_price_per_1k_tokens_rub=1.0,
                output_price_per_1k_tokens_rub=1.0,
                billing_unit_tokens=1000,
                source_url=source_url,
            )
        ]

    def extract_table_rows(self, html: str) -> list[list[str]]:
        return [[html.strip()]]


class FailingParser(StubParser):
    def parse_models_page(self, html: str, source_url: str) -> list[ModelSpec]:
        self.models_parse_calls += 1
        raise ValueError("models parse failed")

    def parse_pricing_page(self, html: str, source_url: str) -> list[PricingSpec]:
        self.pricing_parse_calls += 1
        raise ValueError("pricing parse failed")


class StubRecoveryClient:
    def __init__(self) -> None:
        self.model_recovery_calls = 0
        self.pricing_recovery_calls = 0

    def recover_models(self, rows: list[list[str]], source_url: str) -> list[ModelSpec]:
        self.model_recovery_calls += 1
        assert rows == [["model-v2"]]
        return [
            ModelSpec(
                name="recovered-model",
                input_modalities=["text"],
                output_modalities=["text"],
                context_window_tokens=64000,
                model_size_label="16",
                family="unknown",
                supports_text_input=True,
                supports_image_input=False,
                supports_text_output=True,
                is_embedding_model=False,
                source_url=source_url,
            )
        ]

    def recover_pricing(self, rows: list[list[str]], source_url: str) -> list[PricingSpec]:
        self.pricing_recovery_calls += 1
        assert rows == [["pricing-v2"]]
        return [
            PricingSpec(
                model_name="recovered-model",
                input_price_per_1k_tokens_rub=2.5,
                output_price_per_1k_tokens_rub=3.5,
                billing_unit_tokens=1000,
                source_url=source_url,
            )
        ]


def test_catalog_service_reuses_cached_catalog_for_unchanged_pages() -> None:
    clock = MutableClock()
    client = StubMWSClient(
        {
            "models": ["model-v1", "model-v1"],
            "pricing": ["pricing-v1", "pricing-v1"],
        }
    )
    parser = StubParser()
    service = CatalogService(
        models_url="models",
        pricing_url="pricing",
        client=client,
        parser=parser,
        cache_ttl_seconds=60,
        time_provider=clock,
    )

    first = service.get_catalog()
    second = service.get_catalog()

    assert first is second
    assert client.calls["models"] == 1
    assert client.calls["pricing"] == 1
    assert parser.models_parse_calls == 1
    assert parser.pricing_parse_calls == 1


def test_catalog_service_revalidates_without_rebuild_when_ttl_expires_but_source_is_unchanged(
) -> None:
    clock = MutableClock()
    client = StubMWSClient(
        {
            "models": ["model-v1", "model-v1"],
            "pricing": ["pricing-v1", "pricing-v1"],
        }
    )
    parser = StubParser()
    service = CatalogService(
        models_url="models",
        pricing_url="pricing",
        client=client,
        parser=parser,
        cache_ttl_seconds=60,
        time_provider=clock,
    )

    first = service.get_catalog()
    clock.advance(61)
    second = service.get_catalog()

    assert first is second
    assert client.calls["models"] == 2
    assert client.calls["pricing"] == 2
    assert parser.models_parse_calls == 1
    assert parser.pricing_parse_calls == 1


def test_catalog_service_rebuilds_catalog_when_source_pages_change_after_ttl() -> None:
    clock = MutableClock()
    client = StubMWSClient(
        {
            "models": ["model-v1", "model-v2"],
            "pricing": ["pricing-v1", "pricing-v2"],
        }
    )
    parser = StubParser()
    service = CatalogService(
        models_url="models",
        pricing_url="pricing",
        client=client,
        parser=parser,
        cache_ttl_seconds=60,
        time_provider=clock,
    )

    first = service.get_catalog()
    clock.advance(61)
    second = service.get_catalog()

    assert first is not second
    assert second[0].model.name == "model-v2"
    assert client.calls["models"] == 2
    assert client.calls["pricing"] == 2
    assert parser.models_parse_calls == 2
    assert parser.pricing_parse_calls == 2


def test_catalog_service_uses_llm_recovery_when_deterministic_parser_fails() -> None:
    clock = MutableClock()
    client = StubMWSClient(
        {
            "models": ["model-v2"],
            "pricing": ["pricing-v2"],
        }
    )
    parser = FailingParser()
    recovery_client = StubRecoveryClient()
    service = CatalogService(
        models_url="models",
        pricing_url="pricing",
        client=client,
        parser=parser,
        recovery_client=recovery_client,
        cache_ttl_seconds=60,
        time_provider=clock,
    )

    catalog = service.get_catalog()

    assert len(catalog) == 1
    assert catalog[0].model.name == "recovered-model"
    assert catalog[0].pricing is not None
    assert catalog[0].pricing.input_price_per_1k_tokens_rub == 2.5
    assert recovery_client.model_recovery_calls == 1
    assert recovery_client.pricing_recovery_calls == 1

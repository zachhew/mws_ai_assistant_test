from app.core.config import get_settings
from app.services.catalog_service import CatalogService
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser


def main() -> None:
    settings = get_settings()

    service = CatalogService(
        models_url=settings.mws_models_url,
        pricing_url=settings.mws_pricing_url,
        client=MWSClient(),
        parser=MWSParser(),
    )

    catalog = service.get_catalog(force_refresh=True)

    for entry in catalog:
        print("=" * 80)
        print("model:", entry.model.name)
        print("context:", entry.model.context_window_tokens)
        print("input modalities:", entry.model.input_modalities)
        print("output modalities:", entry.model.output_modalities)
        if entry.pricing:
            print("input price:", entry.pricing.input_price_per_1k_tokens_rub)
            print("output price:", entry.pricing.output_price_per_1k_tokens_rub)
            print("billing unit:", entry.pricing.billing_unit_tokens)
        else:
            print("pricing: None")


if __name__ == "__main__":
    main()

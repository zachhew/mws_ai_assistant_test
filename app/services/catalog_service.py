from __future__ import annotations

from app.core.exceptions import CatalogBuildError
from app.core.logging import get_logger
from app.domain.catalog import CatalogEntry, ModelSpec, PricingSpec
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser


class CatalogService:
    def __init__(
        self,
        models_url: str,
        pricing_url: str,
        client: MWSClient,
        parser: MWSParser,
    ) -> None:
        self._models_url = models_url
        self._pricing_url = pricing_url
        self._client = client
        self._parser = parser
        self._logger = get_logger(self.__class__.__name__)

        self._catalog_cache: list[CatalogEntry] | None = None

    def get_catalog(self, force_refresh: bool = False) -> list[CatalogEntry]:
        if self._catalog_cache is not None and not force_refresh:
            self._logger.info("Returning catalog from in-memory cache.")
            return self._catalog_cache

        self._logger.info("Building catalog from MWS source pages.")

        models_html = self._client.fetch_text(self._models_url)
        pricing_html = self._client.fetch_text(self._pricing_url)

        models = self._parser.parse_models_page(models_html, self._models_url)
        pricing = self._parser.parse_pricing_page(pricing_html, self._pricing_url)

        catalog = self._build_catalog(models=models, prices=pricing)

        self._catalog_cache = catalog
        self._logger.info("Catalog built successfully. entries=%s", len(catalog))
        return catalog

    def _build_catalog(
        self,
        models: list[ModelSpec],
        prices: list[PricingSpec],
    ) -> list[CatalogEntry]:
        pricing_by_name = {item.model_name: item for item in prices}
        catalog: list[CatalogEntry] = []

        for model in models:
            catalog.append(
                CatalogEntry(
                    model=model,
                    pricing=pricing_by_name.get(model.name),
                )
            )

        if not catalog:
            raise CatalogBuildError("Catalog is empty after merge step.")

        return catalog
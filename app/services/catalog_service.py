from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
from typing import Callable

from app.core.exceptions import CatalogBuildError
from app.core.logging import get_logger
from app.domain.catalog import CatalogEntry, ModelSpec, PricingSpec
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser


@dataclass(slots=True)
class CatalogSnapshot:
    catalog: list[CatalogEntry]
    source_signature: str
    expires_at: float


class CatalogService:
    def __init__(
        self,
        models_url: str,
        pricing_url: str,
        client: MWSClient,
        parser: MWSParser,
        cache_ttl_seconds: int = 900,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self._models_url = models_url
        self._pricing_url = pricing_url
        self._client = client
        self._parser = parser
        self._cache_ttl_seconds = max(cache_ttl_seconds, 0)
        self._time_provider = time_provider or monotonic
        self._logger = get_logger(self.__class__.__name__)

        self._catalog_cache: CatalogSnapshot | None = None

    def get_catalog(self, force_refresh: bool = False) -> list[CatalogEntry]:
        now = self._time_provider()

        if (
            self._catalog_cache is not None
            and self._catalog_cache.expires_at > now
            and not force_refresh
        ):
            self._logger.info("Returning catalog from TTL cache.")
            return self._catalog_cache.catalog

        models_html = self._client.fetch_text(self._models_url)
        pricing_html = self._client.fetch_text(self._pricing_url)
        source_signature = self._build_source_signature(models_html, pricing_html)

        if (
            self._catalog_cache is not None
            and self._catalog_cache.source_signature == source_signature
            and not force_refresh
        ):
            self._catalog_cache.expires_at = now + self._cache_ttl_seconds
            self._logger.info("Returning catalog after successful source revalidation.")
            return self._catalog_cache.catalog

        self._logger.info("Building catalog from MWS source pages.")

        models = self._parser.parse_models_page(models_html, self._models_url)
        pricing = self._parser.parse_pricing_page(pricing_html, self._pricing_url)

        catalog = self._build_catalog(models=models, prices=pricing)

        self._catalog_cache = CatalogSnapshot(
            catalog=catalog,
            source_signature=source_signature,
            expires_at=now + self._cache_ttl_seconds,
        )
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

    def _build_source_signature(self, models_html: str, pricing_html: str) -> str:
        digest = sha256()
        digest.update(models_html.encode("utf-8"))
        digest.update(b"\n--pricing--\n")
        digest.update(pricing_html.encode("utf-8"))
        return digest.hexdigest()

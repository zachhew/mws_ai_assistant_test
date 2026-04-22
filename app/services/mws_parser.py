from __future__ import annotations

import re

from bs4 import BeautifulSoup

from app.core.exceptions import MWSParseError
from app.domain.catalog import ModelSpec, PricingSpec


class MWSParser:
    _MODEL_NAMES: tuple[str, ...] = (
        "deepseek-r1-distill-qwen-32b",
        "gemma-3-27b-it",
        "llama-3.3-70b-instruct",
        "qwen3-32b",
        "qwen3-235b-instruct",
        "qwen3-coder-480b-a35b",
        "glm-4.6-357b",
        "kimi-k2-instruct",
        "bge-multilingual-gemma2",
        "bge-m3",
    )

    def parse_models_page(self, html: str, source_url: str) -> list[ModelSpec]:
        text = self._normalize_text(html)
        models: list[ModelSpec] = []

        for name in self._MODEL_NAMES:
            row = self._extract_model_row(text, name)
            if row is None:
                continue

            input_modalities, output_modalities = self._extract_modalities(row)
            context_window_tokens, model_size_label = self._extract_model_numbers(row)
            is_embedding = "embedding" in [item.lower() for item in output_modalities]

            models.append(
                ModelSpec(
                    name=name,
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    context_window_tokens=context_window_tokens,
                    model_size_label=model_size_label,
                    family=self._infer_family(name),
                    supports_text_input="text" in input_modalities,
                    supports_image_input="image" in input_modalities,
                    supports_text_output="text" in output_modalities,
                    is_embedding_model=is_embedding,
                    source_url=source_url,
                )
            )

        if not models:
            raise MWSParseError("Failed to parse any models from MWS models page.")

        return models

    def parse_pricing_page(self, html: str, source_url: str) -> list[PricingSpec]:
        text = self._normalize_text(html)
        prices: list[PricingSpec] = []

        for name in self._MODEL_NAMES:
            row = self._extract_pricing_row(text, name)
            if row is None:
                continue

            price_values = re.findall(r"(\d+(?:,\d+)?)\s*₽", row)
            billing_match = re.search(r"(\d+)\s*$", row)

            if billing_match is None:
                continue

            billing_unit_tokens = int(billing_match.group(1))

            if name in {"bge-multilingual-gemma2", "bge-m3"}:
                if len(price_values) < 2:
                    continue
                input_price = self._parse_decimal(price_values[-1])
                output_price = None
            else:
                if len(price_values) < 4:
                    continue
                input_price = self._parse_decimal(price_values[-2])
                output_price = self._parse_decimal(price_values[-1])

            prices.append(
                PricingSpec(
                    model_name=name,
                    input_price_per_1k_tokens_rub=input_price,
                    output_price_per_1k_tokens_rub=output_price,
                    billing_unit_tokens=billing_unit_tokens,
                    source_url=source_url,
                )
            )

        if not prices:
            raise MWSParseError("Failed to parse any pricing rows from MWS pricing page.")

        return prices

    def _extract_model_row(self, text: str, name: str) -> str | None:
        pattern = rf"{re.escape(name)}(.*?)(?={'|'.join(re.escape(item) for item in self._MODEL_NAMES if item != name)}|В таблице:|Последнее обновление:|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return f"{name} {match.group(1)}".strip()

    def _extract_pricing_row(self, text: str, name: str) -> str | None:
        pattern = rf"{re.escape(name)}(.*?)(?={'|'.join(re.escape(item) for item in self._MODEL_NAMES if item != name)}|Последнее обновление:|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return f"{name} {match.group(1)}".strip()

    def _extract_modalities(self, row: str) -> tuple[list[str], list[str]]:
        lowered = row.lower()

        input_modalities: list[str] = []
        if "text" in lowered:
            input_modalities.append("text")
        if "image" in lowered:
            input_modalities.append("image")

        output_modalities: list[str]
        if "embedding" in lowered:
            output_modalities = ["embedding"]
        else:
            output_modalities = ["text"]

        return input_modalities, output_modalities

    def _extract_model_numbers(self, row: str) -> tuple[int | None, str | None]:
        number_matches = re.findall(r"(\d+(?:\.\d+)?)", row)
        if len(number_matches) < 2:
            return None, None

        context_k = number_matches[-2]
        size_label = number_matches[-1]

        context_window_tokens = int(float(context_k) * 1000)
        return context_window_tokens, size_label

    def _normalize_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)

    def _parse_decimal(self, value: str) -> float:
        return float(value.replace(",", ".").strip())

    def _infer_family(self, name: str) -> str:
        lowered = name.lower()
        families = (
            ("deepseek", "deepseek"),
            ("gemma", "gemma"),
            ("llama", "llama"),
            ("qwen", "qwen"),
            ("glm", "glm"),
            ("kimi", "kimi"),
            ("bge", "bge"),
        )
        for needle, family in families:
            if needle in lowered:
                return family
        return "unknown"
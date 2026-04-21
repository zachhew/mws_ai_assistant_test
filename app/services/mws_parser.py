from __future__ import annotations

import re
from typing import Iterable

from bs4 import BeautifulSoup

from app.core.exceptions import MWSParseError
from app.domain.catalog import ModelSpec, PricingSpec


class MWSParser:
    def parse_models_page(self, html: str, source_url: str) -> list[ModelSpec]:
        text = self._normalize_text(html)

        patterns = [
            (
                r"(deepseek-r1-distill-qwen-32b)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(gemma-3-27b-it)\s+.*?`Text`,\s*`Image\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text", "image"],
                ["text"],
                False,
            ),
            (
                r"(llama-3\.3-70b-instruct)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(qwen3-32b)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(qwen3-235b-instruct)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(qwen3-coder-480b-a35b)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(glm-4\.6-357b)\s+.*?`Text\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["text"],
                False,
            ),
            (
                r"(kimi-k2-instruct)\s+.*?`Text`,\s*`Image\s+``Text\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text", "image"],
                ["text"],
                False,
            ),
            (
                r"(bge-multilingual-gemma2)\s+.*?`Text\s+``Embedding\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["embedding"],
                True,
            ),
            (
                r"(bge-m3)\s+.*?`Text\s+``Embedding\s+`(\d+)\s+(\d+(?:\.\d+)?)",
                ["text"],
                ["embedding"],
                True,
            ),
        ]

        models: list[ModelSpec] = []
        for pattern, input_modalities, output_modalities, is_embedding in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue

            name = match.group(1)
            context_k = int(match.group(2))
            size_label = match.group(3)

            models.append(
                ModelSpec(
                    name=name,
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    context_window_tokens=context_k * 1000,
                    model_size_label=size_label,
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

        patterns = [
            r"(deepseek-r1-distill-qwen-32b)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(gemma-3-27b-it)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(llama-3\.3-70b-instruct)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(qwen3-32b)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(qwen3-235b-instruct)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(qwen3-coder-480b-a35b)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(glm-4\.6-357b)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(kimi-k2-instruct)\s+([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*([\d,]+)\s*₽\s*(\d+)",
            r"(bge-multilingual-gemma2)\s+([\d,]+)\s*₽\s*[–-]\s*([\d,]+)\s*₽\s*[–-]\s*(\d+)",
            r"(bge-m3)\s+([\d,]+)\s*₽\s*[–-]\s*([\d,]+)\s*₽\s*[–-]\s*(\d+)",
        ]

        prices: list[PricingSpec] = []

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue

            name = match.group(1)

            if name in {"bge-multilingual-gemma2", "bge-m3"}:
                input_price = self._parse_decimal(match.group(4))
                output_price = None
                billing_unit_tokens = int(match.group(5))
            else:
                input_price = self._parse_decimal(match.group(4))
                output_price = self._parse_decimal(match.group(5))
                billing_unit_tokens = int(match.group(6))

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

    def _normalize_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)

    def _parse_decimal(self, value: str) -> float:
        return float(value.replace(",", ".").strip())

    def _infer_family(self, name: str) -> str:
        lowered = name.lower()
        families: Iterable[tuple[str, str]] = (
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
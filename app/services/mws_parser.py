from __future__ import annotations

import re

from bs4 import BeautifulSoup

from app.core.exceptions import MWSParseError
from app.domain.catalog import ModelSpec, PricingSpec


class MWSParser:
    _MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.\-]{2,}$", flags=re.IGNORECASE)

    def parse_models_page(self, html: str, source_url: str) -> list[ModelSpec]:
        rows = self.extract_table_rows(html)
        models: list[ModelSpec] = []

        for cells in rows:
            name = self._extract_model_name(cells)
            if name is None:
                continue

            row_text = " ".join(cells)
            input_modalities, output_modalities = self._extract_modalities(row_text)
            context_window_tokens, model_size_label = self._extract_model_numbers(cells)
            is_embedding = any(item.lower() == "embedding" for item in output_modalities)

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
        rows = self.extract_table_rows(html)
        prices: list[PricingSpec] = []

        for cells in rows:
            name = self._extract_model_name(cells)
            if name is None:
                continue

            price_values = self._extract_price_values(cells)
            billing_unit_tokens = self._extract_billing_unit_tokens(cells)
            if billing_unit_tokens is None or not price_values:
                continue

            input_price, output_price = self._extract_io_prices(price_values)
            if input_price is None:
                continue

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

    def extract_table_rows(self, html: str) -> list[list[str]]:
        soup = BeautifulSoup(html, "lxml")
        rows: list[list[str]] = []

        for tr in soup.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
            normalized_cells: list[str] = []
            for cell in cells:
                normalized = self._normalize_cell(cell)
                if normalized:
                    normalized_cells.append(normalized)
            if normalized_cells:
                rows.append(normalized_cells)

        return rows

    def _extract_model_name(self, cells: list[str]) -> str | None:
        if not cells:
            return None

        candidate = cells[0].strip().lower()
        if not self._MODEL_NAME_PATTERN.match(candidate):
            return None

        if candidate in {"model", "модель", "name", "название"}:
            return None

        return candidate

    def _extract_modalities(self, row_text: str) -> tuple[list[str], list[str]]:
        lowered = row_text.lower()

        input_modalities: list[str] = []
        if "text" in lowered:
            input_modalities.append("text")
        if "image" in lowered:
            input_modalities.append("image")

        if "embedding" in lowered:
            output_modalities = ["embedding"]
        else:
            output_modalities = ["text"]

        return input_modalities, output_modalities

    def _extract_model_numbers(self, cells: list[str]) -> tuple[int | None, str | None]:
        numeric_values: list[str] = []

        for cell in cells[1:]:
            if "₽" in cell:
                continue
            numeric_values.extend(re.findall(r"\d+(?:\.\d+)?", cell))

        if len(numeric_values) < 2:
            return None, None

        context_window_tokens = int(float(numeric_values[-2]) * 1000)
        model_size_label = numeric_values[-1]
        return context_window_tokens, model_size_label

    def _extract_price_values(self, cells: list[str]) -> list[float]:
        prices: list[float] = []

        for cell in cells[1:]:
            matches = re.findall(r"(\d+(?:,\d+)?)\s*₽", cell)
            prices.extend(self._parse_decimal(value) for value in matches)

        return prices

    def _extract_billing_unit_tokens(self, cells: list[str]) -> int | None:
        for cell in reversed(cells[1:]):
            match = re.fullmatch(r"\d+", cell.replace(" ", ""))
            if match is not None:
                return int(match.group(0))
        return None

    def _extract_io_prices(
        self,
        price_values: list[float],
    ) -> tuple[float | None, float | None]:
        if len(price_values) >= 4:
            return price_values[-2], price_values[-1]
        if len(price_values) >= 2:
            return price_values[-1], None
        if len(price_values) == 1:
            return price_values[0], None
        return None, None

    def _normalize_cell(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

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

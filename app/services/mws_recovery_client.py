from __future__ import annotations

from app.core.logging import get_logger
from app.domain.catalog import ModelSpec, PricingSpec
from app.services.openrouter_client import OpenRouterClient


class MWSRecoveryClient:
    def __init__(self, llm_client: OpenRouterClient | None = None) -> None:
        self._llm_client = llm_client or OpenRouterClient()
        self._logger = get_logger(self.__class__.__name__)

    def recover_models(
        self,
        rows: list[list[str]],
        source_url: str,
    ) -> list[ModelSpec]:
        prompt = {
            "task": "recover_mws_models",
            "source_url": source_url,
            "rows": rows,
            "schema": {
                "models": [
                    {
                        "name": "string",
                        "input_modalities": ["text", "image"],
                        "output_modalities": ["text", "embedding"],
                        "context_window_tokens": "integer|null",
                        "model_size_label": "string|null",
                        "family": "string|null",
                        "supports_text_input": "boolean",
                        "supports_image_input": "boolean",
                        "supports_text_output": "boolean",
                        "is_embedding_model": "boolean",
                    }
                ]
            },
        }
        data = self._llm_client.extract_structured_json(
            system_prompt=(
                "Ты восстанавливаешь структуру каталога MWS из уже извлеченных строк таблицы. "
                "Верни только JSON. Не выдумывай строки, которых нет. "
                "Поле family можно выводить из названия модели, "
                "если это очевидно, иначе ставь 'unknown'."
            ),
            user_payload=prompt,
        )
        models = data.get("models", [])
        result = [
            ModelSpec(
                source_url=source_url,
                **item,
            )
            for item in models
        ]
        self._logger.info("Recovered %s models via LLM fallback.", len(result))
        return result

    def recover_pricing(
        self,
        rows: list[list[str]],
        source_url: str,
    ) -> list[PricingSpec]:
        prompt = {
            "task": "recover_mws_pricing",
            "source_url": source_url,
            "rows": rows,
            "schema": {
                "prices": [
                    {
                        "model_name": "string",
                        "input_price_per_1k_tokens_rub": "number|null",
                        "output_price_per_1k_tokens_rub": "number|null",
                        "billing_unit_tokens": "integer|null",
                    }
                ]
            },
        }
        data = self._llm_client.extract_structured_json(
            system_prompt=(
                "Ты восстанавливаешь таблицу pricing MWS из уже извлеченных строк таблицы. "
                "Верни только JSON. Не выдумывай цены, которых нет в строках."
            ),
            user_payload=prompt,
        )
        prices = data.get("prices", [])
        result = [
            PricingSpec(
                source_url=source_url,
                **item,
            )
            for item in prices
        ]
        self._logger.info("Recovered %s pricing rows via LLM fallback.", len(result))
        return result

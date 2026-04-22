from __future__ import annotations

import json

from app.core.logging import get_logger
from app.domain.recommendation import RecommendationReport
from app.services.openrouter_client import OpenRouterClient


class ExplanationService:
    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._client: OpenRouterClient | None = None

        try:
            self._client = OpenRouterClient()
        except Exception:
            self._client = None

    def generate_summary(self, report: RecommendationReport) -> str:
        if self._client is None:
            return report.final_summary

        try:
            prompt = self._build_prompt(report)
            summary = self._client.generate_explanation(prompt)
            if summary:
                self._logger.info("LLM explanation generated successfully.")
                return summary
        except Exception as exc:
            self._logger.warning(
                "LLM explanation generation failed, fallback to deterministic summary: %s",
                exc,
            )

        return report.final_summary

    def _build_prompt(self, report: RecommendationReport) -> str:
        payload = {
            "input_data": report.input_data.model_dump(),
            "recommended_models": [item.model_dump() for item in report.recommended_models],
            "calculations": [item.model_dump() for item in report.calculations],
            "limitations": report.limitations,
            "assumptions": report.assumptions,
            "final_summary": report.final_summary,
        }

        return (
            "На основе структурированных данных ниже сформируй краткое и понятное итоговое объяснение для пользователя.\n"
            "Требования:\n"
            "1. Ответ только на русском языке.\n"
            "2. Не выдумывай новых чисел или фактов.\n"
            "3. Если ни одна модель не укладывается в бюджет, скажи об этом прямо.\n"
            "4. Укажи лучшую модель или ближайший технический вариант.\n"
            "5. Ответ должен быть 3-6 предложений, без markdown-таблиц.\n\n"
            f"Данные:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
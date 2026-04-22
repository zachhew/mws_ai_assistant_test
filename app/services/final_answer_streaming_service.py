from __future__ import annotations

import json
from collections.abc import Iterator

from app.domain.recommendation import RecommendationReport
from app.services.openrouter_client import OpenRouterClient


class FinalAnswerStreamingService:
    def __init__(self) -> None:
        self._client = OpenRouterClient()

    def build_prompt(self, report: RecommendationReport) -> str:
        payload = {
            "input_data": report.input_data.model_dump(),
            "recommended_models": [item.model_dump() for item in report.recommended_models],
            "calculations": [item.model_dump() for item in report.calculations],
            "limitations": report.limitations,
            "assumptions": report.assumptions,
            "final_summary": report.final_summary,
        }

        return (
            "Сформируй итоговый ответ пользователю на русском языке на основе структурированных данных.\n"
            "Структура ответа:\n"
            "1. Входные данные\n"
            "2. Подходящие или ближайшие технические варианты\n"
            "3. Расчеты\n"
            "4. Пояснения/ограничения\n"
            "5. Итог\n"
            "Не выдумывай новые числа, не используй LaTeX, не используй искусственные метки.\n\n"
            f"Данные:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def stream_answer(self, report: RecommendationReport) -> Iterator[str]:
        prompt = self.build_prompt(report)
        yield from self._client.stream_explanation(prompt)
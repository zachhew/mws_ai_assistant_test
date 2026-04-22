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

    def generate_final_answer(self, report: RecommendationReport) -> str:
        if self._client is None:
            return self._build_fallback_answer(report)

        try:
            prompt = self._build_prompt(report)
            answer = self._client.generate_explanation(prompt)
            if answer:
                self._logger.info("LLM final answer generated successfully.")
                return answer
        except Exception as exc:
            self._logger.warning(
                "LLM final answer generation failed, fallback to deterministic answer: %s",
                exc,
            )

        return self._build_fallback_answer(report)

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
            "Сформируй итоговый ответ пользователю на русском языке на основе структурированных данных.\n"
            "Требования:\n"
            "1. Ответ только на русском.\n"
            "2. Не выдумывай факты, числа или модели.\n"
            "3. Сохрани структуру ответа с разделами:\n"
            "   - Входные данные\n"
            "   - Рекомендованные модели\n"
            "   - Расчеты\n"
            "   - Пояснения/ограничения\n"
            "   - Итог\n"
            "4. Пиши естественно, не шаблонно, но строго по данным.\n"
            "5. Если ни одна модель не проходит по бюджету, скажи это явно.\n"
            "6. Названия моделей не переводи.\n\n"
            f"Данные:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _build_fallback_answer(self, report: RecommendationReport) -> str:
        lines: list[str] = []

        lines.append("## Входные данные")
        lines.append(f"- Тип задачи: {report.input_data.task_type}")
        lines.append(f"- Формат ввода: {report.input_data.input_modality}")

        if report.input_data.expected_input_tokens is not None:
            lines.append(
                f"- Оценка входных токенов на запрос: {report.input_data.expected_input_tokens}"
            )
        if report.input_data.expected_output_tokens is not None:
            lines.append(
                f"- Оценка выходных токенов на запрос: {report.input_data.expected_output_tokens}"
            )
        if report.input_data.requests_per_month is not None:
            lines.append(f"- Запросов в месяц: {report.input_data.requests_per_month}")
        if report.input_data.budget_limit_rub is not None:
            lines.append(f"- Бюджет: {report.input_data.budget_limit_rub:.2f} RUB")

        lines.append("")
        lines.append("## Рекомендованные модели")
        if not report.recommended_models:
            lines.append("- Подходящие модели не найдены по текущим вводным.")
        else:
            for candidate in report.recommended_models:
                label = f" ({candidate.fit_label})" if candidate.fit_label else ""
                lines.append(f"- {candidate.model_name}{label}: {candidate.fit_summary}")

        lines.append("")
        lines.append("## Расчеты")
        if not report.calculations:
            lines.append("- Расчеты недоступны.")
        else:
            for calc in report.calculations:
                lines.append(
                    f"- {calc.model_name}: "
                    f"input={calc.monthly_input_cost_rub:.2f} RUB, "
                    f"output={calc.monthly_output_cost_rub:.2f} RUB, "
                    f"total={calc.total_monthly_cost_rub:.2f} RUB"
                )

        lines.append("")
        lines.append("## Пояснения/ограничения")
        if report.limitations:
            for limitation in report.limitations:
                lines.append(f"- {limitation}")
        else:
            lines.append("- Существенных ограничений не выявлено.")

        if report.assumptions:
            lines.append("")
            lines.append("## Допущения")
            for assumption in report.assumptions:
                lines.append(f"- {assumption}")

        lines.append("")
        lines.append("## Итог")
        lines.append(report.final_summary)

        return "\n".join(lines)
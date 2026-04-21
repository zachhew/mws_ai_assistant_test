import time
import uuid

from app.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from app.domain.recommendation import RecommendationReport


def extract_session_id(request: ChatCompletionRequest) -> str:
    metadata = request.metadata or {}
    raw_session_id = metadata.get("session_id")

    if isinstance(raw_session_id, str) and raw_session_id.strip():
        return raw_session_id.strip()

    return str(uuid.uuid4())


def render_report_as_text(report: RecommendationReport) -> str:
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


def build_chat_completion_response(
    request: ChatCompletionRequest,
    report: RecommendationReport,
    response_model: str,
) -> ChatCompletionResponse:
    content = render_report_as_text(report)

    message = ChatMessage(role="assistant", content=content)
    choice = ChatCompletionChoice(index=0, message=message)

    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(content.split())

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=response_model,
        choices=[choice],
        usage=usage,
    )
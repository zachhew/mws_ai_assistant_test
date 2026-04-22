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


def build_chat_completion_response(
    request: ChatCompletionRequest,
    report: RecommendationReport,
    response_model: str,
) -> ChatCompletionResponse:
    content = report.final_answer_text or report.final_summary

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
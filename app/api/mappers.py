import json
import time
import uuid
from collections.abc import Iterator

from app.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
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


def build_native_streaming_chunks(
    text_stream: Iterator[str],
    response_model: str,
) -> Iterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=response_model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    for piece in text_stream:
        chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=response_model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionDelta(content=piece),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    final_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=response_model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

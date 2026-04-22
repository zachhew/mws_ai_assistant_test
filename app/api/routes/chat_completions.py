from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.mappers import (
    build_chat_completion_response,
    build_native_streaming_chunks,
    extract_session_id,
)
from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.agent.coordinator import ChatCompletionsCoordinator
from app.core.config import Settings, get_settings
from app.dependencies import (
    get_chat_completions_coordinator,
    get_final_answer_streaming_service,
)
from app.services.final_answer_streaming_service import FinalAnswerStreamingService

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    tags=["chat"],
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    coordinator: ChatCompletionsCoordinator = Depends(get_chat_completions_coordinator),
    settings: Settings = Depends(get_settings),
    final_answer_streaming_service: FinalAnswerStreamingService = Depends(
        get_final_answer_streaming_service
    ),
):
    session_id = extract_session_id(request)
    report = await coordinator.handle(request=request, session_id=session_id)

    response_model = request.model or settings.adk_litellm_model

    if request.stream:
        text_stream = final_answer_streaming_service.stream_answer(report)
        return StreamingResponse(
            build_native_streaming_chunks(
                text_stream=text_stream,
                response_model=response_model,
            ),
            media_type="text/event-stream",
        )

    return build_chat_completion_response(
        request=request,
        report=report,
        response_model=response_model,
    )

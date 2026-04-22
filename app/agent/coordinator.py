from __future__ import annotations

from app.api.schemas import ChatCompletionRequest
from app.agent.adk_runtime import AdkRuntimeService
from app.core.logging import get_logger
from app.domain.recommendation import RecommendationReport


class ChatCompletionsCoordinator:
    def __init__(self, adk_runtime: AdkRuntimeService) -> None:
        self._adk_runtime = adk_runtime
        self._logger = get_logger(self.__class__.__name__)

    async def handle(
        self,
        request: ChatCompletionRequest,
        session_id: str,
    ) -> RecommendationReport:
        self._logger.info("Handling request via ADK runtime. session_id=%s", session_id)
        return await self._adk_runtime.run(request=request, session_id=session_id)

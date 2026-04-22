from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.api.schemas import ChatCompletionRequest, ChatMessage
from app.agent.session_manager import SessionManager
from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.recommendation import RecommendationReport
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


class AdkRuntimeService:
    def __init__(
        self,
        settings: Settings,
        session_manager: SessionManager,
        profile_service: ProfileService,
        catalog_service: CatalogService,
        estimator: CostEstimator,
        recommender: ModelRecommender,
        report_builder: ReportBuilder,
    ) -> None:
        self._settings = settings
        self._session_manager = session_manager
        self._profile_service = profile_service
        self._catalog_service = catalog_service
        self._estimator = estimator
        self._recommender = recommender
        self._report_builder = report_builder
        self._logger = get_logger(self.__class__.__name__)

    async def run(
        self,
        request: ChatCompletionRequest,
        session_id: str,
    ) -> RecommendationReport:
        self._prepare_litellm_env()

        session_service = InMemorySessionService()
        app_name = self._settings.app_name
        user_id = "api-user"

        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )

        run_selection_tool = self._make_run_selection_tool(session_id=session_id)
        get_last_report_tool = self._make_get_last_report_tool(session_id=session_id)

        agent = LlmAgent(
            name="mws_model_selection_agent",
            model=LiteLlm(model=self._settings.adk_litellm_model),
            description="Подбирает модели MWS GPT Model Hub на основе пользовательского сценария.",
            instruction=(
                "Ты — AI-ассистент по подбору моделей MWS GPT Model Hub.\n"
                "Всегда отвечай только на русском языке.\n"
                "Для нового сценария сначала вызови tool run_model_selection.\n"
                "Для follow-up вопроса можешь использовать get_last_report, если нужно опереться на предыдущий расчет.\n"
                "Не выдумывай цены, характеристики моделей, ограничения или расчеты.\n"
                "Считай результаты tools источником истины.\n"
                "Не противоречь данным из tools.\n"
                "\n"
                "Формат ответа:\n"
                "1. Входные данные\n"
                "2. Подходящие или ближайшие технические варианты\n"
                "3. Расчеты\n"
                "4. Пояснения/ограничения\n"
                "5. Итог\n"
                "\n"
                "Требования к формулировкам:\n"
                "- Пиши ясно, по делу и без лишней воды.\n"
                "- Не используй формулировки 'модель 1', 'модель 2', 'оценка 95.0' и подобные искусственные метки.\n"
                "- Называй модели по их реальным именам.\n"
                "- Если модель укладывается в бюджет, так и пиши: 'укладывается в бюджет'.\n"
                "- Если ни одна модель не укладывается в бюджет, не пиши, что моделей нет вообще.\n"
                "- В таком случае явно укажи, что в заданный бюджет подходящих моделей нет, но покажи ближайшие технически подходящие варианты.\n"
                "- Если подходящие модели уже укладываются в бюджет, не советуй увеличивать бюджет без необходимости.\n"
                "- Практическая рекомендация должна соответствовать расчету, а не быть универсальной заготовкой.\n"
                "- Для embeddings-сценариев не описывай выходные токены как значимую часть расчета.\n"
                "- Для multimodal-сценариев подчеркивай поддержку изображений только если это действительно важно для выбора.\n"
                "- Не используй LaTeX, формулы в \\( \\), markdown-математику или специальные математические обозначения.\n"
                "- Если нужно показать расчет, пиши его обычным текстом, например: 200000 × 350 = 70000000 токенов.\n"
                "\n"
                "Правила итоговой рекомендации:\n"
                "- Если есть явный лучший вариант, назови его прямо.\n"
                "- Если есть более дешевый и более функциональный варианты, кратко объясни компромисс.\n"
                "- Если все подходящие варианты укладываются в бюджет, рекомендация должна помогать выбрать между ними, а не предлагать повысить бюджет.\n"
            ),
            tools=[run_selection_tool, get_last_report_tool],
        )

        runner = Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )

        last_user_message = self._extract_last_user_message(request.messages)
        content = types.Content(
            role="user",
            parts=[types.Part(text=last_user_message)],
        )

        final_text = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text
                break

        state = self._session_manager.get(session_id)
        if state is None or state.last_report is None:
            raise RuntimeError("ADK flow finished without producing a structured report.")

        report = state.last_report
        report.final_answer_text = final_text or report.final_answer_text or report.final_summary
        return report

    def _prepare_litellm_env(self) -> None:
        os.environ["OR_API_KEY"] = self._settings.or_api_key
        os.environ["OR_SITE_URL"] = self._settings.or_site_url
        os.environ["OR_APP_NAME"] = self._settings.or_app_name

    def _make_run_selection_tool(self, session_id: str):
        def run_model_selection(user_request: str) -> dict[str, Any]:
            """
            Выполняет полный детерминированный пайплайн подбора модели:
            строит профиль сценария, загружает каталог MWS, считает стоимость,
            подбирает модели и формирует структурированный отчет.
            """
            state = self._session_manager.get_or_create(session_id)

            profile = self._profile_service.build_profile(
                messages=[ChatMessage(role="user", content=user_request)],
                previous_profile=state.last_user_case,
            )

            full_catalog = self._catalog_service.get_catalog(force_refresh=False)
            filtered_catalog = self._recommender.filter_catalog(profile, full_catalog)
            costs = self._estimator.estimate_for_catalog(profile, filtered_catalog)
            recommendations = self._recommender.recommend(profile, filtered_catalog, costs)
            report = self._report_builder.build(profile, recommendations, costs)

            self._session_manager.save_artifacts(
                session_id=session_id,
                profile=profile,
                catalog=filtered_catalog,
                costs=costs,
                recommendations=recommendations,
                report=report,
            )

            return report.model_dump()

        return run_model_selection

    def _make_get_last_report_tool(self, session_id: str):
        def get_last_report() -> dict[str, Any] | None:
            """
            Возвращает последний структурированный отчет по текущей сессии, если он уже существует.
            """
            state = self._session_manager.get(session_id)
            if state is None or state.last_report is None:
                return None
            return state.last_report.model_dump()

        return get_last_report

    def _extract_last_user_message(self, messages: list[ChatMessage]) -> str:
        user_messages = [message.content for message in messages if message.role == "user"]
        if not user_messages:
            return "Помоги подобрать модель."
        return user_messages[-1]

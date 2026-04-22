from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from app.agent.prompts import SYSTEM_INSTRUCTIONS
from app.agent.tools import AssistantTools
from app.agent.session_manager import SessionManager
from app.core.config import get_settings
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.mws_client import MWSClient
from app.services.mws_parser import MWSParser
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


def build_tools() -> AssistantTools:
    settings = get_settings()

    session_manager = SessionManager(ttl_minutes=settings.default_session_ttl_minutes)
    profile_service = ProfileService()
    catalog_service = CatalogService(
        models_url=settings.mws_models_url,
        pricing_url=settings.mws_pricing_url,
        client=MWSClient(),
        parser=MWSParser(),
    )
    estimator = CostEstimator()
    recommender = ModelRecommender()
    report_builder = ReportBuilder()

    return AssistantTools(
        session_manager=session_manager,
        profile_service=profile_service,
        catalog_service=catalog_service,
        estimator=estimator,
        recommender=recommender,
        report_builder=report_builder,
    )


async def main() -> None:
    load_dotenv()
    settings = get_settings()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is not set in environment.")

    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner

    tools = build_tools()

    agent = LlmAgent(
        name="mws_model_selection_agent",
        model=settings.adk_agent_model,
        description="Подбирает модели MWS GPT Model Hub с использованием tools.",
        instruction=SYSTEM_INSTRUCTIONS,
        tools=[
            tools.build_usage_profile,
            tools.load_mws_catalog,
            tools.estimate_costs,
            tools.recommend_models,
            tools.build_report,
            tools.get_last_report,
        ],
    )

    user_prompt = """
Сессия: demo-session-adk-1

Помоги подобрать модель MWS для customer support чата.
Около 50000 запросов в месяц.
Вход: 700 токенов на запрос.
Выход: 250 токенов на запрос.
Бюджет: 12000 рублей в месяц.
Нужен хороший баланс цены и качества.

Обязательно:
1. Используй tools.
2. Построй профиль использования.
3. Загрузи каталог MWS.
4. Посчитай стоимость.
5. Выбери модели.
6. Сформируй итоговый ответ на русском языке.
7. Если ни одна модель не укладывается в бюджет, скажи об этом прямо.
"""

    runner = InMemoryRunner(
        app_name="mws_model_selection_assistant_debug",
        agent=agent,
    )
    events = await runner.run_debug(user_prompt, quiet=True)

    final_text: str | None = None
    for event in reversed(events):
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            text_parts = [part.text for part in event.content.parts if getattr(part, "text", None)]
            if text_parts:
                final_text = "".join(text_parts)
                break

    print("\n" + "=" * 80)
    print("FINAL AGENT RESPONSE")
    print("=" * 80)
    print(final_text or "[No final text response found]")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
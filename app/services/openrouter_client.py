from __future__ import annotations

import json

from openai import OpenAI
from collections.abc import Iterator

from app.core.config import get_settings
from app.core.logging import get_logger


class OpenRouterClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._model = settings.openrouter_model
        self._http_referer = settings.openrouter_http_referer
        self._app_title = settings.openrouter_app_title
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    def extract_profile(self, user_text: str) -> dict:
        system_prompt = """
        Ты извлекаешь структурированный профиль пользовательского сценария для подбора LLM-модели.

        Верни только JSON-объект без markdown, пояснений и лишнего текста.

        Поля JSON:
        - task_type: one of ["chat","reasoning","coding","multimodal","embeddings","unknown"]
        - input_modality: one of ["text","text_image","unknown"]
        - expected_input_tokens: integer or null
        - expected_output_tokens: integer or null
        - requests_per_day: integer or null
        - requests_per_month: integer or null
        - quality_priority: one of ["low","balanced","high"]
        - latency_priority: one of ["low","balanced","high"]
        - budget_limit_rub: number or null
        - needs_long_context: boolean
        - context_min_tokens: integer or null
        - notes: string or null
        - assumptions: array of strings

        Правила:
        1. Не выдумывай числа, если их нет в запросе.
        2. Если пользователь пишет "около", "примерно", "в среднем" — все равно извлекай число.
        3. Если явно сказано, что нужны изображения, ставь input_modality="text_image" и task_type="multimodal".
        4. Если явно сказано, что нужны эмбеддинги / embeddings / векторный поиск, ставь task_type="embeddings" и input_modality="text".
        5. Если пользователь указал число токенов для входа, заполняй expected_input_tokens.
        6. Если пользователь указал число токенов для ответа/выхода, заполняй expected_output_tokens.
        7. Для embeddings-сценариев expected_output_tokens должен быть 0, если явно не указано иное.
        8. Если поле отсутствует, ставь null.
        9. Ответ должен быть валидным JSON.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
            response_format={"type": "json_object"},
            extra_headers={
                "HTTP-Referer": self._http_referer,
                "X-Title": self._app_title,
            },
        )

        content = response.choices[0].message.content or "{}"
        self._logger.info("OpenRouter profile extraction completed.")
        return json.loads(content)

    def generate_explanation(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — технический AI-ассистент. "
                        "Отвечай только на русском языке. "
                        "Пиши ясно, кратко и по делу. "
                        "Не выдумывай факты и не противоречь входным данным. "
                        "Не используй искусственные метки вроде 'модель 1', 'модель 2', 'оценка 95.0'. "
                        "Названия моделей не переводи. "
                        "Не используй LaTeX, markdown-математику или математические формулы в специальной нотации. "
                        "Все расчеты записывай обычным текстом."
                        "Если подходящие модели укладываются в бюджет, не советуй увеличивать бюджет без необходимости. "
                        "Если ни одна модель не укладывается в бюджет, скажи это прямо и предложи ближайшие технические варианты. "
                        "Финальная рекомендация должна быть конкретной и соответствовать расчетам."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": self._http_referer,
                "X-Title": self._app_title,
            },
        )

        content = response.choices[0].message.content or ""
        self._logger.info("OpenRouter explanation generation completed.")
        return content.strip()

    def stream_explanation(self, prompt: str) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — технический AI-ассистент. "
                        "Отвечай только на русском языке. "
                        "Пиши ясно, кратко и по делу. "
                        "Не выдумывай факты и не противоречь входным данным. "
                        "Не используй LaTeX, markdown-математику или специальные математические обозначения. "
                        "Не используй искусственные метки вроде 'модель 1', 'модель 2', 'оценка 95.0'. "
                        "Названия моделей не переводи. "
                        "Если подходящие модели укладываются в бюджет, не советуй увеличивать бюджет без необходимости. "
                        "Если ни одна модель не укладывается в бюджет, скажи это прямо и предложи ближайшие технические варианты. "
                        "Финальная рекомендация должна быть конкретной и соответствовать расчетам."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            stream=True,
            extra_headers={
                "HTTP-Referer": self._http_referer,
                "X-Title": self._app_title,
            },
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content

from __future__ import annotations

import json

from openai import OpenAI

from app.core.config import get_settings
from app.core.logging import get_logger


class OpenRouterClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._logger = get_logger(self.__class__.__name__)
        self._model = settings.openrouter_model
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
        )

    def extract_profile(self, user_text: str) -> dict:
        system_prompt = """
Ты — помощник по нормализации пользовательского кейса для подбора LLM-модели.

Верни только JSON-объект без markdown и без пояснений.
Поля:
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
- Не выдумывай значения, если их нет.
- Если число явно не указано, ставь null.
- Если из текста ясно, что нужны изображения, ставь input_modality="text_image".
- Ответ должен быть валидным JSON.
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
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "mws-ai-assistant-test",
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
                        "Пиши ясно и по делу. "
                        "Не выдумывай факты и не противоречь входным данным."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "mws-ai-assistant-test",
            },
        )

        content = response.choices[0].message.content or ""
        self._logger.info("OpenRouter explanation generation completed.")
        return content.strip()
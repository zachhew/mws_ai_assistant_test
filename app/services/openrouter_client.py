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
        5. Если пользователь указал одно число токенов и оно относится к входу, заполняй expected_input_tokens.
        6. Если число токенов относится к выходу/ответу, заполняй expected_output_tokens.
        7. Если поле отсутствует, ставь null.
        8. Ответ должен быть валидным JSON.
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
                        "Пиши ясно, кратко и по делу. "
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
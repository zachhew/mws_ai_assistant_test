from __future__ import annotations

import re

from app.api.schemas import ChatMessage
from app.core.exceptions import ProfileBuildError
from app.core.logging import get_logger
from app.domain.user_case import UserCaseProfile
from app.services.openrouter_client import OpenRouterClient


class ProfileService:
    def __init__(self) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._llm_client: OpenRouterClient | None = None

        try:
            self._llm_client = OpenRouterClient()
        except Exception as exc:
            self._logger.warning("OpenRouter client initialization failed: %s", exc)
            self._llm_client = None

    def build_profile(
        self,
        messages: list[ChatMessage],
        previous_profile: UserCaseProfile | None = None,
    ) -> UserCaseProfile:
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        if not user_messages and previous_profile is None:
            raise ProfileBuildError("Cannot build profile without user messages.")

        latest_text = "\n".join(user_messages[-2:]) if user_messages else ""
        current = self._extract_profile_with_fallback(latest_text)

        if previous_profile is not None:
            current = self._merge_with_previous(previous_profile, current)

        return self._fill_assumptions(current)

    def _extract_profile_with_fallback(self, text: str) -> UserCaseProfile:
        if self._llm_client is not None:
            try:
                data = self._llm_client.extract_profile(text)
                profile = UserCaseProfile(**data)
                self._logger.info("Profile extracted via OpenRouter.")
                return profile
            except Exception as exc:
                self._logger.warning(
                    "OpenRouter profile extraction failed, fallback to deterministic parser: %s",
                    exc,
                )

        return self._extract_profile_from_text(text)

    def _extract_profile_from_text(self, text: str) -> UserCaseProfile:
        lowered = text.lower()

        task_type = "unknown"
        input_modality = "unknown"

        if any(word in lowered for word in ["embedding", "эмбед", "вектор"]):
            task_type = "embeddings"
            input_modality = "text"
        elif any(word in lowered for word in ["image", "картин", "изображен"]):
            task_type = "multimodal"
            input_modality = "text_image"
        elif any(word in lowered for word in ["code", "coding", "код", "программ"]):
            task_type = "coding"
            input_modality = "text"
        elif any(word in lowered for word in ["reasoning", "сложн", "рассужд", "аналит"]):
            task_type = "reasoning"
            input_modality = "text"
        elif any(word in lowered for word in ["chat", "чат", "support", "поддержк", "assistant"]):
            task_type = "chat"
            input_modality = "text"

        budget_limit_rub = self._extract_number(
            text,
            patterns=[
                r"budget[^0-9]{0,20}(\d[\d\s.,]*)",
                r"бюджет[^0-9]{0,20}(\d[\d\s.,]*)",
            ],
        )

        requests_per_month = self._extract_number(
            text,
            patterns=[
                r"(\d[\d\s.,]*)\s*(?:requests|reqs)\s*(?:per|/)\s*month",
                r"(\d[\d\s.,]*)\s*(?:запрос\w*)\s*(?:в|/)\s*месяц",
                r"(\d[\d\s.,]*)\s*k\s*(?:requests|reqs)\s*(?:per|/)\s*month",
            ],
            allow_k_suffix=True,
        )

        requests_per_day = self._extract_number(
            text,
            patterns=[
                r"(\d[\d\s.,]*)\s*(?:requests|reqs)\s*(?:per|/)\s*day",
                r"(\d[\d\s.,]*)\s*(?:запрос\w*)\s*(?:в|/)\s*день",
            ],
        )

        expected_input_tokens = self._extract_number(
            text,
            patterns=[
                r"input[^0-9]{0,20}(\d[\d\s.,]*)\s*tokens?",
                r"вход[^0-9]{0,20}(\d[\d\s.,]*)\s*ток",
                r"(\d[\d\s.,]*)\s*ток\w*\s*(?:на\s*вход|входных?)",
                r"(?:около|примерно|в среднем)?\s*(\d[\d\s.,]*)\s*ток\w*\s*(?:на\s*вход)?",
            ],
        )

        expected_output_tokens = self._extract_number(
            text,
            patterns=[
                r"output[^0-9]{0,20}(\d[\d\s.,]*)\s*tokens?",
                r"выход[^0-9]{0,20}(\d[\d\s.,]*)\s*ток",
                r"(\d[\d\s.,]*)\s*ток\w*\s*(?:на\s*выход|выходных?)",
                r"(?:около|примерно|в среднем)?\s*(\d[\d\s.,]*)\s*ток\w*\s*(?:на\s*выход)?",
            ],
        )

        quality_priority = "balanced"
        if any(
            word in lowered
            for word in ["high quality", "best quality", "максимальн", "высокое качество"]
        ):
            quality_priority = "high"
        elif any(word in lowered for word in ["cheap", "дешев", "минимальный бюджет"]):
            quality_priority = "low"

        latency_priority = "balanced"
        if any(word in lowered for word in ["fast", "low latency", "быстр", "низкая задержка"]):
            latency_priority = "high"

        needs_long_context = any(
            word in lowered
            for word in ["long context", "длинный контекст", "large context", "большой контекст"]
        )

        context_min_tokens = self._extract_number(
            text,
            patterns=[
                r"context[^0-9]{0,20}(\d[\d\s.,]*)\s*tokens?",
                r"контекст[^0-9]{0,20}(\d[\d\s.,]*)\s*ток",
            ],
        )

        return UserCaseProfile(
            task_type=task_type,
            input_modality=input_modality,
            expected_input_tokens=expected_input_tokens,
            expected_output_tokens=expected_output_tokens,
            requests_per_day=requests_per_day,
            requests_per_month=requests_per_month,
            quality_priority=quality_priority,
            latency_priority=latency_priority,
            budget_limit_rub=float(budget_limit_rub) if budget_limit_rub is not None else None,
            needs_long_context=needs_long_context,
            context_min_tokens=context_min_tokens,
            notes=text.strip() or None,
            assumptions=["Profile extracted via deterministic fallback parser."],
        )

    def _merge_with_previous(
        self,
        previous: UserCaseProfile,
        current: UserCaseProfile,
    ) -> UserCaseProfile:
        data = previous.model_dump()

        for key, value in current.model_dump().items():
            if key == "assumptions":
                continue
            if value not in (None, "", [], "unknown"):
                data[key] = value

        merged = UserCaseProfile(**data)
        merged.assumptions = list(previous.assumptions) + list(current.assumptions)
        return merged

    def _fill_assumptions(self, profile: UserCaseProfile) -> UserCaseProfile:
        assumptions = list(profile.assumptions)

        if profile.requests_per_month is None and profile.requests_per_day is not None:
            profile.requests_per_month = profile.requests_per_day * 30
            assumptions.append("Requests per month estimated as requests_per_day × 30.")

        if profile.expected_input_tokens is None:
            profile.expected_input_tokens = 500
            assumptions.append("Input tokens per request defaulted to 500.")

        if profile.expected_output_tokens is None:
            if profile.task_type == "embeddings":
                profile.expected_output_tokens = 0
                assumptions.append("Output tokens per request set to 0 for embeddings use case.")
            else:
                profile.expected_output_tokens = 200
                assumptions.append("Output tokens per request defaulted to 200.")

        if profile.input_modality == "unknown":
            profile.input_modality = "text"
            assumptions.append("Input modality defaulted to text.")

        if profile.task_type == "unknown":
            profile.task_type = "chat"
            assumptions.append("Task type defaulted to chat.")

        profile.assumptions = assumptions
        return profile

    def _extract_number(
        self,
        text: str,
        patterns: list[str],
        allow_k_suffix: bool = False,
    ) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue

            raw = match.group(1).replace(" ", "").replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue

            suffix = match.group(0).lower()
            if allow_k_suffix and "k" in suffix:
                value *= 1000

            return int(value)

        return None

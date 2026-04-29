from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from app.agent.session_manager import SessionManager
from app.agent.workflow_models import RankingAgentOutput
from app.core.logging import get_logger
from app.domain.user_case import UserCaseProfile
from app.services.catalog_service import CatalogService
from app.services.estimator import CostEstimator
from app.services.profile_service import ProfileService
from app.services.recommender import ModelRecommender
from app.services.report_builder import ReportBuilder


class RecommendationPreparationAgent(BaseAgent):
    def __init__(
        self,
        *,
        name: str,
        session_manager: SessionManager,
        profile_service: ProfileService,
        catalog_service: CatalogService,
        estimator: CostEstimator,
        recommender: ModelRecommender,
    ) -> None:
        super().__init__(name=name)
        self._session_manager = session_manager
        self._profile_service = profile_service
        self._catalog_service = catalog_service
        self._estimator = estimator
        self._recommender = recommender
        self._logger = get_logger(self.__class__.__name__)

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        raw_request = ctx.session.state.get("raw_user_request")
        request_text = raw_request if isinstance(raw_request, str) else ""
        target_session_id = self._resolve_target_session_id(ctx)
        raw_profile = self._normalize_profile_payload(
            self._load_json_state(ctx.session.state.get("user_profile_json"))
        )
        extracted_profile = UserCaseProfile.model_validate(raw_profile)
        deterministic_profile = self._profile_service.extract_profile_from_text(request_text)
        extracted_profile = self._profile_service.merge_profiles(
            base_profile=deterministic_profile,
            override_profile=extracted_profile,
        )

        state = self._session_manager.get(target_session_id)
        previous_profile = state.last_user_case if state is not None else None
        profile = self._profile_service.finalize_profile(extracted_profile, previous_profile)

        full_catalog = self._catalog_service.get_catalog(force_refresh=False)
        filtered_catalog = self._recommender.filter_catalog(profile, full_catalog)
        costs = self._estimator.estimate_for_catalog(profile, filtered_catalog)
        ranking_context = self._recommender.build_ranking_context(profile, filtered_catalog, costs)

        self._session_manager.save_artifacts(
            session_id=target_session_id,
            profile=profile,
            catalog=filtered_catalog,
            costs=costs,
        )

        self._logger.info(
            "Prepared ranking context for %s options in session %s.",
            len(ranking_context.options),
            target_session_id,
        )

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text="Prepared ranking context.")]),
            actions=EventActions(
                skip_summarization=True,
                state_delta={
                    "user_profile": profile.model_dump(exclude_none=True),
                    "ranking_context_json": ranking_context.model_dump_json(exclude_none=True),
                },
            ),
        )

    def _load_json_state(self, raw_value: object) -> dict:
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            return json.loads(raw_value)
        return {}

    def _resolve_target_session_id(self, ctx: InvocationContext) -> str:
        root_session_id = ctx.session.state.get("root_session_id")
        if isinstance(root_session_id, str) and root_session_id:
            return root_session_id
        return ctx.session.id

    def _normalize_profile_payload(self, payload: dict) -> dict:
        normalized = dict(payload)

        assumptions = normalized.get("assumptions")
        if isinstance(assumptions, dict):
            normalized["assumptions"] = [
                f"{key}: {value}" for key, value in assumptions.items() if value not in (None, "")
            ]
        elif isinstance(assumptions, str):
            normalized["assumptions"] = [assumptions]
        elif not isinstance(assumptions, list):
            normalized["assumptions"] = []

        alias_map = {
            "monthly_requests": "requests_per_month",
            "daily_requests": "requests_per_day",
            "input_tokens": "expected_input_tokens",
            "output_tokens": "expected_output_tokens",
            "budget": "budget_limit_rub",
            "min_context_tokens": "context_min_tokens",
        }
        for alias, canonical in alias_map.items():
            if canonical not in normalized and alias in normalized:
                normalized[canonical] = normalized[alias]

        for key in (
            "expected_input_tokens",
            "expected_output_tokens",
            "requests_per_day",
            "requests_per_month",
            "context_min_tokens",
        ):
            normalized[key] = self._coerce_int(normalized.get(key))

        normalized["budget_limit_rub"] = self._coerce_float(normalized.get("budget_limit_rub"))
        normalized["needs_long_context"] = self._coerce_bool(normalized.get("needs_long_context"))

        return normalized

    def _coerce_int(self, value: object) -> int | None:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str):
            sanitized = value.replace(" ", "").replace(",", ".")
            try:
                return int(float(sanitized))
            except ValueError:
                return None
        return None

    def _coerce_float(self, value: object) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            sanitized = value.replace(" ", "").replace(",", ".")
            try:
                return float(sanitized)
            except ValueError:
                return None
        return None

    def _coerce_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "да"}
        if isinstance(value, int | float):
            return bool(value)
        return False


class RecommendationFinalizationAgent(BaseAgent):
    def __init__(
        self,
        *,
        name: str,
        session_manager: SessionManager,
        recommender: ModelRecommender,
        report_builder: ReportBuilder,
    ) -> None:
        super().__init__(name=name)
        self._session_manager = session_manager
        self._recommender = recommender
        self._report_builder = report_builder
        self._logger = get_logger(self.__class__.__name__)

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        target_session_id = self._resolve_target_session_id(ctx)
        state = self._session_manager.get(target_session_id)
        if state is None or state.last_user_case is None:
            raise RuntimeError("Profile is missing before recommendation finalization.")
        if state.last_catalog is None or state.last_cost_estimates is None:
            raise RuntimeError("Catalog or cost estimates are missing before finalization.")

        ranking_output_raw = self._load_json_state(ctx.session.state.get("ranking_output_json"))
        ranking_output = RankingAgentOutput.model_validate(ranking_output_raw)

        recommendations = self._recommender.materialize_recommendations(
            profile=state.last_user_case,
            catalog=state.last_catalog,
            cost_estimates=state.last_cost_estimates,
            ranking_output=ranking_output,
        )
        report = self._report_builder.build(
            profile=state.last_user_case,
            recommendations=recommendations,
            costs=state.last_cost_estimates,
        )

        self._session_manager.save_artifacts(
            session_id=target_session_id,
            recommendations=recommendations,
            report=report,
        )

        self._logger.info(
            "Finalized recommendation report with %s recommendations in session %s.",
            len(recommendations),
            target_session_id,
        )

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text="Recommendation report finalized.")]),
            actions=EventActions(
                skip_summarization=True,
                state_delta={
                    "report_json": report.model_dump_json(exclude_none=True),
                },
            ),
        )

    def _load_json_state(self, raw_value: object) -> dict:
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            return json.loads(raw_value)
        return {}

    def _resolve_target_session_id(self, ctx: InvocationContext) -> str:
        root_session_id = ctx.session.state.get("root_session_id")
        if isinstance(root_session_id, str) and root_session_id:
            return root_session_id
        return ctx.session.id

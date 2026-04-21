from __future__ import annotations

from app.domain.catalog import CatalogEntry
from app.domain.recommendation import CostEstimate
from app.domain.user_case import UserCaseProfile


class CostEstimator:
    def estimate_for_catalog(
        self,
        profile: UserCaseProfile,
        catalog: list[CatalogEntry],
    ) -> list[CostEstimate]:
        estimates: list[CostEstimate] = []

        for entry in catalog:
            estimate = self.estimate_for_model(profile, entry)
            if estimate is not None:
                estimates.append(estimate)

        return estimates

    def estimate_for_model(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
    ) -> CostEstimate | None:
        if entry.pricing is None:
            return None

        requests_per_month = self._resolve_requests_per_month(profile)
        input_tokens = self._resolve_input_tokens(profile)
        output_tokens = self._resolve_output_tokens(profile)

        monthly_input_tokens = requests_per_month * input_tokens
        monthly_output_tokens = requests_per_month * output_tokens

        input_price = entry.pricing.input_price_per_1k_tokens_rub or 0.0
        output_price = entry.pricing.output_price_per_1k_tokens_rub or 0.0

        monthly_input_cost = (monthly_input_tokens / 1000) * input_price
        monthly_output_cost = (monthly_output_tokens / 1000) * output_price
        total_monthly_cost = monthly_input_cost + monthly_output_cost

        within_budget = None
        if profile.budget_limit_rub is not None:
            within_budget = total_monthly_cost <= profile.budget_limit_rub

        return CostEstimate(
            model_name=entry.model.name,
            requests_per_month=requests_per_month,
            input_tokens_per_request=input_tokens,
            output_tokens_per_request=output_tokens,
            monthly_input_tokens=monthly_input_tokens,
            monthly_output_tokens=monthly_output_tokens,
            monthly_input_cost_rub=round(monthly_input_cost, 4),
            monthly_output_cost_rub=round(monthly_output_cost, 4),
            total_monthly_cost_rub=round(total_monthly_cost, 4),
            within_budget=within_budget,
            assumptions=self._build_assumptions(profile, entry),
        )

    def _resolve_requests_per_month(self, profile: UserCaseProfile) -> int:
        return profile.requests_per_month or 0

    def _resolve_input_tokens(self, profile: UserCaseProfile) -> int:
        return profile.expected_input_tokens or 0

    def _resolve_output_tokens(self, profile: UserCaseProfile) -> int:
        return profile.expected_output_tokens or 0

    def _build_assumptions(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
    ) -> list[str]:
        assumptions = list(profile.assumptions)

        if entry.pricing and entry.pricing.billing_unit_tokens is not None:
            assumptions.append(
                "Estimate is aggregate-based and does not simulate per-call billing unit accumulation."
            )

        return assumptions
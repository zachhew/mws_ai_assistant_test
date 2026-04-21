from __future__ import annotations

from app.domain.recommendation import (
    CostEstimate,
    RecommendationCandidate,
    RecommendationReport,
)
from app.domain.user_case import UserCaseProfile


class ReportBuilder:
    def build(
        self,
        profile: UserCaseProfile,
        recommendations: list[RecommendationCandidate],
        costs: list[CostEstimate],
    ) -> RecommendationReport:
        assumptions = self._build_assumptions(profile, costs)
        limitations = self._build_limitations(profile, recommendations)
        summary = self._build_summary(recommendations)

        return RecommendationReport(
            input_data=profile,
            recommended_models=recommendations,
            calculations=costs,
            limitations=limitations,
            assumptions=assumptions,
            final_summary=summary,
        )

    def _build_assumptions(
        self,
        profile: UserCaseProfile,
        costs: list[CostEstimate],
    ) -> list[str]:
        assumptions = list(profile.assumptions)

        if costs:
            assumptions.append(
                "Cost estimate is calculated from aggregate monthly token volume."
            )

        return assumptions

    def _build_limitations(
        self,
        profile: UserCaseProfile,
        recommendations: list[RecommendationCandidate],
    ) -> list[str]:
        limitations: list[str] = []

        if not recommendations:
            limitations.append("No models matched the current use-case constraints.")

        if profile.budget_limit_rub is None:
            limitations.append("Budget was not explicitly specified by the user.")

        limitations.append(
            "Actual billed amount may differ slightly from estimate due to provider billing rules."
        )

        return limitations

    def _build_summary(
        self,
        recommendations: list[RecommendationCandidate],
    ) -> str:
        if not recommendations:
            return "No suitable models were found for the current scenario."

        best = recommendations[0]
        return (
            f"Best overall option: {best.model_name}. "
            f"Reason: {best.fit_summary}"
        )
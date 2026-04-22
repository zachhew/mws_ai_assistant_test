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
        summary = self._build_summary(profile, recommendations)

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
                "Стоимость рассчитана как агрегированная месячная оценка по объему токенов."
            )

        has_billing_units = any(cost.assumptions for cost in costs)
        if has_billing_units:
            assumptions.append(
                "Расчет не моделирует накопление токенов по каждой отдельной единице биллинга."
            )

        return self._deduplicate(assumptions)

    def _build_limitations(
        self,
        profile: UserCaseProfile,
        recommendations: list[RecommendationCandidate],
    ) -> list[str]:
        limitations: list[str] = []

        if not recommendations:
            limitations.append("По текущим ограничениям подходящие модели не найдены.")

        if profile.budget_limit_rub is None:
            limitations.append("Пользователь не указал явный бюджет.")

        has_in_budget = any(
            recommendation.cost_estimate is not None
            and recommendation.cost_estimate.within_budget is True
            for recommendation in recommendations
        )

        if recommendations and profile.budget_limit_rub is not None and not has_in_budget:
            limitations.append("Ни одна из подходящих моделей не укладывается в указанный бюджет.")

        limitations.append(
            "Фактический billed amount может незначительно отличаться из-за правил тарификации провайдера."
        )

        return self._deduplicate(limitations)

    def _build_summary(
        self,
        profile: UserCaseProfile,
        recommendations: list[RecommendationCandidate],
    ) -> str:
        if not recommendations:
            return (
                "По текущему сценарию подходящие модели не найдены. "
                "Стоит уточнить требования, бюджет или ожидаемую нагрузку."
            )

        best = recommendations[0]
        best_cost = best.cost_estimate.total_monthly_cost_rub if best.cost_estimate else None
        within_budget = best.cost_estimate is not None and best.cost_estimate.within_budget is True

        if within_budget:
            if best_cost is not None:
                return (
                    f"Наиболее подходящий вариант — {best.model_name}. "
                    f"Оценочная месячная стоимость составляет {best_cost:.2f} RUB."
                )
            return f"Наиболее подходящий вариант — {best.model_name}."

        if profile.budget_limit_rub is not None and best_cost is not None:
            return (
                f"Ближайший технически подходящий вариант — {best.model_name}, "
                f"но его оценочная месячная стоимость {best_cost:.2f} RUB "
                f"превышает бюджет {profile.budget_limit_rub:.2f} RUB."
            )

        return (
            f"Ближайший технически подходящий вариант — {best.model_name}, "
            "но он не укладывается в текущие ограничения."
        )

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []

        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)

        return result

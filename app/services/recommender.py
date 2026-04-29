from __future__ import annotations

from app.domain.catalog import CatalogEntry
from app.domain.recommendation import CostEstimate, RecommendationCandidate
from app.domain.user_case import UserCaseProfile


class ModelRecommender:
    def recommend(
        self,
        profile: UserCaseProfile,
        catalog: list[CatalogEntry],
        cost_estimates: list[CostEstimate],
    ) -> list[RecommendationCandidate]:
        estimates_by_name = {estimate.model_name: estimate for estimate in cost_estimates}
        filtered = self.filter_catalog(profile, catalog)
        price_bounds = self._build_price_bounds(filtered, estimates_by_name)

        candidates: list[RecommendationCandidate] = []
        for entry in filtered:
            estimate = estimates_by_name.get(entry.model.name)
            score = self._score_candidate(profile, entry, estimate, price_bounds)
            candidate = self._build_candidate(profile, entry, score, estimate)
            candidates.append(candidate)

        candidates.sort(key=self._sort_key, reverse=True)
        return self._assign_fit_labels(candidates[:3])

    def filter_catalog(
        self,
        profile: UserCaseProfile,
        catalog: list[CatalogEntry],
    ) -> list[CatalogEntry]:
        result: list[CatalogEntry] = []

        for entry in catalog:
            model = entry.model

            if profile.task_type != "embeddings" and model.is_embedding_model:
                continue
            if profile.task_type == "embeddings" and not model.is_embedding_model:
                continue
            if profile.input_modality == "text_image" and not model.supports_image_input:
                continue
            if profile.context_min_tokens and model.context_window_tokens:
                if model.context_window_tokens < profile.context_min_tokens:
                    continue

            result.append(entry)

        return result

    def _score_candidate(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
        cost_estimate: CostEstimate | None,
        price_bounds: tuple[float, float] | None,
    ) -> float:
        score = 0.0
        model = entry.model

        score += self._score_modality(profile, entry)
        score += self._score_task_type(profile, entry)
        score += self._score_context(profile, entry)

        if cost_estimate is not None:
            score += self._score_budget(cost_estimate)
            score += self._score_cost_efficiency(cost_estimate, price_bounds)
        elif profile.budget_limit_rub is not None:
            score -= 5

        if profile.quality_priority == "high" and model.family in {"qwen", "llama", "kimi", "glm"}:
            score += 8

        if profile.task_type == "coding" and "coder" in model.name.lower():
            score += 16

        return round(max(score, 0.0), 3)

    def _build_price_bounds(
        self,
        catalog: list[CatalogEntry],
        estimates_by_name: dict[str, CostEstimate],
    ) -> tuple[float, float] | None:
        costs = [
            estimate.total_monthly_cost_rub
            for entry in catalog
            if (estimate := estimates_by_name.get(entry.model.name)) is not None
        ]
        if not costs:
            return None
        return min(costs), max(costs)

    def _score_modality(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
    ) -> float:
        if profile.input_modality == "text_image":
            return 20.0 if entry.model.supports_image_input else 0.0
        return 12.0 if entry.model.supports_text_input else 0.0

    def _score_task_type(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
    ) -> float:
        if profile.task_type == "embeddings":
            return 30.0 if entry.model.is_embedding_model else 0.0
        return 22.0 if not entry.model.is_embedding_model else 0.0

    def _score_context(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
    ) -> float:
        context_window = entry.model.context_window_tokens
        if not context_window:
            return 0.0

        if profile.context_min_tokens:
            ratio = min(context_window / profile.context_min_tokens, 1.5)
            return min(ratio * 12, 18.0)

        if profile.needs_long_context:
            return min(context_window / 64000 * 15, 15.0)

        return 5.0

    def _score_budget(self, cost_estimate: CostEstimate) -> float:
        if cost_estimate.within_budget is True:
            return 28.0
        if cost_estimate.within_budget is False:
            return -18.0
        return 0.0

    def _score_cost_efficiency(
        self,
        cost_estimate: CostEstimate,
        price_bounds: tuple[float, float] | None,
    ) -> float:
        if price_bounds is None:
            return 0.0

        min_cost, max_cost = price_bounds
        if max_cost <= min_cost:
            return 10.0

        normalized = (cost_estimate.total_monthly_cost_rub - min_cost) / (max_cost - min_cost)
        return round((1 - normalized) * 15, 3)

    def _sort_key(self, candidate: RecommendationCandidate) -> tuple[bool, float, float]:
        estimate = candidate.cost_estimate
        within_budget = estimate is not None and estimate.within_budget is True
        total_cost = estimate.total_monthly_cost_rub if estimate is not None else float("inf")
        return within_budget, candidate.score, -total_cost

    def _build_candidate(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
        score: float,
        cost_estimate: CostEstimate | None,
    ) -> RecommendationCandidate:
        strengths: list[str] = []
        weaknesses: list[str] = []
        risks: list[str] = []

        if entry.model.supports_image_input:
            strengths.append("Supports image input.")
        if entry.model.context_window_tokens:
            strengths.append(f"Context window: {entry.model.context_window_tokens} tokens.")
        if cost_estimate and cost_estimate.within_budget is True:
            strengths.append("Fits within stated budget.")

        if cost_estimate and cost_estimate.within_budget is False:
            weaknesses.append("Estimated monthly cost exceeds stated budget.")

        if entry.pricing is None:
            risks.append("Pricing data unavailable.")

        fit_summary = self._build_fit_summary(profile, entry, cost_estimate)

        return RecommendationCandidate(
            model_name=entry.model.name,
            score=score,
            fit_summary=fit_summary,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            cost_estimate=cost_estimate,
        )

    def _assign_fit_labels(
        self,
        candidates: list[RecommendationCandidate],
    ) -> list[RecommendationCandidate]:
        if not candidates:
            return candidates

        has_in_budget = any(
            candidate.cost_estimate is not None and candidate.cost_estimate.within_budget is True
            for candidate in candidates
        )

        candidates[0].fit_label = "best_fit"

        if len(candidates) > 1:
            candidates[1].fit_label = "budget_option" if has_in_budget else "alternative"

        if len(candidates) > 2:
            candidates[2].fit_label = "premium_option" if has_in_budget else "alternative"

        return candidates

    def _build_fit_summary(
        self,
        profile: UserCaseProfile,
        entry: CatalogEntry,
        cost_estimate: CostEstimate | None,
    ) -> str:
        parts = [f"Suitable for {profile.task_type} workload"]

        if entry.model.supports_image_input:
            parts.append("supports multimodal input")

        if cost_estimate is not None:
            parts.append(
                f"estimated monthly cost is {cost_estimate.total_monthly_cost_rub:.2f} RUB"
            )
            if cost_estimate.within_budget is True:
                parts.append("fits the stated budget")
            elif cost_estimate.within_budget is False:
                parts.append("exceeds the stated budget")

        return ", ".join(parts) + "."

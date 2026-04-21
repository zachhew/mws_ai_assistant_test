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
        filtered = self._filter_candidates(profile, catalog)

        candidates: list[RecommendationCandidate] = []
        for entry in filtered:
            estimate = estimates_by_name.get(entry.model.name)
            score = self._score_candidate(profile, entry, estimate)
            candidate = self._build_candidate(profile, entry, score, estimate)
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        return self._assign_fit_labels(candidates[:3])

    def _filter_candidates(
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
    ) -> float:
        score = 0.0
        model = entry.model

        if profile.input_modality == "text" and model.supports_text_input:
            score += 20
        if profile.input_modality == "text_image" and model.supports_image_input:
            score += 25
        if profile.task_type == "embeddings" and model.is_embedding_model:
            score += 30
        if profile.task_type in {"chat", "reasoning", "coding", "multimodal"} and not model.is_embedding_model:
            score += 15

        if model.context_window_tokens:
            if profile.needs_long_context:
                score += min(model.context_window_tokens / 10000, 20)
            else:
                score += 5

        if cost_estimate is not None:
            if cost_estimate.within_budget is True:
                score += 20
            elif cost_estimate.within_budget is False:
                score -= 15

            total_cost = cost_estimate.total_monthly_cost_rub
            if total_cost <= 1000:
                score += 15
            elif total_cost <= 5000:
                score += 10
            elif total_cost <= 15000:
                score += 5

        if profile.quality_priority == "high":
            if model.family in {"qwen", "llama", "kimi", "glm"}:
                score += 10

        if profile.task_type == "coding" and "coder" in model.name.lower():
            score += 15

        return round(score, 3)

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
        if entry.model.is_embedding_model and profile.task_type != "embeddings":
            risks.append("Embedding model may not fit generative scenario.")

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

        candidates[0].fit_label = "best_fit"

        if len(candidates) > 1:
            candidates[1].fit_label = "budget_option"

        if len(candidates) > 2:
            candidates[2].fit_label = "premium_option"

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

        return ", ".join(parts) + "."
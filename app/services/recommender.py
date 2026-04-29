from __future__ import annotations

from app.agent.workflow_models import RankingAgentOutput, RankingContextPayload, RankingOption
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

        candidates: list[RecommendationCandidate] = []
        for entry in filtered:
            estimate = estimates_by_name.get(entry.model.name)
            candidate = self._build_candidate(profile, entry, 0.0, estimate)
            candidates.append(candidate)

        candidates.sort(key=self._fallback_sort_key)
        top_candidates = candidates[:3]
        self._assign_descending_scores(top_candidates)
        return self._assign_fit_labels(top_candidates)

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

    def build_ranking_context(
        self,
        profile: UserCaseProfile,
        catalog: list[CatalogEntry],
        cost_estimates: list[CostEstimate],
    ) -> RankingContextPayload:
        estimates_by_name = {estimate.model_name: estimate for estimate in cost_estimates}
        options: list[RankingOption] = []

        for entry in catalog:
            estimate = estimates_by_name.get(entry.model.name)
            candidate = self._build_candidate(profile, entry, 0.0, estimate)
            options.append(
                RankingOption(
                    model_name=candidate.model_name,
                    model_family=entry.model.family,
                    context_window_tokens=entry.model.context_window_tokens,
                    supports_image_input=entry.model.supports_image_input,
                    fit_summary=candidate.fit_summary,
                    strengths=candidate.strengths,
                    weaknesses=candidate.weaknesses,
                    risks=candidate.risks,
                    cost_estimate=estimate,
                )
            )

        return RankingContextPayload(profile=profile, options=options)

    def materialize_recommendations(
        self,
        profile: UserCaseProfile,
        catalog: list[CatalogEntry],
        cost_estimates: list[CostEstimate],
        ranking_output: RankingAgentOutput,
    ) -> list[RecommendationCandidate]:
        estimates_by_name = {estimate.model_name: estimate for estimate in cost_estimates}
        entries_by_name = {entry.model.name: entry for entry in catalog}

        recommendations: list[RecommendationCandidate] = []
        for index, decision in enumerate(ranking_output.recommended_models[:3]):
            entry = entries_by_name.get(decision.model_name)
            if entry is None:
                continue

            estimate = estimates_by_name.get(decision.model_name)
            candidate = self._build_candidate(
                profile=profile,
                entry=entry,
                score=float(max(3 - index, 1)),
                cost_estimate=estimate,
            )
            candidate.fit_summary = decision.rationale
            recommendations.append(candidate)

        if not recommendations:
            return self.recommend(profile, catalog, cost_estimates)

        return self._assign_fit_labels(recommendations)

    def _fallback_sort_key(self, candidate: RecommendationCandidate) -> tuple[int, float, int, str]:
        estimate = candidate.cost_estimate
        within_budget_rank = 1
        total_cost = float("inf")

        if estimate is not None:
            total_cost = estimate.total_monthly_cost_rub
            within_budget_rank = 0 if estimate.within_budget is True else 2

        context_window = 0
        for strength in candidate.strengths:
            if strength.startswith("Context window: "):
                try:
                    context_value = strength.split(": ", maxsplit=1)[1]
                    context_window = int(context_value.split(" ", maxsplit=1)[0])
                except (IndexError, ValueError):
                    context_window = 0

        return within_budget_rank, total_cost, -context_window, candidate.model_name.lower()

    def _assign_descending_scores(
        self,
        candidates: list[RecommendationCandidate],
    ) -> None:
        for index, candidate in enumerate(candidates):
            candidate.score = float(max(len(candidates) - index, 1))

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

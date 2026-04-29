from __future__ import annotations

from collections.abc import Iterator

from app.domain.recommendation import RecommendationReport


class FinalAnswerStreamingService:
    def stream_answer(self, report: RecommendationReport) -> Iterator[str]:
        final_text = (report.final_answer_text or report.final_summary).strip()
        if not final_text:
            return

        yield from self._split_text(final_text)

    def _split_text(self, text: str, chunk_size: int = 160) -> Iterator[str]:
        words = text.split()
        current: list[str] = []
        current_len = 0

        for word in words:
            addition = len(word) if not current else len(word) + 1
            if current and current_len + addition > chunk_size:
                yield " ".join(current) + " "
                current = [word]
                current_len = len(word)
                continue

            current.append(word)
            current_len += addition

        if current:
            yield " ".join(current)

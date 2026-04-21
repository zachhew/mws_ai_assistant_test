from __future__ import annotations

import httpx

from app.core.exceptions import MWSFetchError
from app.core.logging import get_logger


class MWSClient:
    def __init__(self, timeout_seconds: float = 15.0) -> None:
        self._timeout = timeout_seconds
        self._logger = get_logger(self.__class__.__name__)

    def fetch_text(self, url: str) -> str:
        self._logger.info("Fetching MWS page: %s", url)

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MWSFetchError(f"Failed to fetch MWS page: {url}") from exc

        if not response.text.strip():
            raise MWSFetchError(f"Fetched empty response from MWS page: {url}")

        return response.text
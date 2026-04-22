from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppError,
    CatalogBuildError,
    EstimationError,
    MWSFetchError,
    MWSParseError,
    ProfileBuildError,
    RecommendationError,
    ReportBuildError,
    SessionError,
)
from app.core.logging import get_logger


logger = get_logger("api.error_handlers")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.exception(
            "Application error. path=%s method=%s error=%s",
            request.url.path,
            request.method,
            exc.__class__.__name__,
        )

        status_code = _map_app_error_to_status_code(exc)
        payload = {
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        }
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unexpected error. path=%s method=%s",
            request.url.path,
            request.method,
        )
        payload = {
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected internal error occurred.",
            }
        }
        return JSONResponse(status_code=500, content=payload)


def _map_app_error_to_status_code(exc: AppError) -> int:
    if isinstance(exc, (MWSFetchError, MWSParseError, CatalogBuildError)):
        return 502

    if isinstance(
        exc,
        (
            ProfileBuildError,
            EstimationError,
            RecommendationError,
            ReportBuildError,
            SessionError,
        ),
    ):
        return 400

    return 500
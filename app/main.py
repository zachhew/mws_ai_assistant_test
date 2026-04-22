from fastapi import FastAPI

from app.api.error_handlers import register_exception_handlers
from app.api.routes.chat_completions import router as chat_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Assistant for selecting MWS GPT Model Hub models via OpenAI-compatible API.",
    )

    register_exception_handlers(app)

    @app.get("/health", tags=["service"])
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "env": settings.app_env,
        }

    app.include_router(chat_router)
    return app


app = create_app()

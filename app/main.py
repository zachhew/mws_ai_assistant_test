from fastapi import FastAPI

from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Assistant for selecting MWS GPT Model Hub models via OpenAI-compatible API.",
    )

    @app.get("/health", tags=["service"])
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "env": settings.app_env,
        }

    return app


app = create_app()
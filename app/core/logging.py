import logging
import sys


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class SuppressAdkAppNameMismatchFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "App name mismatch detected" not in message


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    root_logger = logging.getLogger()
    target_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(target_level)

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.handlers = []
    litellm_logger.propagate = True
    litellm_logger.setLevel(logging.INFO)

    litellm_lower_logger = logging.getLogger("litellm")
    litellm_lower_logger.handlers = []
    litellm_lower_logger.propagate = True
    litellm_lower_logger.setLevel(logging.INFO)

    adk_runner_logger = logging.getLogger("google_adk.google.adk.runners")
    adk_runner_logger.addFilter(SuppressAdkAppNameMismatchFilter())

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

import logging
import sys


LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(message)s"
)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
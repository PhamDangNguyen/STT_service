import logging
import logging.handlers
from pathlib import Path

from .config import Settings

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(settings: Settings) -> None:
    """Configure the root stt_service logger with console + rotating file handlers."""
    log_dir = Path(settings.log_folder)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = settings.log_level.upper()

    root = logging.getLogger("stt_service")
    root.setLevel(level)
    root.propagate = False

    # Avoid duplicate handlers on reload (e.g. uvicorn --reload)
    if root.handlers:
        root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(_FORMATTER)
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "stt_service.log",
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the stt_service namespace.

    Usage:
        logger = get_logger(__name__)
    """
    return logging.getLogger(f"stt_service.{name}")

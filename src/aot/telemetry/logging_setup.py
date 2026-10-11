"""Structured application logging with final-output credential redaction."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


URL_USERINFO = re.compile(
    r"\b((?:https?|rtsps?)://)[^/@\s]+@",
    flags=re.IGNORECASE,
)
CONTEXT_FIELDS = ("source_id", "session_id", "track_id", "identity_id")


class LoggingSetupError(RuntimeError):
    """Raised when application logging cannot be configured safely."""


def redact_text(value: str, secrets: Iterable[str] = ()) -> str:
    """Remove explicit secrets and URL user-info from displayable text."""

    redacted = str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        redacted = redacted.replace(secret, "[REDACTED]")
    return URL_USERINFO.sub(r"\1[REDACTED]@", redacted)


class _JsonFormatter(logging.Formatter):
    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = tuple(secret for secret in secrets if secret)

    def _redact(self, value: Any) -> str:
        return redact_text(str(value), self._secrets)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": self._redact(record.getMessage()),
        }
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = self._redact(value)
        if record.exc_info:
            payload["exception"] = self._redact(self.formatException(record.exc_info))
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return redact_text(serialized, self._secrets)


def _close_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_aot_owned", False):
            logger.removeHandler(handler)
            handler.close()


def configure_logging(
    level: str,
    log_dir: Path,
    secrets: Iterable[str] = (),
) -> logging.Logger:
    """Configure the project logger with stderr and rotating JSON-line output."""

    log_dir = Path(log_dir)
    secret_values = tuple(secret for secret in secrets if secret)
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise LoggingSetupError(f"Unsupported log level {level!r}")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LoggingSetupError(
            redact_text(f"Cannot create log directory {log_dir}", secret_values)
        ) from error

    formatter = _JsonFormatter(secret_values)
    logger = logging.getLogger("aot")
    _close_owned_handlers(logger)
    logger.setLevel(numeric_level)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler._aot_owned = True  # type: ignore[attr-defined]
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(numeric_level)

    try:
        file_handler = RotatingFileHandler(
            log_dir / "aot.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    except OSError as error:
        stream_handler.close()
        raise LoggingSetupError(
            redact_text(f"Cannot open log file under {log_dir}", secret_values)
        ) from error
    file_handler._aot_owned = True  # type: ignore[attr-defined]
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

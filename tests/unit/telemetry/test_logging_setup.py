from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from aot.telemetry.logging_setup import (
    LoggingSetupError,
    configure_logging,
    redact_text,
)


def _flush(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _read_log(log_dir: Path) -> list[dict[str, object]]:
    lines = (log_dir / "aot.log").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_json_log_contains_timestamp_level_logger_message_and_context(tmp_path) -> None:
    logger = configure_logging("INFO", tmp_path)
    try:
        logger.info(
            "identity linked",
            extra={
                "source_id": "camera-1",
                "session_id": "session-1",
                "track_id": "track-7",
                "identity_id": "identity-3",
            },
        )
        _flush(logger)

        record = _read_log(tmp_path)[0]

        assert str(record["timestamp"]).endswith("Z")
        assert record["level"] == "INFO"
        assert record["logger"] == "aot"
        assert record["message"] == "identity linked"
        assert record["source_id"] == "camera-1"
        assert record["session_id"] == "session-1"
        assert record["track_id"] == "track-7"
        assert record["identity_id"] == "identity-3"
    finally:
        _close(logger)


def test_final_json_redacts_literals_url_userinfo_arguments_and_exceptions(tmp_path) -> None:
    secret = "literal-secret"
    rtsp_url = "rtsp://" + "admin:camera-pass" + "@example.test/live"
    logger = configure_logging("INFO", tmp_path, secrets=[secret])
    try:
        try:
            raise RuntimeError(f"failure included {secret}")
        except RuntimeError:
            logger.exception("connection %s failed for %s", secret, rtsp_url)
        _flush(logger)

        raw = (tmp_path / "aot.log").read_text(encoding="utf-8")
        record = json.loads(raw)

        assert secret not in raw
        assert "admin:camera-pass" not in raw
        assert raw.count("[REDACTED]") >= 3
        assert "exception" in record
    finally:
        _close(logger)


def test_redact_text_hides_supported_url_credentials_and_literal_secrets() -> None:
    https_url = "https://" + "user:pass" + "@example.test/path"
    rtsp_url = "rtsp://" + "cam:secret" + "@host/live"
    value = f"token-123 {https_url} {rtsp_url}"

    redacted = redact_text(value, ["token-123"])

    assert redacted == (
        "[REDACTED] https://[REDACTED]@example.test/path "
        "rtsp://[REDACTED]@host/live"
    )


def test_reconfiguration_replaces_owned_handlers_without_duplicate_lines(tmp_path) -> None:
    first = configure_logging("INFO", tmp_path)
    second = configure_logging("INFO", tmp_path)
    try:
        assert first is second

        second.info("one event")
        _flush(second)

        assert len(_read_log(tmp_path)) == 1
    finally:
        _close(second)


def test_unwritable_log_directory_raises_sanitized_error(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "forbidden"
    secret = "directory-secret"

    def deny_directory_creation(*_args, **_kwargs) -> None:
        raise PermissionError(f"access denied with {secret}")

    monkeypatch.setattr(Path, "mkdir", deny_directory_creation)

    with pytest.raises(LoggingSetupError) as error:
        configure_logging("INFO", log_dir, secrets=[secret])

    assert str(log_dir) in str(error.value)
    assert secret not in str(error.value)

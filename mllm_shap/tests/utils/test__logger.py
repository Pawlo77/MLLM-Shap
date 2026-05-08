"""Tests for logger utility."""

import logging

import pytest

from mllm_shap.utils.logger import get_logger


def test_get_logger_configures_handler_and_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_logger should attach stream handler and disable propagation."""
    logger = logging.Logger("unit.logger")
    logger.handlers.clear()
    logger.propagate = True
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger)

    configured = get_logger("ignored")
    assert configured.handlers
    assert configured.propagate is False


def test_get_logger_does_not_duplicate_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls should not create duplicate handlers."""
    logger = logging.Logger("unit.idempotent")
    logger.handlers.clear()
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger)

    first = get_logger("ignored")
    first_count = len(first.handlers)
    second = get_logger("ignored")
    second_count = len(second.handlers)

    assert first_count == second_count

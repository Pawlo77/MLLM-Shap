"""Shared test helpers for liquid connector tests."""

from enum import IntEnum


class FakeLFMModality(IntEnum):
    """Minimal replacement for liquid_audio.LFMModality."""

    TEXT = 0
    AUDIO_OUT = 1
    AUDIO_IN = 2

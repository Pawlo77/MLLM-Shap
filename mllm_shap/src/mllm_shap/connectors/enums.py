"""Configuration possible roles."""

from enum import Enum


class ModalityFlag(int, Enum):
    """Possible modality flags."""

    TEXT = 0
    AUDIO = 1


class Role(int, Enum):
    """Possible roles."""

    USER = 0
    ASSISTANT = 1
    SYSTEM = 2

    def __str__(self) -> str:
        """String representation of the Role."""
        return self.name


class SystemRolesSetup(int, Enum):
    """Possible system roles setups."""

    NONE = 0
    SYSTEM = 1
    SYSTEM_ASSISTANT = 2


class ModelHistoryTrackingMode(int, Enum):
    """Possible model history tracking modes."""

    TEXT = 0
    AUDIO = 1
    TEXT_AUDIO = 2

"""Unit tests for connector enums."""

import pytest
from mllm_shap.connectors.enums import (
    ModelHistoryTrackingMode,
    ModalityFlag,
    Role,
    SystemRolesSetup,
)


class TestRole:
    """Tests for Role helpers."""

    @pytest.mark.parametrize(
        "ordinal,expected",
        [(0, Role.USER), (1, Role.ASSISTANT), (2, Role.SYSTEM)],
    )
    def test_from_ordinal_returns_expected_member(
        self, ordinal: int, expected: Role
    ) -> None:
        """Checks that from ordinal returns expected member."""
        assert Role.from_ordinal(ordinal) is expected

    def test_from_ordinal_raises_for_unknown_value(self) -> None:
        """Checks that from ordinal raises for unknown value."""
        with pytest.raises(ValueError, match="No Role found for ordinal 99"):
            _ = Role.from_ordinal(99)

    @pytest.mark.parametrize(
        "role,expected",
        [(Role.USER, "USER"), (Role.ASSISTANT, "ASSISTANT"), (Role.SYSTEM, "SYSTEM")],
    )
    def test_str_returns_enum_name(self, role: Role, expected: str) -> None:
        """Checks that str returns enum name."""
        assert str(role) == expected


class TestOtherEnums:
    """Smoke tests for plain enum values."""

    def test_modality_flag_values_are_stable(self) -> None:
        """Checks that modality flag values are stable."""
        assert ModalityFlag.IGNORE.value == -1
        assert ModalityFlag.TEXT.value == 0
        assert ModalityFlag.AUDIO.value == 1

    def test_system_roles_setup_values_are_stable(self) -> None:
        """Checks that system roles setup values are stable."""
        assert SystemRolesSetup.NONE.value == 0
        assert SystemRolesSetup.SYSTEM.value == 1
        assert SystemRolesSetup.SYSTEM_ASSISTANT.value == 2

    def test_model_history_tracking_mode_values_are_stable(self) -> None:
        """Checks that model history tracking mode values are stable."""
        assert ModelHistoryTrackingMode.TEXT.value == 0
        assert ModelHistoryTrackingMode.AUDIO.value == 1
        assert ModelHistoryTrackingMode.TEXT_AUDIO.value == 2

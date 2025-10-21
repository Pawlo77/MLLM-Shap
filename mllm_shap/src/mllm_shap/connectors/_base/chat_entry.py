"""Conversation entry data structure for audio and text modalities."""

from logging import Logger
from typing import cast

from pydantic import BaseModel, ConfigDict

from ...utils.audio import display_audio
from ...utils.logger import get_logger
from ..enums import ModalityFlag, Role

logger: Logger = get_logger(__name__)


class ChatEntry(BaseModel):
    """Conversation entry data structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content_type: int
    roles: list[int]
    content: list[str | bytes]
    shap_values: list[float | None] | None

    def display(self) -> None:
        """Display the ChatEntry content."""
        from IPython.display import display  # pylint: disable=import-outside-toplevel

        roles = sorted(set(self.roles))

        roles_str: list[str] = [str(Role(v)) for v in roles]
        print("BY: " + ", ".join(roles_str))

        if self.content_type == ModalityFlag.TEXT.value:
            print("TEXT CONTENT:")
            print("\t" + " ".join(cast(list[str], self.content)).replace("\n", "\n\t"))

        else:  # ModalityFlag.AUDIO
            print("AUDIO CONTENT:")
            audio_bytes = b"".join(cast(list[bytes], self.content))
            _ = display(display_audio(audio_bytes))  # type: ignore[no-untyped-call]

    def __str__(self) -> str:
        """String representation of the ChatEntry."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Official string representation of the ChatEntry."""
        if self.content_type == ModalityFlag.TEXT.value:
            content_str = ", ".join(cast(list[str], self.content)).replace("\n", "\\n")
        else:
            content_str = (
                f"Audio bytes of total length {sum(len(c) if isinstance(c, bytes) else 0 for c in self.content)}"
            )

        # Limit to 100 characters
        if len(content_str) > 100:  # pylint: disable=magic-value-comparison
            content_str = content_str[:100] + "..."

        # Build final representation
        return (
            f"ChatEntry("
            f"content_type={self.content_type}, "
            f"roles={self.roles}, "
            f"content='{content_str}', "
            f"shap_values={self.shap_values}"
            f")"
        )

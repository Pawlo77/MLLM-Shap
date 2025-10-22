"""Conversation entry data structure for audio and text modalities."""

from typing import cast

from pydantic import BaseModel
from pydantic import ConfigDict

from ...utils.audio import display_audio
from ..enums import ModalityFlag, Role


class ChatEntry(BaseModel):
    """Conversation entry data structure."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Configuration for pydantic model."""

    content_type: int
    """Modality of the content (e.g., text or audio), refer to :class:`ModalityFlag`."""
    roles: list[int]
    """List of roles associated with this entry, refer to :class:`Role`."""
    content: list[str | bytes]
    """Content of the entry, can be text (str) or audio bytes (bytes)."""
    shap_values: list[float | None] | None
    """SHAP values associated with the content tokens, if one have been computed for this entry."""

    def display(self) -> None:
        """
        Display the ChatEntry content.

        Example:
        ::
            BY: USER, SYSTEM
            TEXT CONTENT:
                <|im_start|> user
                Who  are  you ? <|im_end|>
        """
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

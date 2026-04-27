"""Audio utilities for saving and loading audio artifacts."""

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from torch import Tensor

from .constants import InputModality, OutputModality


def tensor_to_audio_bytes(
    waveform: Tensor,
    sample_rate: int = 24_000,
    audio_format: str = "wav",
) -> bytes:
    """
    Convert a waveform tensor to audio bytes.

    Args:
        waveform: Audio waveform tensor.
        sample_rate: Sample rate of the audio.
        audio_format: Output format (wav, mp3, etc.).

    Returns:
        Audio content as bytes.
    """
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio is required for audio serialization") from e

    buffer = io.BytesIO()
    # Ensure correct shape: (channels, samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(buffer, waveform.cpu(), sample_rate, format=audio_format)
    buffer.seek(0)
    return buffer.read()


def save_audio_file(
    audio_bytes: bytes,
    filepath: Path,
    audio_format: str = "wav",
) -> Path:
    """
    Save audio bytes to a file.

    Args:
        audio_bytes: Audio content as bytes.
        filepath: Target file path (extension will be added if missing).
        audio_format: Audio format for extension.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(f".{audio_format}")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(audio_bytes)
    return filepath


def audio_bytes_to_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64 string for JSON serialization."""
    return base64.b64encode(audio_bytes).decode("utf-8")


def base64_to_audio_bytes(b64_string: str) -> bytes:
    """Decode base64 string back to audio bytes."""
    return base64.b64decode(b64_string.encode("utf-8"))


class AudioArtifact:
    """Container for audio artifacts with metadata."""

    def __init__(
        self,
        audio_bytes: bytes,
        sample_rate: int = 24_000,
        audio_format: str = "wav",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.audio_bytes = audio_bytes
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.metadata = metadata or {}

    def save(self, base_path: Path, name: str) -> Dict[str, Any]:
        """
        Save audio artifact to disk.

        Args:
            base_path: Directory to save to.
            name: Base name for the audio file.

        Returns:
            Dictionary with file path and metadata for JSON serialization.
        """
        audio_path = save_audio_file(
            self.audio_bytes,
            base_path / f"{name}.{self.audio_format}",
            self.audio_format,
        )

        return {
            "audio_file": str(audio_path.relative_to(base_path)),
            "sample_rate": self.sample_rate,
            "format": self.audio_format,
            "size_bytes": len(self.audio_bytes),
            "metadata": self.metadata,
        }

    def to_dict(self, include_bytes: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Args:
            include_bytes: If True, include base64-encoded audio bytes.

        Returns:
            Dictionary representation.
        """
        result: Dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "format": self.audio_format,
            "size_bytes": len(self.audio_bytes),
            "metadata": self.metadata,
        }
        if include_bytes:
            result["audio_base64"] = audio_bytes_to_base64(self.audio_bytes)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioArtifact":
        """Reconstruct AudioArtifact from dictionary."""
        audio_bytes = base64_to_audio_bytes(data["audio_base64"])
        return cls(
            audio_bytes=audio_bytes,
            sample_rate=data.get("sample_rate", 24_000),
            audio_format=data.get("format", "wav"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, filepath: Path, sample_rate: int = 24_000) -> "AudioArtifact":
        """Load AudioArtifact from a file."""
        filepath = Path(filepath)
        audio_bytes = filepath.read_bytes()
        audio_format = filepath.suffix.lstrip(".")
        return cls(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            audio_format=audio_format,
            metadata={"source_file": str(filepath)},
        )


def _get_exception_chain(e: Exception) -> str:
    """Get full exception chain as string for debugging."""
    messages = [f"{type(e).__name__}: {str(e)}"]
    cause = e.__cause__
    while cause is not None:
        messages.append(f"  Caused by {type(cause).__name__}: {str(cause)}")
        cause = cause.__cause__
    return " | ".join(messages)


def extract_audio_from_chat(
    chat: Any,
    output_dir: Path,
    prefix: str = "audio",
    sample_rate: int = 24_000,
) -> Dict[str, Any]:
    """
    Extract audio tokens from a chat and save them.

    Args:
        chat: Chat instance with audio tokens.
        output_dir: Directory to save audio files.
        prefix: Prefix for audio file names.
        sample_rate: Sample rate for the audio.

    Returns:
        Dictionary mapping audio identifiers to file info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_info: Dict[str, Any] = {}

    # Try to decode audio_out directly (for LiquidAudioChat)
    # audio_out has shape (K, T) where K=8 codebooks for output
    if hasattr(chat, "audio_out") and chat.audio_out is not None:
        audio_out = chat.audio_out
        if isinstance(audio_out, Tensor) and audio_out.numel() > 0:
            try:
                # audio_out is shape (K, T) where K=8 for AUDIO_OUT_SHAPE
                # decode_audio expects shape (K, T) directly when K matches AUDIO_OUT_SHAPE
                if hasattr(chat, "decode_audio"):
                    audio_bytes = chat.decode_audio(audio_out)
                    if audio_bytes:
                        artifact = AudioArtifact(
                            audio_bytes=audio_bytes,
                            sample_rate=sample_rate,
                            metadata={
                                "source": "chat_audio_out",
                                "shape": list(audio_out.shape),
                            },
                        )
                        audio_info["output_audio"] = artifact.save(
                            output_dir, f"{prefix}_output"
                        )
                    else:
                        audio_info["output_audio_info"] = {
                            "note": "decode_audio returned empty bytes",
                            "audio_out_shape": list(audio_out.shape),
                        }
                else:
                    # Store raw tokens info if we can't decode
                    audio_info["output_audio_tokens"] = {
                        "shape": list(audio_out.shape),
                        "dtype": str(audio_out.dtype),
                    }
            except Exception as e:
                audio_info["output_audio_error"] = _get_exception_chain(e)
                audio_info["output_audio_debug"] = {
                    "audio_out_shape": list(audio_out.shape)
                    if hasattr(audio_out, "shape")
                    else "unknown",
                }

    # Fallback: try audio_tokens if audio_out is not available
    elif hasattr(chat, "audio_tokens") and chat.audio_tokens is not None:
        audio_tokens = chat.audio_tokens
        if isinstance(audio_tokens, Tensor) and audio_tokens.numel() > 0:
            try:
                # audio_tokens is typically _audio_map (indices) for LiquidAudioChat
                # Try to decode using the chat's method
                if hasattr(chat, "decode_audio"):
                    audio_bytes = chat.decode_audio(audio_tokens)
                    if audio_bytes:
                        artifact = AudioArtifact(
                            audio_bytes=audio_bytes,
                            sample_rate=sample_rate,
                            metadata={"source": "chat_audio_tokens"},
                        )
                        audio_info["output_audio"] = artifact.save(
                            output_dir, f"{prefix}_output"
                        )
                else:
                    audio_info["output_audio_tokens"] = {
                        "shape": list(audio_tokens.shape),
                        "dtype": str(audio_tokens.dtype),
                    }
            except Exception as e:
                audio_info["output_audio_error"] = _get_exception_chain(e)

    # Check for audio waveform directly
    if hasattr(chat, "audio_waveform") and chat.audio_waveform is not None:
        waveform = chat.audio_waveform
        if isinstance(waveform, Tensor) and waveform.numel() > 0:
            try:
                audio_bytes = tensor_to_audio_bytes(waveform, sample_rate)
                artifact = AudioArtifact(
                    audio_bytes=audio_bytes,
                    sample_rate=sample_rate,
                    metadata={"source": "chat_waveform"},
                )
                audio_info["output_waveform"] = artifact.save(
                    output_dir, f"{prefix}_waveform"
                )
            except Exception as e:
                audio_info["output_waveform_error"] = _get_exception_chain(e)

    return audio_info


def save_input_audio(
    audio_bytes: bytes | None,
    input_modality: InputModality,
    output_dir: Path,
    prefix: str = "input",
    sample_rate: int = 24_000,
) -> Dict[str, Any]:
    """
    Save input audio and return metadata.

    Args:
        audio_bytes: Input audio bytes.
        input_modality: The input modality used.
        output_dir: Directory to save audio.
        prefix: File name prefix.
        sample_rate: Audio sample rate.

    Returns:
        Dictionary with input audio info.
    """
    result: Dict[str, Any] = {
        "input_modality": input_modality.value,
        "has_audio": audio_bytes is not None,
    }

    if audio_bytes is None:
        return result

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact = AudioArtifact(
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        metadata={"input_modality": input_modality.value},
    )

    saved_info = artifact.save(output_dir, f"{prefix}_audio")
    result.update(saved_info)

    return result


def serialize_result_with_audio(
    result: Any,
    output_dir: Path,
    input_audio_bytes: Optional[bytes | list[bytes]] = None,
    input_modality: InputModality = InputModality.TEXT,
    output_modality: OutputModality = OutputModality.TEXT,
    sample_id: str = "sample",
    sample_rate: int = 24_000,
) -> Dict[str, Any]:
    """
    Serialize explanation result including audio artifacts.

    Args:
        result: ExplainerResult instance.
        output_dir: Directory to save artifacts.
        input_audio_bytes: Original input audio bytes (single or list for multi-turn).
        input_modality: Input modality used.
        output_modality: Output modality used.
        sample_id: Identifier for this sample.
        sample_rate: Audio sample rate.

    Returns:
        Dictionary with serialized result and audio file references.
    """
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    serialized: Dict[str, Any] = {
        "input_modality": input_modality.value,
        "output_modality": output_modality.value,
    }

    # Save input audio if present (handle both single bytes and list)
    # Check if input_audio_bytes is not None/empty (handle numpy arrays properly)
    has_audio = input_audio_bytes is not None and (
        not hasattr(input_audio_bytes, "__len__") or len(input_audio_bytes) > 0
    )
    if has_audio and input_modality in (
        InputModality.AUDIO_MALE,
        InputModality.AUDIO_FEMALE,
    ):
        # Normalize to list
        audio_list = (
            [input_audio_bytes]
            if isinstance(input_audio_bytes, (bytes, np.ndarray))
            else input_audio_bytes
        )
        if len(audio_list) == 1:
            # Single audio - save with original naming
            serialized["input_audio"] = save_input_audio(
                audio_list[0],
                input_modality,
                audio_dir,
                f"{sample_id}_input",
                sample_rate,
            )
        else:
            # Multiple audios - save each with index
            serialized["input_audios"] = [
                save_input_audio(
                    audio,
                    input_modality,
                    audio_dir,
                    f"{sample_id}_input_{i:02d}",
                    sample_rate,
                )
                for i, audio in enumerate(audio_list)
                if audio is not None
            ]

    # Extract and save output audio if output modality is audio
    if output_modality == OutputModality.AUDIO and hasattr(result, "full_chat"):
        serialized["output_audio"] = extract_audio_from_chat(
            result.full_chat,
            audio_dir,
            f"{sample_id}_output",
            sample_rate,
        )

    return serialized

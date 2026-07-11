"""Audio utilities for serializing audio artifacts to MLflow."""

import io
from typing import Any, Dict

import numpy as np
from torch import Tensor

from .constants import InputModality, OutputModality


def tensor_to_audio_bytes(
    waveform: Tensor,
    sample_rate: int = 24_000,
    audio_format: str = "wav",
) -> bytes:
    """Convert a waveform tensor to audio bytes."""
    try:
        import torchaudio
    except ImportError as e:
        raise ImportError("torchaudio is required for audio serialization") from e

    buffer = io.BytesIO()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    torchaudio.save(buffer, waveform.cpu(), sample_rate, format=audio_format)
    buffer.seek(0)
    return buffer.read()


def _get_exception_chain(e: Exception) -> str:
    """Get full exception chain as string for debugging."""
    messages = [f"{type(e).__name__}: {str(e)}"]
    cause = e.__cause__
    while cause is not None:
        messages.append(f"  Caused by {type(cause).__name__}: {str(cause)}")
        cause = cause.__cause__
    return " | ".join(messages)


def serialize_result_with_audio(
    result: Any,
    tracker: Any,
    input_audio_bytes: (bytes | list[bytes]) | None = None,
    input_modality: InputModality = InputModality.TEXT,
    output_modality: OutputModality = OutputModality.TEXT,
    sample_id: str = "sample",
    sample_rate: int = 24_000,
) -> Dict[str, Any]:
    """
    Serialize explanation result including audio artifacts to MLflow.

    Args:
        result: ExplainerResult instance.
        tracker: MlflowTracker for artifact storage.
        input_audio_bytes: Original input audio bytes (single or list for multi-turn).
        input_modality: Input modality used.
        output_modality: Output modality used.
        sample_id: Identifier for this sample.
        sample_rate: Audio sample rate.

    Returns:
        Dictionary with serialized result and audio metadata.
    """
    serialized: Dict[str, Any] = {
        "input_modality": input_modality.value,
        "output_modality": output_modality.value,
    }

    # Save input audio if present (handle both single bytes and list)
    has_audio = input_audio_bytes is not None and (
        not hasattr(input_audio_bytes, "__len__") or len(input_audio_bytes) > 0
    )
    if has_audio and input_modality in (
        InputModality.AUDIO_ORIGINAL,
        InputModality.AUDIO_MALE,
        InputModality.AUDIO_FEMALE,
    ):
        if isinstance(input_audio_bytes, bytes):
            audio_list = [input_audio_bytes]
        elif isinstance(input_audio_bytes, np.ndarray):
            audio_list = input_audio_bytes.tolist()
        else:
            audio_list = list(input_audio_bytes)
        if len(audio_list) == 1:
            tracker.log_bytes_artifact(
                audio_list[0],
                f"{sample_id}_input.wav",
                artifact_path=f"audio/{sample_id}",
            )
            serialized["input_audio"] = {
                "input_modality": input_modality.value,
                "has_audio": True,
                "size_bytes": len(audio_list[0]),
                "sample_rate": sample_rate,
            }
        else:
            input_infos = []
            for i, audio in enumerate(audio_list):
                if audio is not None:
                    tracker.log_bytes_artifact(
                        audio,
                        f"{sample_id}_input_{i:02d}.wav",
                        artifact_path=f"audio/{sample_id}",
                    )
                    input_infos.append({
                        "input_modality": input_modality.value,
                        "has_audio": True,
                        "size_bytes": len(audio),
                        "sample_rate": sample_rate,
                    })
            serialized["input_audios"] = input_infos

    # Extract and log output audio if output modality is audio
    if output_modality == OutputModality.AUDIO and hasattr(result, "full_chat"):
        chat = result.full_chat
        output_audio_info = _extract_output_audio_to_tracker(
            chat, tracker, sample_id, sample_rate
        )
        if output_audio_info:
            serialized["output_audio"] = output_audio_info

    return serialized


def _extract_output_audio_to_tracker(
    chat: Any, tracker: Any, sample_id: str, sample_rate: int
) -> Dict[str, Any]:
    """Extract audio from chat and log to MLflow tracker."""
    audio_info: Dict[str, Any] = {}

    if hasattr(chat, "audio_out") and chat.audio_out is not None:
        audio_out = chat.audio_out
        if isinstance(audio_out, Tensor) and audio_out.numel() > 0:
            try:
                if hasattr(chat, "decode_audio"):
                    audio_bytes = chat.decode_audio(audio_out)
                    if audio_bytes:
                        tracker.log_bytes_artifact(
                            audio_bytes,
                            f"{sample_id}_output.wav",
                            artifact_path=f"audio/{sample_id}",
                        )
                        audio_info["output_audio"] = {
                            "size_bytes": len(audio_bytes),
                            "sample_rate": sample_rate,
                            "source": "chat_audio_out",
                            "shape": list(audio_out.shape),
                        }
                    else:
                        audio_info["output_audio_info"] = {
                            "note": "decode_audio returned empty bytes",
                            "audio_out_shape": list(audio_out.shape),
                        }
                else:
                    audio_info["output_audio_tokens"] = {
                        "shape": list(audio_out.shape),
                        "dtype": str(audio_out.dtype),
                    }
            except Exception as e:
                audio_info["output_audio_error"] = _get_exception_chain(e)

    elif hasattr(chat, "audio_tokens") and chat.audio_tokens is not None:
        audio_tokens = chat.audio_tokens
        if isinstance(audio_tokens, Tensor) and audio_tokens.numel() > 0:
            try:
                if hasattr(chat, "decode_audio"):
                    audio_bytes = chat.decode_audio(audio_tokens)
                    if audio_bytes:
                        tracker.log_bytes_artifact(
                            audio_bytes,
                            f"{sample_id}_output.wav",
                            artifact_path=f"audio/{sample_id}",
                        )
                        audio_info["output_audio"] = {
                            "size_bytes": len(audio_bytes),
                            "sample_rate": sample_rate,
                            "source": "chat_audio_tokens",
                        }
                else:
                    audio_info["output_audio_tokens"] = {
                        "shape": list(audio_tokens.shape),
                        "dtype": str(audio_tokens.dtype),
                    }
            except Exception as e:
                audio_info["output_audio_error"] = _get_exception_chain(e)

    if hasattr(chat, "audio_waveform") and chat.audio_waveform is not None:
        waveform = chat.audio_waveform
        if isinstance(waveform, Tensor) and waveform.numel() > 0:
            try:
                audio_bytes = tensor_to_audio_bytes(waveform, sample_rate)
                tracker.log_bytes_artifact(
                    audio_bytes,
                    f"{sample_id}_waveform.wav",
                    artifact_path=f"audio/{sample_id}",
                )
                audio_info["output_waveform"] = {
                    "size_bytes": len(audio_bytes),
                    "sample_rate": sample_rate,
                    "source": "chat_waveform",
                }
            except Exception as e:
                audio_info["output_waveform_error"] = _get_exception_chain(e)

    return audio_info

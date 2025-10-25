"""Helper functions for timing SHAP explanations on LiquidAudio model."""

import os
import gc
import time
import traceback
from typing import Any, Dict, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
from mllm_shap.connectors import LiquidAudio
from mllm_shap.connectors.enums import (
    ModalityFlag,
    ModelHistoryTrackingMode,
    Role,
    SystemRolesSetup,
)
from mllm_shap.connectors.filters import KeepAllTokens
from mllm_shap.shap import Explainer, McShapExplainer
from mllm_shap.shap.enums import Mode


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Column / key names used in dataset rows
SENTENCES_COL = "sentences"
PROMPT_COL = "prompt"
CONVERSATION_COL = "conversation"
AUDIO_FEMALE_COL = "audio__female"
AUDIO_MALE_COL = "audio__male"
TEXT_TOKENS_COL = "text_tokens"
AUDIO_TOKENS_COL = "audio_tokens"
SOURCE_FILE_COL = "source_file"

# Known parquet sources
SINGLE_SENTENCE_FILE = "single_sentence.parquet"
MULTI_SENTENCE_FILE = "multi_sentence.parquet"
MULTI_LINGUAL_FILE = "multi_lingual.parquet"
MULTI_TURN_FILE = "multi_turn.parquet"

CUDA_DEVICE_TYPE = "cuda"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
T = TypeVar("T")


def _ensure_not_none(val: Optional[T], name: str) -> T:
    """Runtime + static check that ``val`` is not ``None``.

    This helps mypy understand that objects created in a ``try:`` block are
    definitely available afterwards, and also guards at runtime in case that
    assumption ever breaks.
    """
    if val is None:
        raise RuntimeError(f"{name} unexpectedly None")
    return val


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def load_sample_from_parquet(data_dir: str, filename: str = "single_sentence.parquet") -> Tuple[str, bytes, str]:
    """Load a random row from the parquet file and extract text + audio bytes.

    Args:
        data_dir: Base directory containing parquet files.
        filename: Parquet filename to load.

    Returns:
        (sample_text, sample_audio_bytes, audio_col_used)

    Raises:
        FileNotFoundError: If the parquet file doesn't exist.
        RuntimeError: If loading or sampling fails.
        ValueError: If required text/audio fields are missing or invalid.
    """
    sample_file_path = os.path.join(data_dir, filename)

    if not os.path.exists(sample_file_path):
        raise FileNotFoundError(f"Sample file not found: {sample_file_path}. " "Please check the DATA_DIR path.")

    print(f"Loading sample data from: {sample_file_path}")

    try:
        df = pd.read_parquet(sample_file_path, engine="pyarrow")
        print(f"Loaded {len(df)} rows.")

        sample_entry = df.sample(1, random_state=42).iloc[0].to_dict()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise RuntimeError("Failed to load or sample parquet file " f"{sample_file_path}: {exc}") from exc

    # Extract text
    sentences_list = sample_entry.get("sentences")
    if not sentences_list or not isinstance(sentences_list[0], str) or len(sentences_list[0].strip()) == 0:
        raise ValueError("Sample row does not contain valid text in the 'sentences' column.")
    sample_text = sentences_list[0]

    # Extract audio bytes
    sample_audio_bytes = None
    audio_col_used = None

    female_audio_list = sample_entry.get("audio__female")
    if female_audio_list and len(female_audio_list) > 0 and isinstance(female_audio_list[0], bytes):
        sample_audio_bytes = female_audio_list[0]
        audio_col_used = "audio__female"
    else:
        male_audio_list = sample_entry.get("audio__male")
        if male_audio_list and len(male_audio_list) > 0 and isinstance(male_audio_list[0], bytes):
            sample_audio_bytes = male_audio_list[0]
            audio_col_used = "audio__male"

    if sample_audio_bytes is None or audio_col_used is None:
        raise ValueError("Sample row does not contain valid audio bytes in " + "'audio__female' or 'audio__male'.")

    # Clean up DataFrame ASAP to free memory/GPU VRAM pressure
    del df
    gc.collect()

    return sample_text, sample_audio_bytes, audio_col_used


def run_test_on_device(
    target_device: torch.device,
    text_data: str,
    audio_data_bytes: bytes,
    num_runs: int = 3,
) -> Tuple[float, float, int, int]:
    """Loads model onto ``target_device``, runs SHAP, returns timing.

    Returns:
        (avg_time_seconds, std_time_seconds, total_text_tokens, total_audio_tokens)
    """
    # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    timings = []
    total_text_tokens = 0
    total_audio_tokens = 0

    print(f"\n--- Starting Test on {target_device} ---")

    # Load model onto target device
    print(f"Loading model onto {target_device}...")
    load_start = time.time()

    chat: Optional[Any] = None
    temp_explainer: Optional[Explainer] = None
    temp_model: Optional[LiquidAudio] = None
    temp_shap_explainer: Optional[McShapExplainer] = None

    try:
        # Load a fresh instance of the model directly onto the target device
        temp_model = LiquidAudio(
            device=target_device,
            history_tracking_mode=ModelHistoryTrackingMode.TEXT_AUDIO,
        )
        # Instantiate explainer with the temp model
        temp_shap_explainer = McShapExplainer(
            num_samples=-1,
            mode=Mode.CONTEXTUAL,
        )
        temp_explainer = Explainer(
            model=temp_model,
            shap_explainer=temp_shap_explainer,
        )
        load_end = time.time()
        print(f"Model loaded on {target_device} in {load_end - load_start:.2f}s.")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"FAILED to load model on {target_device}: {exc}")

        # Clean up if model loading failed partially
        if temp_explainer is not None:
            del temp_explainer
        if temp_shap_explainer is not None:
            del temp_shap_explainer
        if temp_model is not None:
            del temp_model
        gc.collect()
        if target_device.type == CUDA_DEVICE_TYPE:
            torch.cuda.empty_cache()
        return float("nan"), float("nan"), 0, 0

    # Tell mypy and runtime that these are non-None now
    temp_explainer = _ensure_not_none(temp_explainer, "temp_explainer")
    temp_model = _ensure_not_none(temp_model, "temp_model")
    temp_shap_explainer = _ensure_not_none(
        temp_shap_explainer,
        "temp_shap_explainer",
    )

    # Run Explainer
    for i in range(num_runs):
        print(f"  Starting Run {i + 1}/{num_runs} on {target_device}...")

        # Create a new chat for each run
        if not isinstance(temp_explainer.model, LiquidAudio):
            raise TypeError(
                "Explainer model must be LiquidAudio, " f"got {type(temp_explainer.model).__name__}",
            )

        chat = temp_explainer.model.get_new_chat(
            system_roles_setup=SystemRolesSetup.SYSTEM,
            token_filter=KeepAllTokens(),
        )
        chat.new_turn(Role.USER)
        chat.add_text(text_data)
        try:
            # Add audio
            chat.add_audio(audio_data_bytes)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"    Error adding audio in run {i + 1}: {exc}")

            # Clean up and return NaN if audio fails
            if chat is not None:
                del chat
            if temp_explainer is not None:
                del temp_explainer
            if temp_shap_explainer is not None:
                del temp_shap_explainer
            if temp_model is not None:
                del temp_model
            gc.collect()
            if target_device.type == CUDA_DEVICE_TYPE:
                torch.cuda.empty_cache()
            return float("nan"), float("nan"), 0, 0
        chat.end_turn()

        start_run_time = time.time()
        try:
            # Run explainer
            result = temp_explainer(
                chat=chat,
                generation_kwargs={"max_new_tokens": 16},
                progress_bar=False,
                verbose=False,
            )

            # Get token counts (only on the first successful run)
            if (
                i == 0
                and result
                and result.full_chat
                and hasattr(result.full_chat, "shap_values_mask")
                and result.full_chat.shap_values_mask is not None
            ):
                chat_device = result.full_chat.torch_device
                shap_mask = result.full_chat.shap_values_mask.bool().to(chat_device)
                modality_flags = result.full_chat.tokens_modality_flag.to(chat_device)

                # Cast .item() to int to satisfy mypy
                total_text_tokens = int((shap_mask & (modality_flags == ModalityFlag.TEXT)).sum().item())
                total_audio_tokens = int((shap_mask & (modality_flags == ModalityFlag.AUDIO)).sum().item())
            elif i == 0:
                print(
                    "    Warning: Could not retrieve token counts on first run.",
                )

            end_run_time = time.time()
            run_time = end_run_time - start_run_time
            timings.append(run_time)
            print(
                f"  Run {i + 1}/{num_runs} completed in {run_time:.2f}s "
                f"(Text Tokens: {total_text_tokens}, "
                f"Audio Tokens: {total_audio_tokens})",
            )

        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"  Run {i + 1}/{num_runs} FAILED: {exc}")
            traceback.print_exc()  # pylint: disable=import-outside-toplevel
            timings = [float("nan")]

            if chat is not None:
                del chat
            if temp_explainer is not None:
                del temp_explainer
            if temp_shap_explainer is not None:
                del temp_shap_explainer
            if temp_model is not None:
                del temp_model
            gc.collect()
            if target_device.type == CUDA_DEVICE_TYPE:
                torch.cuda.empty_cache()
            break

    avg_time = np.nanmean(timings)
    std_dev_time = np.nanstd(timings)

    # Clean up the temporary model and explainer
    if chat is not None:
        del chat
    if temp_explainer is not None:
        del temp_explainer
    if temp_shap_explainer is not None:
        del temp_shap_explainer
    if temp_model is not None:
        del temp_model
    gc.collect()
    if target_device.type == CUDA_DEVICE_TYPE:
        torch.cuda.empty_cache()

    print(
        f"Test on {target_device} complete. " f"Avg time: {avg_time:.2f}s (±{std_dev_time:.2f}s)",
    )
    # Cast numpy.floating to float for return type
    return (
        float(avg_time),
        float(std_dev_time),
        total_text_tokens,
        total_audio_tokens,
    )


def compare_and_prepare(
    primary_device: torch.device,
    txt: str,
    audio: bytes,
    shap_explainer_obj: McShapExplainer,
) -> Tuple[
    Dict[str, float],
    Optional[LiquidAudio],
    Optional[Explainer],
]:
    """Compare CPU vs GPU timing and reload model+explainer for downstream use.

    Steps:
    - Run timing on CPU (and GPU if available).
    - Print out feasibility / speedup summary.
    - Reload a fresh model instance on the chosen device
      and wrap it in an Explainer with the provided SHAP explainer.

    Returns:
        (timing_results_local, model_obj, timing_explainer_obj)
        where:
        - timing_results_local: {"CPU": cpu_avg_time, "GPU": gpu_avg_time}
        - model_obj: LiquidAudio or None if reload failed
        - timing_explainer_obj: Explainer or None if reload failed
    """
    # pylint: disable=too-many-locals
    # CPU / GPU result placeholders
    cpu_avg, cpu_std = float("nan"), float("nan")
    cpu_text_tokens, cpu_audio_tokens = 0, 0

    gpu_avg, gpu_std = float("nan"), float("nan")
    gpu_text_tokens, gpu_audio_tokens = 0, 0

    # Run on CPU (or fallback to CPU if no GPU)
    if primary_device.type == CUDA_DEVICE_TYPE:
        cpu_device = torch.device("cpu")
        (
            cpu_avg,
            cpu_std,
            cpu_text_tokens,
            cpu_audio_tokens,
        ) = run_test_on_device(cpu_device, txt, audio)
    else:
        print("GPU not available. Running test only on CPU.")
        (
            cpu_avg,
            cpu_std,
            cpu_text_tokens,
            cpu_audio_tokens,
        ) = run_test_on_device(primary_device, txt, audio)
        gpu_avg, gpu_std = cpu_avg, cpu_std
        gpu_text_tokens, gpu_audio_tokens = (
            cpu_text_tokens,
            cpu_audio_tokens,
        )

    # Run on GPU (if primary device is GPU)
    if primary_device.type == CUDA_DEVICE_TYPE:
        (
            gpu_avg,
            gpu_std,
            gpu_text_tokens,
            gpu_audio_tokens,
        ) = run_test_on_device(primary_device, txt, audio)

    # Comparison / reporting
    print("\n--- CPU vs GPU Feasibility Results ---")
    chosen_text_token_count = gpu_text_tokens if not np.isnan(gpu_avg) and gpu_text_tokens > 0 else cpu_text_tokens
    chosen_audio_token_count = gpu_audio_tokens if not np.isnan(gpu_avg) and gpu_audio_tokens > 0 else cpu_audio_tokens

    print(f"Sample Text Tokens Explained: {chosen_text_token_count}")
    print(f"Sample Audio Tokens Explained: {chosen_audio_token_count}")

    if primary_device.type == CUDA_DEVICE_TYPE:
        print(f"Average time on CPU: {cpu_avg:.2f} ± {cpu_std:.2f} seconds")
        print(f"Average time on GPU: {gpu_avg:.2f} ± {gpu_std:.2f} seconds")
        if not np.isnan(cpu_avg) and not np.isnan(gpu_avg) and cpu_avg > 0 and gpu_avg > 0:
            speedup = cpu_avg / gpu_avg
            print(f"GPU Speedup: {speedup:.2f}x")
        else:
            print("Could not calculate speedup (CPU run may have failed or times were zero/NaN).")
        print("\nConclusion: Based on performance, further analysis should use the GPU.")
    else:
        print(f"Average time on CPU: {cpu_avg:.2f} ± {cpu_std:.2f} seconds")
        print("\nConclusion: GPU not available. Analysis proceeding on CPU.")

    # Store basic results for plotting
    timing_results_local: Dict[str, float] = {
        "CPU": cpu_avg,
        "GPU": gpu_avg,
    }

    print("\nReloading primary model instance for subsequent cells...")

    # Predeclare so mypy knows they're always defined
    model_obj: Optional[LiquidAudio] = None
    timing_explainer_obj: Optional[Explainer] = None

    try:
        # Reload the main model instance onto the primary device
        model_obj = LiquidAudio(
            device=primary_device,
            history_tracking_mode=ModelHistoryTrackingMode.TEXT_AUDIO,
        )
        timing_explainer_obj = Explainer(
            model=model_obj,
            shap_explainer=shap_explainer_obj,
        )
        print(f"Primary model reloaded successfully on {primary_device}.")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Error reloading primary model: {exc}. " "You may need to rerun Cell 2.")
        model_obj = None
        timing_explainer_obj = None

    return timing_results_local, model_obj, timing_explainer_obj


def count_tokens_for_row(
    row: pd.Series,
    model_instance: LiquidAudio,
) -> Tuple[int, int, Optional[str]]:
    """
    Creates a chat object, adds text/audio, and counts tokens by checking
    the change in the chat object's internal token storage dimensions.
    This avoids calling get_conversation().

    Args:
        row: A pandas Series representing a row from the dataset.
        model_instance: The loaded LiquidAudio model instance.

    Returns:
        (text_token_count, audio_token_count, error_message_or_None).
    """
    # pylint: disable=too-many-branches
    text_tokens_added = 0
    audio_tokens_added = 0
    error_msg: Optional[str] = None
    chat: Optional[Any] = None

    try:
        # Extract text
        row_text: Optional[str] = None

        if (
            SENTENCES_COL in row
            and len(row[SENTENCES_COL]) > 0
            and isinstance(row[SENTENCES_COL][0], str)
            and len(row[SENTENCES_COL][0].strip()) > 0
        ):
            row_text = row[SENTENCES_COL][0]

        elif PROMPT_COL in row and isinstance(row.get(PROMPT_COL), str) and len(row[PROMPT_COL].strip()) > 0:
            row_text = row[PROMPT_COL]

        elif CONVERSATION_COL in row:
            row_text = " ".join(
                turn.get("content", "") for turn in row[CONVERSATION_COL] if isinstance(turn.get("content"), str)
            )
            if len(row_text.strip()) == 0:
                row_text = None

        if not row_text:
            missing_text_err = "Missing or invalid text data " + "(checked 'sentences'[0], 'prompt', 'conversation')"
            return 0, 0, missing_text_err

        # Extract audio
        row_audio_bytes: Optional[bytes] = None
        if AUDIO_FEMALE_COL in row and len(row[AUDIO_FEMALE_COL]) > 0 and isinstance(row[AUDIO_FEMALE_COL][0], bytes):
            row_audio_bytes = row[AUDIO_FEMALE_COL][0]
        elif AUDIO_MALE_COL in row and len(row[AUDIO_MALE_COL]) > 0 and isinstance(row[AUDIO_MALE_COL][0], bytes):
            row_audio_bytes = row[AUDIO_MALE_COL][0]

        # Create Chat Object
        chat = model_instance.get_new_chat(
            system_roles_setup=SystemRolesSetup.NONE,
            token_filter=None,
        )

        # Record initial token counts
        initial_text_tokens = chat.text.shape[1] if hasattr(chat, "text") and chat.text is not None else 0
        initial_audio_in_tokens = (
            chat.audio_in.shape[1] if hasattr(chat, "audio_in") and chat.audio_in is not None else 0
        )

        # Add content within a single turn context for simplicity
        chat.new_turn(Role.USER)
        chat.add_text(row_text)
        if row_audio_bytes:
            chat.add_audio(row_audio_bytes, audio_format="mp3")  # Assuming MP3
        chat.end_turn()

        # Final token count
        final_text_tokens = chat.text.shape[1] if hasattr(chat, "text") and chat.text is not None else 0
        final_audio_in_tokens = chat.audio_in.shape[1] if hasattr(chat, "audio_in") and chat.audio_in is not None else 0

        text_tokens_added = final_text_tokens - initial_text_tokens
        audio_tokens_added = final_audio_in_tokens - initial_audio_in_tokens

    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_msg = f"Error processing row: {type(exc).__name__} - {str(exc)}"
        text_tokens_added = 0
        audio_tokens_added = 0
    finally:
        # Explicitly delete the chat object
        if chat is not None:
            del chat
            # gc.collect()  # optional

    # Basic sanity check: counts should not be negative
    text_tokens_added = max(text_tokens_added, 0)
    audio_tokens_added = max(audio_tokens_added, 0)

    return text_tokens_added, audio_tokens_added, error_msg


def time_shap_on_gpu(
    row: pd.Series,
    explainer: Explainer,
) -> Dict[str, Any]:
    """
    Runs SHAP explanation (MC, n=-1) for a single row ON THE GPU.
    Extracts text/audio from the row based on ``source_file``.
    Returns timing, token counts, and any error.
    """
    # pylint: disable=too-many-locals,too-many-branches
    start_time = time.time()

    text_tokens = row.get(TEXT_TOKENS_COL, 0)  # Use pre-calculated tokens
    audio_tokens = row.get(AUDIO_TOKENS_COL, 0)  # Use pre-calculated tokens
    source_file = row.get(SOURCE_FILE_COL, "unknown")

    error_msg: Optional[str] = None
    row_text: Optional[str] = None
    row_audio_bytes: Optional[bytes] = None
    chat: Optional[Any] = None  # Initialize for cleanup

    # Basic check: only time rows with tokens
    if text_tokens == 0 and audio_tokens == 0:
        return {
            "time_seconds": 0.0,
            "text_tokens": 0,
            "audio_tokens": 0,
            "error": "Skipped (zero tokens)",
        }

    try:
        # Extract data
        if source_file in (SINGLE_SENTENCE_FILE, MULTI_SENTENCE_FILE):
            sentences_list = row.get(SENTENCES_COL)
            if sentences_list and len(sentences_list) > 0 and isinstance(sentences_list[0], str):
                row_text = sentences_list[0]

            audio_female_list = row.get(AUDIO_FEMALE_COL)
            if audio_female_list and len(audio_female_list) > 0 and isinstance(audio_female_list[0], bytes):
                row_audio_bytes = audio_female_list[0]
            else:
                audio_male_list = row.get(AUDIO_MALE_COL)
                if audio_male_list and len(audio_male_list) > 0 and isinstance(audio_male_list[0], bytes):
                    row_audio_bytes = audio_male_list[0]

        elif source_file == MULTI_LINGUAL_FILE:
            raise ValueError(
                f"Timing logic for {source_file} not fully implemented yet",
            )

        elif source_file == MULTI_TURN_FILE:
            raise ValueError(
                f"Timing logic for {source_file} not fully implemented yet",
            )

        else:
            raise ValueError(f"Unknown source_file: {source_file}")

        # Ensure we have text for timing
        if row_text is None:
            raise ValueError("Could not extract valid text for timing")

        if not isinstance(explainer.model, LiquidAudio):
            raise TypeError(
                "Explainer model must be LiquidAudio, " f"got {type(explainer.model).__name__}",
            )

        chat = explainer.model.get_new_chat(
            system_roles_setup=SystemRolesSetup.SYSTEM,
            token_filter=KeepAllTokens(),
        )
        chat.new_turn(Role.USER)

        # mypy is happy: row_text is guaranteed to be str by this point
        chat.add_text(row_text)

        if row_audio_bytes:
            chat.add_audio(row_audio_bytes)
        chat.end_turn()

        _ = explainer(
            chat=chat,
            generation_kwargs={"max_new_tokens": 8},
            progress_bar=False,
            verbose=False,
        )

    except Exception as exc:  # pylint: disable=broad-exception-caught
        error_msg = f"Error during SHAP timing: {type(exc).__name__} - {str(exc)}"

    finally:
        if chat is not None:
            del chat

    end_time = time.time()
    elapsed_time = end_time - start_time if error_msg is None else float("nan")

    return {
        "time_seconds": elapsed_time,
        "text_tokens": text_tokens,  # Return pre-calculated tokens
        "audio_tokens": audio_tokens,  # Return pre-calculated tokens
        "error": error_msg,
    }

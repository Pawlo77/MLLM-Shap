"""Common utilities for analyzing experiment outputs in `experiments/experiments_output/`.

This module is intentionally lightweight and notebook-friendly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from mllm_shap.connectors.enums import Role  # pylint: disable=wrong-import-order

# Named constants (avoid magic values)
RESULT_FILE_SUBSTRING: str = "result"
MIN_CONVERSATION_TURNS: int = 2
TEXT_CONTENT_TYPE: int = 0
AUDIO_SEGMENTS_FIELD: str = "audio_segments"
DEFAULT_AUDIO_SEGMENT_KEY: str = "2"
LANGUAGE_COL: str = "language"
ORIGINAL_LANGUAGE_COL: str = "original_language"

try:
    from scipy.stats import entropy as SCIPY_ENTROPY
except Exception:  # pylint: disable=broad-exception-caught
    SCIPY_ENTROPY = None


EXPERIMENTS_OUTPUT_DIR: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments_output")
)


@dataclass(frozen=True)
class LoadedCase:
    """A loaded run/case pair."""

    run_name: str
    case_dir: str
    df: pd.DataFrame


def _list_sample_files(samples_dir: str) -> list[str]:
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"Missing samples dir: {samples_dir}")

    files = [
        os.path.join(samples_dir, f)
        for f in os.listdir(samples_dir)
        if f.endswith(".json") and RESULT_FILE_SUBSTRING in f
    ]
    if not files:
        raise FileNotFoundError(f"No sample result json files found in: {samples_dir}")

    return sorted(files)


def load_case_results(run_name: str, case_dir: str) -> LoadedCase:
    """Load all `sample_*_result.json` files for a given run/case."""

    case_path = os.path.join(EXPERIMENTS_OUTPUT_DIR, run_name, case_dir)
    samples_dir = os.path.join(case_path, "samples")

    dts: list[dict[str, Any]] = []
    for file_path in _list_sample_files(samples_dir):
        with open(file_path, "r", encoding="utf-8") as fh:
            dts.append(json.load(fh))

    df = pd.DataFrame(dts)
    _validate_minimal_schema(df, run_name=run_name, case_dir=case_dir)

    return LoadedCase(run_name=run_name, case_dir=case_dir, df=df)


def _validate_minimal_schema(df: pd.DataFrame, *, run_name: str, case_dir: str) -> None:
    required_top = {
        "row_index",
        "runtime_sec",
        "n_calls",
        "neyman_steps",
        "prompt_texts",
        "input_modality",
        "output_modality",
        "attr_summary",
        "conversation",
    }
    missing = required_top - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required fields in results for {run_name}/{case_dir}: {sorted(missing)}"
        )

    if df["row_index"].nunique() != len(df):
        raise RuntimeError(f"row_index is not unique in {run_name}/{case_dir}")

    # Basic conversation integrity check
    if not df["conversation"].apply(
        lambda x: isinstance(x, list) and len(x) >= MIN_CONVERSATION_TURNS
    ).all():
        raise RuntimeError(f"Invalid conversation format in {run_name}/{case_dir}")


def check_roles_against_sv(
    roles: np.ndarray,
    shap_values: np.ndarray,
    *,
    expected_role: Role = Role.USER,
) -> bool:
    """Any non-NaN SHAP values should correspond to `expected_role`."""

    if len(roles) != len(shap_values):
        return False

    mask = ~pd.isna(shap_values)
    return (roles[mask] == expected_role).all()


def iter_text_messages(conversation: list[list[dict[str, Any]]]) -> Iterable[tuple[int, int, dict[str, Any]]]:
    """Yield (turn_index, msg_index, msg) for text messages."""
    for turn_index, turn in enumerate(conversation):
        for msg_index, msg in enumerate(turn):
            if msg.get("content_type") == TEXT_CONTENT_TYPE:
                yield turn_index, msg_index, msg


def ensure_language_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure language columns exist (nullable) and return a copy."""

    out = df.copy()
    if LANGUAGE_COL not in out.columns:
        out[LANGUAGE_COL] = None
    if ORIGINAL_LANGUAGE_COL not in out.columns:
        out[ORIGINAL_LANGUAGE_COL] = None
    return out


def extract_text_units(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract explainable text token units from a sample.

    Returns a list of units at "message" granularity (one entry per text message
    that contains any non-null shap value).
    """

    conversation = sample["conversation"]
    units: list[dict[str, Any]] = []

    for turn_index, msg_index, msg in iter_text_messages(conversation):
        sv = np.asarray(msg.get("shap_values", []), dtype=object)
        if sv.size == 0:
            continue

        mask = ~pd.isna(sv)
        if not mask.any():
            continue

        roles = np.asarray(msg.get("roles", []))
        if roles.size != sv.size:
            raise RuntimeError(
                (
                    "roles/shap_values length mismatch "
                    f"(row_index={sample.get('row_index')}, turn={turn_index}, msg={msg_index})"
                )
            )
        if not check_roles_against_sv(roles, sv):
            raise RuntimeError(
                (
                    "Non-null shap_values not aligned to USER roles "
                    f"(row_index={sample.get('row_index')}, turn={turn_index}, msg={msg_index})"
                )
            )

        content = np.asarray(msg.get("content", []), dtype=object)
        if content.size != sv.size:
            raise RuntimeError(
                (
                    "content/shap_values length mismatch "
                    f"(row_index={sample.get('row_index')}, turn={turn_index}, msg={msg_index})"
                )
            )

        tokens = content[mask].astype(str).tolist()
        sv_f = sv[mask].astype(float)

        units.append(
            {
                "row_index": sample.get("row_index"),
                "turn_index": turn_index,
                "msg_index": msg_index,
                "unit_type": "text",
                "tokens": tokens,
                "sv": sv_f,
                "inputs": "".join(tokens),
                "count_explainable_units": int(len(tokens)),
            }
        )

    return units


def _pick_audio_segment_key(audio_segments: dict[str, Any]) -> str:
    if DEFAULT_AUDIO_SEGMENT_KEY in audio_segments:
        return DEFAULT_AUDIO_SEGMENT_KEY

    # Some runs may store a different key; pick the first deterministic key.
    keys = sorted(audio_segments.keys(), key=str)
    if not keys:
        raise RuntimeError("audio_segments is empty")
    return keys[0]


def extract_audio_units(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract explainable audio segment units from a sample."""

    if AUDIO_SEGMENTS_FIELD not in sample:
        return []

    audio_segments = sample[AUDIO_SEGMENTS_FIELD]
    if not isinstance(audio_segments, dict):
        raise RuntimeError(f"audio_segments has unexpected type: {type(audio_segments)}")

    key = _pick_audio_segment_key(audio_segments)
    segments = audio_segments.get(key)
    if segments is None:
        return []

    tokens = [seg["token"] for seg in segments]
    sv = np.asarray([seg["shap_value"] for seg in segments], dtype=float)

    return [
        {
            "row_index": sample.get("row_index"),
            "turn_index": None,
            "msg_index": None,
            "unit_type": "audio",
            "tokens": tokens,
            "sv": sv,
            "inputs": " ".join(tokens),
            "count_explainable_units": int(len(tokens)),
        }
    ]


def build_units_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw results into a long DataFrame of explainable units.

    Output columns are stable across datasets:
    - row_index, unit_type, tokens, sv, inputs, count_explainable_units
    - plus any passthrough metadata copied from the source results.
    """

    rows: list[dict[str, Any]] = []

    for sample in results_df.to_dict(orient="records"):
        meta = {
            "runtime_sec": sample.get("runtime_sec"),
            "n_calls": sample.get("n_calls"),
            "neyman_steps": sample.get("neyman_steps"),
            "input_modality": sample.get("input_modality"),
            "output_modality": sample.get("output_modality"),
            "prompt_texts": sample.get("prompt_texts"),
            "language": sample.get("language"),
            "original_language": sample.get("original_language"),
            "attr_summary": sample.get("attr_summary"),
        }

        for unit in extract_text_units(sample):
            unit.update(meta)
            rows.append(unit)

        for unit in extract_audio_units(sample):
            unit.update(meta)
            rows.append(unit)

    if not rows:
        raise RuntimeError("No explainable units extracted (text or audio)")

    return pd.DataFrame(rows)


def shap_concentration_metrics(sv: np.ndarray) -> dict[str, float]:
    """Compute simple, robust SV concentration metrics."""

    sv = np.asarray(sv, dtype=float)
    abs_sv = np.abs(sv)
    total = float(abs_sv.sum())

    if total <= 0:
        return {
            "abs_sum": 0.0,
            "abs_max": 0.0,
            "top1_frac": 0.0,
            "top5_frac": 0.0,
            "effective_tokens": 0.0,
        }

    sorted_abs = np.sort(abs_sv)[::-1]
    top1 = float(sorted_abs[0])
    top5 = float(sorted_abs[: min(5, len(sorted_abs))].sum())

    p = abs_sv / total
    effective = float(1.0 / np.sum(p**2))  # Simpson / inverse participation

    return {
        "abs_sum": total,
        "abs_max": float(abs_sv.max()),
        "top1_frac": top1 / total,
        "top5_frac": top5 / total,
        "effective_tokens": effective,
    }


def shapley_entropy(sv: np.ndarray, *, base: float = 2.0) -> float:
    """Entropy of the normalized |SV| distribution.

    Higher values indicate more diffuse attribution; lower values indicate
    concentration on fewer units.
    """

    if SCIPY_ENTROPY is None:
        raise RuntimeError(
            "SciPy is required for shapley_entropy. Install it with: uv add scipy"
        )

    sv = np.asarray(sv, dtype=float)
    if sv.size == 0:
        return 0.0

    abs_sv = np.abs(sv)
    total = float(abs_sv.sum())
    if total <= 0:
        return 0.0

    p = abs_sv / total
    return float(SCIPY_ENTROPY(p, base=base))


def attention_density(sv: np.ndarray) -> float:
    """Mean |SV| per explainable unit (token/segment)."""

    sv = np.asarray(sv, dtype=float)
    if sv.size == 0:
        return 0.0
    return float(np.abs(sv).sum() / max(1, sv.size))


def history_retention_percent(message_abs_sums: Iterable[float]) -> float:
    """Percent of |SV| mass attributed to history vs current.

    Assumes `message_abs_sums` are ordered chronologically and the last element
    corresponds to the current (latest) user message.
    """

    vals = np.asarray(list(message_abs_sums), dtype=float)
    if vals.size == 0:
        return 0.0

    total = float(np.nansum(vals))
    if total <= 0:
        return 0.0

    history = float(np.nansum(vals[:-1]))
    return 100.0 * history / total


def history_retention_score(message_abs_sums: Iterable[float]) -> float:
    """History retention as a 0..1 fraction."""

    return history_retention_percent(message_abs_sums) / 100.0

"""Alpha-grid search helpers for SGPA boundary refinement."""

import csv
import gc
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
import resource
import shutil
import subprocess
import tkinter as tk
from tkinter import messagebox
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from mllm_shap.connectors.base.audio import SpectrogramGuidedAligner
from mllm_shap.utils.audio import TorchAudioHandler
from tqdm.auto import tqdm

EPS: float = 1e-9


@dataclass(frozen=True)
class AlphaSearchConfig:
    """Configuration for alpha-grid search."""

    alphas: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
    n_held_out: int = 20
    masks_per_utterance: int = 3
    rng_seed: int = 77
    dataset_name: str = "Pawlo77/mllm-swap"
    dataset_config: str = "single_sentence"
    dataset_revision: str = "3a8e7fbe8da0b3caaf865978e92c86ee670bda65"
    output_dir: Path = Path("outputs/alpha_search")
    manual_subset_seed: int = 11
    auto_all_dirname: str = "auto_all"
    manual_auto_dirname: str = "manual_auto_20"
    flush_every: int = 100
    memory_log_every: int = 200


@dataclass
class StimulusMeta:
    """Metadata for one generated masked-audio stimulus."""

    utterance_id: int
    alpha: float
    beta: float
    mask_id: int
    kept_tokens: int
    total_tokens: int
    transcript: str
    wav_path: str


def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    """Convert audio to mono and float32, normalizing to [-1, 1]."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    mx = np.max(np.abs(audio)) + EPS
    return audio / mx


def _find_hard_boundaries(x: np.ndarray, quantile: float = 0.995) -> np.ndarray:
    """Detect likely masking-induced discontinuities."""
    if len(x) < 4:
        return np.array([], dtype=int)
    dx = np.abs(np.diff(x))
    thr = np.quantile(dx, quantile)
    idx = np.where(dx >= thr)[0]
    if len(idx) == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > 8:
            keep.append(i)
    return np.array(keep, dtype=int)


def _boundary_click_score(x: np.ndarray, sr: int, edge_idx: np.ndarray) -> float:
    """Transient and high-frequency proxy for click severity."""
    if len(edge_idx) == 0:
        return 0.0

    half_win = max(8, int(0.004 * sr))
    transient_vals = []
    hf_vals = []

    for i in edge_idx:
        lo = max(0, i - half_win)
        hi = min(len(x), i + half_win)
        seg = x[lo:hi]
        if len(seg) < 16:
            continue

        transient_vals.append(float(np.max(np.abs(np.diff(seg)))))
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        power = np.abs(spec) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        hf = power[freqs >= 4000.0].sum()
        hf_vals.append(float(hf / (power.sum() + EPS)))

    if not transient_vals:
        return 0.0
    return float(np.mean(transient_vals) + 0.5 * np.mean(hf_vals))


def _mean_spectral_flux(
    x: np.ndarray,
    sr: int,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> float:
    """Compute mean spectral flux over short-time FFT magnitudes."""
    if len(x) < 4:
        return 0.0
    frame_len = max(16, int(sr * frame_ms / 1000.0))
    hop_len = max(8, int(sr * hop_ms / 1000.0))
    if len(x) < frame_len:
        return 0.0

    window = np.hanning(frame_len).astype(np.float32)
    prev_mag: np.ndarray | None = None
    flux_vals: list[float] = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start : start + frame_len]
        mag = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        if prev_mag is not None:
            diff = mag - prev_mag
            flux_vals.append(float(np.mean(np.maximum(diff, 0.0))))
        prev_mag = mag

    if not flux_vals:
        return 0.0
    return float(np.mean(flux_vals))


def auto_click_score_from_wav(wav_path: Path) -> float:
    """Compute automated click-artifact score from a WAV file."""
    audio, sr = sf.read(wav_path)
    x = _to_mono_float(audio)
    edges = _find_hard_boundaries(x)
    return _boundary_click_score(x=x, sr=int(sr), edge_idx=edges)


def load_english_voicebench_like(config: AlphaSearchConfig) -> pd.DataFrame:
    """
    Load deterministic held-out English data with prompt/audio columns.

    Uses the single-sentence public dataset split used in experiments.
    """
    ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        revision=config.dataset_revision,  # nosec B615
    )["test"]
    df = ds.to_pandas()

    required_cols = {"sentences", "audio__male"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["sentences"].isnull().any() or df["sentences"].apply(len).max() != 1:
        raise ValueError("Expected 'sentences' to contain lists of length 1.")

    df["prompt"] = df["sentences"].apply(lambda x: x[0])
    df["audio"] = df["audio__male"].apply(lambda x: x[0])
    df = df.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    df["sent_length"] = df["prompt"].str.len()
    df = df.sort_values("sent_length").reset_index(drop=True)

    return df[["prompt", "audio"]].reset_index(drop=True)


def load_heldout_english_voicebench_like(config: AlphaSearchConfig) -> pd.DataFrame:
    """Backward-compatible helper: deterministic first n held-out rows."""
    df = load_english_voicebench_like(config)
    if len(df) < config.n_held_out:
        raise ValueError(
            f"Requested {config.n_held_out} held-out samples but only {len(df)} available."
        )
    return df.head(config.n_held_out).reset_index(drop=True)


def sample_manual_subset(
    full_df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.Index:
    """Pick deterministic manual-review subset from full dataset."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if len(full_df) < n_samples:
        raise ValueError(
            f"Requested {n_samples} manual samples but only {len(full_df)} available."
        )
    return full_df.sample(
        n=n_samples, random_state=seed, replace=False
    ).index.sort_values()


def _apply_segment_mask(
    waveform: torch.Tensor,
    segments: list[Any],
    keep_mask: np.ndarray,
) -> torch.Tensor:
    """Zero out dropped segments in waveform."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    out = waveform.clone()
    for i, seg in enumerate(segments):
        start = int(seg.start_sample or 0)
        end = int(seg.end_sample or start)
        if start < end and not bool(keep_mask[i]):
            out[:, start:end] = 0.0
    return out


def _make_random_keep_masks(
    n_tokens: int,
    n_masks: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate random binary masks for which tokens to keep."""
    masks: list[np.ndarray] = []
    while len(masks) < n_masks:
        m = rng.integers(low=0, high=2, size=n_tokens, endpoint=False).astype(bool)
        if m.any() and (~m).any():
            masks.append(m)
    return masks


def generate_stimuli(
    heldout_df: pd.DataFrame,
    config: AlphaSearchConfig,
    output_dir: Path | None = None,
    device: str | torch.device | None = None,
) -> pd.DataFrame:
    """Generate masked-audio stimuli for each alpha."""
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    output_dir = output_dir or config.output_dir
    wav_dir = output_dir / "masked_wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.rng_seed)
    stimuli: list[StimulusMeta] = []

    for alpha in tqdm(config.alphas, desc="alpha grid"):
        beta = 1.0 - alpha
        aligner = SpectrogramGuidedAligner(
            device=torch.device(device),
            boundary_energy_weight=alpha,
            boundary_flux_weight=beta,
        )

        for utt_id, row in tqdm(
            heldout_df.iterrows(),
            total=len(heldout_df),
            leave=False,
            desc=f"alpha={alpha:.1f}",
        ):
            waveform, sr = TorchAudioHandler.from_bytes(
                row["audio"], audio_format="wav"
            )
            segments = aligner(
                transcript=row["prompt"],
                waveform=waveform,
                original_sr=sr,
                attach_audio=False,
            )
            if len(segments) < 2:
                continue

            keep_masks = _make_random_keep_masks(
                len(segments), config.masks_per_utterance, rng
            )
            for mask_id, keep_mask in enumerate(keep_masks):
                masked_wave = _apply_segment_mask(
                    waveform=waveform, segments=segments, keep_mask=keep_mask
                )
                wav_name = f"utt{utt_id:03d}_a{alpha:.1f}_m{mask_id:02d}.wav"
                wav_path = wav_dir / wav_name
                wav_path.write_bytes(
                    TorchAudioHandler.to_bytes(
                        masked_wave,
                        sample_rate=sr,
                        audio_format="wav",
                    )
                )
                stimuli.append(
                    StimulusMeta(
                        utterance_id=int(utt_id),
                        alpha=float(alpha),
                        beta=float(beta),
                        mask_id=int(mask_id),
                        kept_tokens=int(keep_mask.sum()),
                        total_tokens=int(len(keep_mask)),
                        transcript=str(row["prompt"]),
                        wav_path=str(wav_path),
                    )
                )

    stimuli_df = pd.DataFrame([asdict(s) for s in stimuli]).sort_values(
        ["alpha", "utterance_id", "mask_id"]
    )
    stimuli_df.to_csv(output_dir / "stimuli_manifest.csv", index=False)
    return stimuli_df.reset_index(drop=True)


def run_machine_annotation(
    stimuli_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add automated click scores and aggregate ranking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = stimuli_df.copy()
    click_scores: list[float] = []
    spectral_fluxes: list[float] = []
    for wav_path in scored["wav_path"]:
        audio, sr = sf.read(Path(str(wav_path)))
        x = _to_mono_float(audio)
        edges = _find_hard_boundaries(x)
        click_scores.append(_boundary_click_score(x=x, sr=int(sr), edge_idx=edges))
        spectral_fluxes.append(_mean_spectral_flux(x=x, sr=int(sr)))
    scored["auto_click_score"] = click_scores
    scored["spectral_flux"] = spectral_fluxes
    scored.to_csv(output_dir / "auto_scores.csv", index=False)

    summary = (
        scored.groupby("alpha", as_index=False)
        .agg(
            beta=("beta", "first"),
            mean_auto_click=("auto_click_score", "mean"),
            std_auto_click=("auto_click_score", "std"),
            mean_spectral_flux=("spectral_flux", "mean"),
            std_spectral_flux=("spectral_flux", "std"),
            n_stimuli=("auto_click_score", "count"),
        )
        .sort_values(["mean_auto_click", "alpha"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(output_dir / "auto_summary.csv", index=False)
    return scored, summary


def _update_alpha_stats(
    stats: dict[float, dict[str, float]],
    alpha: float,
    beta: float,
    value: float,
) -> None:
    bucket = stats.setdefault(
        float(alpha),
        {"beta": float(beta), "sum": 0.0, "sum_sq": 0.0, "count": 0.0},
    )
    bucket["sum"] += float(value)
    bucket["sum_sq"] += float(value) * float(value)
    bucket["count"] += 1.0


def _alpha_summary_from_stats(
    stats: dict[float, dict[str, float]],
    mean_col: str = "mean_auto_click",
    std_col: str = "std_auto_click",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for alpha, bucket in stats.items():
        count = int(bucket["count"])
        if count == 0:
            continue
        mean = float(bucket["sum"] / bucket["count"])
        if count > 1:
            var = max(0.0, (bucket["sum_sq"] / bucket["count"]) - mean * mean)
            std = float(np.sqrt(var))
        else:
            std = float("nan")
        rows.append(
            {
                "alpha": float(alpha),
                "beta": float(bucket["beta"]),
                mean_col: mean,
                std_col: std,
                "n_stimuli": count,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [mean_col, "alpha"], ascending=[True, False]
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def _score_masked_waveform(
    masked_wave: torch.Tensor, sample_rate: int
) -> tuple[float, float]:
    waveform_np = masked_wave.detach().cpu().numpy()
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=0)
    x = _to_mono_float(waveform_np)
    edges = _find_hard_boundaries(x)
    click_score = _boundary_click_score(x=x, sr=int(sample_rate), edge_idx=edges)
    spectral_flux = _mean_spectral_flux(x=x, sr=int(sample_rate))
    return float(click_score), float(spectral_flux)


def _process_rss_gb() -> float:
    """Return current process RSS in GB (best effort, Unix units aware)."""
    rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS ru_maxrss is bytes, Linux is KB.
    if rss_kb > 10_000_000:  # clearly bytes-scale
        rss_bytes = rss_kb
    else:
        rss_bytes = rss_kb * 1024.0
    return float(rss_bytes / (1024.0**3))


def _prompt_key(prompt: str) -> int:
    """Compact stable key for deduplication (8-byte blake2b digest)."""
    digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def build_dual_alpha_datasets(
    config: AlphaSearchConfig,
    device: str | torch.device | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Create two datasets:
    1) manual+automatic labels for 20 samples
    2) automatic-only labels for the full dataset

    WAV files are generated once (for full dataset) and reused by the manual subset.
    """
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    base_output_dir = config.output_dir
    auto_all_dir = base_output_dir / config.auto_all_dirname
    manual_auto_dir = base_output_dir / config.manual_auto_dirname
    auto_all_dir.mkdir(parents=True, exist_ok=True)
    manual_auto_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        config.dataset_name,
        config.dataset_config,
        revision=config.dataset_revision,  # nosec B615
    )["test"]

    # Pass 1: determine deduplicated utterance count without loading all audio into memory.
    seen_prompts: set[int] = set()
    utterance_count = 0
    for row in ds:
        sentences = row.get("sentences")
        if not isinstance(sentences, list) or len(sentences) != 1:
            continue
        prompt = str(sentences[0])
        prompt_key = _prompt_key(prompt)
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        utterance_count += 1

    if utterance_count < config.n_held_out:
        raise ValueError(
            f"Requested {config.n_held_out} manual samples but only {utterance_count} available."
        )

    manual_ids = set(
        np.random.default_rng(config.manual_subset_seed)
        .choice(utterance_count, size=config.n_held_out, replace=False)
        .tolist()
    )

    # Prepare streaming outputs.
    auto_manifest_path = auto_all_dir / "stimuli_manifest.csv"
    auto_scores_path = auto_all_dir / "auto_scores.csv"
    manual_manifest_path = manual_auto_dir / "stimuli_manifest.csv"
    manual_scores_path = manual_auto_dir / "auto_scores.csv"

    auto_wav_dir = auto_all_dir / "masked_wavs"
    auto_wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "utterance_id",
        "alpha",
        "beta",
        "mask_id",
        "kept_tokens",
        "total_tokens",
        "transcript",
        "wav_path",
    ]
    score_fields = [*manifest_fields, "auto_click_score", "spectral_flux"]

    auto_stats: dict[float, dict[str, float]] = {}
    manual_stats: dict[float, dict[str, float]] = {}
    auto_flux_stats: dict[float, dict[str, float]] = {}
    manual_flux_stats: dict[float, dict[str, float]] = {}
    rng = np.random.default_rng(config.rng_seed)
    aligners = {
        float(alpha): SpectrogramGuidedAligner(
            device=torch.device(device),
            boundary_energy_weight=float(alpha),
            boundary_flux_weight=float(1.0 - float(alpha)),
        )
        for alpha in config.alphas
    }

    with (
        auto_manifest_path.open(
            "w", encoding="utf-8", newline=""
        ) as auto_manifest_file,
        auto_scores_path.open("w", encoding="utf-8", newline="") as auto_scores_file,
        manual_manifest_path.open(
            "w", encoding="utf-8", newline=""
        ) as manual_manifest_file,
        manual_scores_path.open(
            "w", encoding="utf-8", newline=""
        ) as manual_scores_file,
    ):
        auto_manifest_writer = csv.DictWriter(
            auto_manifest_file, fieldnames=manifest_fields
        )
        auto_scores_writer = csv.DictWriter(auto_scores_file, fieldnames=score_fields)
        manual_manifest_writer = csv.DictWriter(
            manual_manifest_file, fieldnames=manifest_fields
        )
        manual_scores_writer = csv.DictWriter(
            manual_scores_file, fieldnames=score_fields
        )

        auto_manifest_writer.writeheader()
        auto_scores_writer.writeheader()
        manual_manifest_writer.writeheader()
        manual_scores_writer.writeheader()

        flush_every = max(1, int(config.flush_every))
        log_every = max(1, int(config.memory_log_every))
        seen_prompts.clear()
        utterance_id = 0
        for row in tqdm(ds, desc="dataset"):
            sentences = row.get("sentences")
            audios = row.get("audio__male")
            if not isinstance(sentences, list) or len(sentences) != 1:
                continue
            if not isinstance(audios, list) or len(audios) == 0:
                continue

            prompt = str(sentences[0])
            prompt_key = _prompt_key(prompt)
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)

            waveform, sr = TorchAudioHandler.from_bytes(audios[0], audio_format="wav")
            for alpha in config.alphas:
                alpha_f = float(alpha)
                beta_f = float(1.0 - alpha_f)
                with torch.inference_mode():
                    segments = aligners[alpha_f](
                        transcript=prompt,
                        waveform=waveform,
                        original_sr=sr,
                        attach_audio=False,
                    )
                if len(segments) < 2:
                    del segments
                    continue

                keep_masks = _make_random_keep_masks(
                    n_tokens=len(segments),
                    n_masks=config.masks_per_utterance,
                    rng=rng,
                )
                for mask_id, keep_mask in enumerate(keep_masks):
                    with torch.inference_mode():
                        masked_wave = _apply_segment_mask(
                            waveform=waveform, segments=segments, keep_mask=keep_mask
                        )
                    wav_name = (
                        f"utt{utterance_id:05d}_a{alpha_f:.1f}_m{mask_id:02d}.wav"
                    )
                    wav_path = auto_wav_dir / wav_name
                    wav_path.write_bytes(
                        TorchAudioHandler.to_bytes(
                            masked_wave, sample_rate=sr, audio_format="wav"
                        )
                    )
                    click_score, spectral_flux = _score_masked_waveform(
                        masked_wave, int(sr)
                    )

                    base_row = {
                        "utterance_id": int(utterance_id),
                        "alpha": alpha_f,
                        "beta": beta_f,
                        "mask_id": int(mask_id),
                        "kept_tokens": int(keep_mask.sum()),
                        "total_tokens": int(len(keep_mask)),
                        "transcript": prompt,
                        "wav_path": str(wav_path),
                    }
                    scored_row = {
                        **base_row,
                        "auto_click_score": float(click_score),
                        "spectral_flux": float(spectral_flux),
                    }

                    auto_manifest_writer.writerow(base_row)
                    auto_scores_writer.writerow(scored_row)
                    _update_alpha_stats(auto_stats, alpha_f, beta_f, click_score)
                    _update_alpha_stats(auto_flux_stats, alpha_f, beta_f, spectral_flux)

                    if utterance_id in manual_ids:
                        manual_manifest_writer.writerow(base_row)
                        manual_scores_writer.writerow(scored_row)
                        _update_alpha_stats(manual_stats, alpha_f, beta_f, click_score)
                        _update_alpha_stats(
                            manual_flux_stats, alpha_f, beta_f, spectral_flux
                        )
                    del masked_wave
                del keep_masks
                del segments

            if utterance_id > 0 and utterance_id % flush_every == 0:
                auto_manifest_file.flush()
                auto_scores_file.flush()
                manual_manifest_file.flush()
                manual_scores_file.flush()

            if utterance_id > 0 and utterance_id % log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                tqdm.write(
                    f"[mem] processed={utterance_id} rss_gb={_process_rss_gb():.2f}"
                )

            # Explicitly release per-utterance tensors/references.
            del waveform
            utterance_id += 1

    auto_all_scores = pd.read_csv(auto_scores_path)
    auto_all_summary = _alpha_summary_from_stats(auto_stats)
    auto_flux_summary = _alpha_summary_from_stats(
        auto_flux_stats,
        mean_col="mean_spectral_flux",
        std_col="std_spectral_flux",
    ).drop(columns=["rank"], errors="ignore")
    auto_all_summary = auto_all_summary.merge(
        auto_flux_summary,
        on=["alpha", "beta", "n_stimuli"],
        how="left",
    )
    auto_all_summary.to_csv(auto_all_dir / "auto_summary.csv", index=False)

    manual_scores = pd.read_csv(manual_scores_path)
    manual_summary = _alpha_summary_from_stats(manual_stats)
    manual_flux_summary = _alpha_summary_from_stats(
        manual_flux_stats,
        mean_col="mean_spectral_flux",
        std_col="std_spectral_flux",
    ).drop(columns=["rank"], errors="ignore")
    manual_summary = manual_summary.merge(
        manual_flux_summary,
        on=["alpha", "beta", "n_stimuli"],
        how="left",
    )
    manual_summary.to_csv(manual_auto_dir / "auto_summary.csv", index=False)

    manual_manifest = pd.read_csv(manual_manifest_path)
    ratings_template = create_human_ratings_template(
        stimuli_df=manual_manifest,
        output_dir=manual_auto_dir,
        reviewer_count=1,
    )

    return {
        "auto_all_scores": auto_all_scores,
        "auto_all_summary": auto_all_summary,
        "manual_auto_scores": manual_scores,
        "manual_auto_summary": manual_summary,
        "manual_ratings_template": pd.DataFrame({"path": [str(ratings_template)]}),
    }


def create_human_ratings_template(
    stimuli_df: pd.DataFrame,
    output_dir: Path,
    reviewer_count: int = 1,
) -> Path:
    """Create CSV template for click-MOS annotation."""
    if reviewer_count < 1:
        raise ValueError("reviewer_count must be >= 1")
    ratings = stimuli_df.copy()
    for i in range(1, reviewer_count + 1):
        ratings[f"annotator_{i}"] = np.nan
    out = output_dir / "ratings_template.csv"
    ratings.to_csv(out, index=False)
    return out


def summarize_human_ratings(ratings_csv: Path) -> pd.DataFrame:
    """Aggregate 3-annotator MOS ratings by alpha."""
    ratings = pd.read_csv(ratings_csv)
    annotator_cols = sorted(c for c in ratings.columns if c.startswith("annotator_"))
    if not annotator_cols:
        raise ValueError("No annotator_* columns found in ratings CSV.")
    required = ["alpha", *annotator_cols]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise ValueError(f"Missing rating columns: {missing}")

    for col in annotator_cols:
        if ratings[col].isna().any():
            raise ValueError(f"Column `{col}` contains missing ratings.")

    ratings["mos_click"] = ratings[annotator_cols].mean(axis=1)
    summary = (
        ratings.groupby("alpha", as_index=False)
        .agg(
            beta=("beta", "first"),
            mean_click_mos=("mos_click", "mean"),
            std_click_mos=("mos_click", "std"),
            n_stimuli=("mos_click", "count"),
        )
        .sort_values(["mean_click_mos", "alpha"], ascending=[True, False])
        .reset_index(drop=True)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def interactive_single_reviewer_annotation(
    ratings_csv: Path,
    reviewer_col: str = "annotator_1",
    score_min: int = 1,
    score_max: int = 5,
    autosave_every: int = 1,
) -> pd.DataFrame:
    """
    Run interactive loop for one reviewer using tkinter UI.

    Flow:
    - autoplay clip first
    - allow replay button
    - accept numeric score in range
    - save progress to CSV
    """
    ratings = pd.read_csv(ratings_csv)
    if reviewer_col not in ratings.columns:
        raise ValueError(f"Missing reviewer column: {reviewer_col}")
    if "wav_path" not in ratings.columns:
        raise ValueError("Missing `wav_path` column in ratings CSV.")

    pending_idx = ratings.index[ratings[reviewer_col].isna()].tolist()
    if not pending_idx:
        print(f"No pending rows for {reviewer_col}.")
        return ratings

    print(
        f"Starting annotation for {reviewer_col}. "
        f"Valid scores: [{score_min}, {score_max}]. Pending: {len(pending_idx)}"
    )

    saved = 0
    transcript_col = "transcript" if "transcript" in ratings.columns else None

    def _play_note_and_get_score_tk(
        wav_path: Path,
        info_text: str,
    ) -> int | None:
        """Play audio and show tkinter UI to get score. Returns None if user quits."""
        result: dict[str, int | bool | None] = {"score": None, "quit": False}
        player: dict[str, subprocess.Popen | None] = {"proc": None}
        afplay = shutil.which("afplay")

        def stop_audio() -> None:
            proc = player["proc"]
            if proc is not None and proc.poll() is None:
                proc.terminate()

        def play_audio() -> None:
            stop_audio()
            if afplay is None:
                return
            player["proc"] = subprocess.Popen([afplay, str(wav_path)])

        root = tk.Tk()
        root.title("Audio Rating")
        root.geometry("680x280")

        header = tk.Label(root, text=info_text, justify="left", wraplength=650)
        header.pack(pady=8, padx=12, anchor="w")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=4)
        tk.Button(btn_frame, text="Replay", command=play_audio, width=12).pack(
            side="left", padx=4
        )

        score_var = tk.IntVar(value=0)
        score_frame = tk.Frame(root)
        score_frame.pack(pady=8)
        for s in range(score_min, score_max + 1):
            tk.Radiobutton(score_frame, text=str(s), variable=score_var, value=s).pack(
                side="left", padx=6
            )

        def submit() -> None:
            """Validate score and close UI."""
            val = int(score_var.get())
            if val < score_min or val > score_max:
                messagebox.showwarning("Invalid", "Pick score first.")
                return
            result["score"] = val
            root.destroy()

        def quit_now() -> None:
            """Set quit flag and close UI."""
            result["quit"] = True
            root.destroy()

        action_frame = tk.Frame(root)
        action_frame.pack(pady=10)
        tk.Button(action_frame, text="Submit", command=submit, width=12).pack(
            side="left", padx=6
        )
        tk.Button(action_frame, text="Quit", command=quit_now, width=12).pack(
            side="left", padx=6
        )

        play_audio()  # autoplay on window open
        root.mainloop()
        stop_audio()

        if bool(result["quit"]):
            return None
        return int(result["score"]) if result["score"] is not None else None

    for k, idx in enumerate(pending_idx, start=1):
        row = ratings.loc[idx]
        wav_path = Path(str(row["wav_path"]))
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        transcript = (
            str(row[transcript_col]) if transcript_col is not None else "<missing>"
        )
        info_text = (
            f"[{k}/{len(pending_idx)}] alpha={row.get('alpha')} "
            f"utt={row.get('utterance_id')} mask={row.get('mask_id')}\n"
            f"Transcript: {transcript}"
        )

        score = _play_note_and_get_score_tk(wav_path=wav_path, info_text=info_text)

        if score is None:
            ratings.to_csv(ratings_csv, index=False)
            print(f"Progress saved to: {ratings_csv}")
            return ratings

        ratings.at[idx, reviewer_col] = score
        saved += 1
        if saved % autosave_every == 0:
            ratings.to_csv(ratings_csv, index=False)

    ratings.to_csv(ratings_csv, index=False)
    print(f"Done. Saved to: {ratings_csv}")
    return ratings


def answer_per_alpha_mean_spectral_flux(
    summary_df: pd.DataFrame | None = None,
    summary_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Answer whether per-alpha mean spectral-flux values are available.

    Note:
    - Current pipeline stores *weights* (`alpha`, `beta=1-alpha`) and click scores.
    - It does not log per-stimulus spectral-flux measurements.
    """
    if summary_df is None:
        if summary_csv is None:
            raise ValueError("Provide either `summary_df` or `summary_csv`.")
        summary_df = pd.read_csv(summary_csv)

    if "alpha" not in summary_df.columns:
        raise ValueError("Missing `alpha` column in summary data.")

    out = summary_df[["alpha"]].drop_duplicates().copy()
    out = (
        summary_df[["alpha", "mean_spectral_flux"]]
        .drop_duplicates(subset=["alpha"])
        .sort_values("alpha")
        .reset_index(drop=True)
    )
    out["flux_weight_beta"] = 1.0 - out["alpha"].astype(float)
    return out

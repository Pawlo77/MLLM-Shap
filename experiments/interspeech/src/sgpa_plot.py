"""Paper-Grade SGPA Visualization Script"""

import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from typing import Any

# --- 1. Centralized Paper-Grade Styles ---
STYLES = {
    "paper_rc": {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
    "text": dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3.0),
    "annotation": dict(
        facecolor="white",
        edgecolor="#cccccc",
        alpha=0.9,
        pad=4.0,
        boxstyle="round,pad=0.3",
    ),
    "success": dict(
        facecolor="#E8F8E8",
        edgecolor="#009E73",
        alpha=0.9,
        pad=4.0,
        boxstyle="round,pad=0.3",
    ),
    "fail": dict(
        facecolor="#F8E8E8",
        edgecolor="#D55E00",
        alpha=0.9,
        pad=4.0,
        boxstyle="round,pad=0.3",
    ),
}


def _render_sgpa_panels(data: dict, layout: dict) -> plt.Figure:
    """Unified plotting function to render the 4-panel SGPA figure."""
    plt.rcParams.update(STYLES["paper_rc"])

    fig, axes = plt.subplots(
        4,
        1,
        figsize=layout["figsize"],
        sharex=True,
        gridspec_kw={"height_ratios": layout.get("height_ratios", [1, 1, 1, 1])},
    )
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.08, left=0.08, right=0.95)

    is_exact = layout.get("exact", False)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x")
        ax.set_xlim(layout.get("x_min", 0), layout.get("x_max", 1.2))

    t_L, t_R, refined = data["t_L"], data["t_R"], data["refined_boundary"]
    search_start, search_end = data["search_start"], data["search_end"]

    # -- (a) Stage 1: Waveform --
    ax0 = axes[0]
    ax0.plot(data["t_wave"], data["waveform"], color="#404040", lw=0.8)
    ax0.set_ylabel("Amplitude")
    ax0.set_title(
        "Stage 1: Waveform & Transcript Decomposition",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax0.set_ylim(-layout["stage1_ymax"], layout["stage1_ymax"])

    ax0.text(
        data["w1_t"],
        layout["stage1_ymax"] * 0.7,
        data["w1"],
        ha=layout.get("w1_ha", "center"),
        va="center",
        fontweight="bold",
        fontsize=12,
        color="#0055A4",
        bbox=STYLES["text"],
    )
    ax0.text(
        data["w2_t"],
        layout["stage1_ymax"] * 0.7,
        data["w2"],
        ha=layout.get("w2_ha", "center"),
        va="center",
        fontweight="bold",
        fontsize=12,
        color="#0055A4",
        bbox=STYLES["text"],
    )

    # -- (b) Stage 2: Initial Alignment --
    ax1 = axes[1]
    ax1.plot(
        data["t_emissions"],
        data["char_prob"],
        color="#0055A4",
        lw=2,
        label="Character Prob",
    )
    ax1.plot(
        data["t_emissions"],
        data["blank_prob"],
        color="#777777",
        lw=1.5,
        linestyle=":",
        label="Blank Prob",
    )
    ax1.fill_between(data["t_emissions"], data["char_prob"], color="#0055A4", alpha=0.1)

    ax1.axvline(
        t_L,
        color="#D55E00",
        linestyle="--",
        alpha=0.8,
        label=r"Raw Bounds ($t_L, t_R$)",
    )
    ax1.axvline(t_R, color="#D55E00", linestyle="--", alpha=0.8)
    ax1.axvspan(t_L, t_R, color="#D55E00", alpha=0.1)
    ax1.text(
        data["gap_midpoint"],
        layout["stage2_text_y"],
        "Raw Inter-Character\nGap",
        ha="center",
        va="center",
        fontsize=11,
        color="#D55E00",
        bbox=STYLES["text"],
    )

    ax1.set_ylabel("Probability")
    ax1.set_title(
        "Stage 2: Initial Alignment & Gap Extraction",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylim(0, 1.35)
    ax1.legend(
        loc="upper right" if is_exact else (0.82, 0.77),
        frameon=True,
        edgecolor="#dddddd",
        ncol=1,
    )

    # -- (c) Stage 3: Spectral Boundary Refinement --
    ax2 = axes[2]
    ax2.plot(
        data["t_features"],
        data["energy"],
        color="#CC79A7",
        lw=1.5,
        alpha=0.7,
        label=r"Energy $E[t]$",
    )
    ax2.plot(
        data["t_features"],
        data["flux"],
        color="#E69F00",
        lw=1.5,
        alpha=0.7,
        label=r"Flux $SF[t]$",
    )
    ax2.plot(
        data["t_features"],
        data["cost"],
        color="#009E73",
        lw=2.5,
        label=r"Cost ($\alpha E + \beta SF$)",
    )

    ax2.axvspan(search_start, search_end, color="#009E73", alpha=0.1)
    window_text = (
        "Search Window [0.41s, 0.69s]"
        if is_exact
        else f"Search Window [{search_start:.2f}s, {search_end:.2f}s]"
    )
    ax2.text(
        search_start + 0.01,
        1.15,
        window_text,
        color="#009E73",
        fontsize=10,
        fontweight="bold",
        bbox=STYLES["text"],
    )

    ax2.axvline(
        t_L,
        color="#777777" if is_exact else "#D55E00",
        linestyle=":" if is_exact else "--",
        lw=2,
        label="np.clip() Limits" if is_exact else "Naive Cut ($t_L$)",
    )
    ax2.axvline(
        t_R,
        color="#777777",
        linestyle=":",
        lw=2,
        label=None if is_exact else "Right Bound ($t_R$)",
    )
    ax2.axvline(refined, color="black", linestyle="-", lw=2, label="Refined Minimum")

    cost_idx = np.abs(data["t_features"] - refined).argmin()
    ax2.scatter([refined], [data["cost"][cost_idx]], color="black", zorder=5, s=60)

    passed = data.get("passed_silence", True)
    ax2.text(
        refined + 0.02,
        0.45,
        "Passes Silence Threshold\n(min < 0.5 * mean)"
        if passed
        else "Falls back to Midpoint",
        fontsize=10,
        fontweight="bold",
        color="#009E73" if passed else "#D55E00",
        bbox=STYLES["success"] if passed else STYLES["fail"],
    )

    ax2.set_ylabel("Normalized Cost")
    ax2.set_title(
        f"Stage 3: Spectrogram-Guided Refinement ({'Algorithm Constraints' if is_exact else 'Real Implementation'})",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax2.set_ylim(0, 1.35)
    ax2.legend(
        loc=(0.66, 0.74) if is_exact else (0.66, 0.8),
        frameon=True,
        edgecolor="#dddddd",
        ncol=2,
    )

    # -- (d) Stage 4: Word-Level Aggregation --
    ax3 = axes[3]
    ax3.set_title(
        "Stage 4: Word-Level Aggregation (Joint Boundary Resolution)",
        loc="left",
        fontweight="bold",
        pad=10,
    )
    ax3.set_yticks([])
    ax3.set_ylim(layout["stage4_ylim"])
    ax3.spines["left"].set_visible(False)
    ax3.grid(False)

    ax3.axvline(
        t_L,
        color="#D55E00",
        linestyle="--",
        linewidth=2,
        label="Naive Cut (Left Gap Edge)",
    )
    ax3.axvline(
        refined,
        color="#009E73",
        linestyle="-",
        linewidth=2.5,
        label="SGPA Cut (Silence)",
    )

    if data["shift_ms"] > 0:
        arrow_y = layout["stage4_arrow_y"]
        ax3.annotate(
            text="",
            xy=(refined, arrow_y),
            xytext=(t_L, arrow_y),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5),
        )
        ax3.text(
            (t_L + refined) / 2,
            arrow_y + layout["arrow_offset"],
            f"Correction (+{data['shift_ms']:.0f}ms)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            color="#333333",
            bbox=STYLES["annotation"],
        )

    rect_y, rect_h, text_y = (
        layout["stage4_rect_y"],
        layout["stage4_rect_h"],
        layout["stage4_text_y"],
    )

    if (r1_w := refined - data["w1_start"]) > 0:
        ax3.add_patch(
            patches.Rectangle(
                (data["w1_start"], rect_y),
                r1_w,
                rect_h,
                linewidth=1.5,
                edgecolor="black",
                facecolor="#E8F4F8",
            )
        )
        ax3.text(
            data["w1_start"] + r1_w / 2,
            text_y,
            f"Segment 1: {data['w1']}",
            ha="center",
            va="center",
            fontweight="bold",
            color="#333333",
        )

    if (r2_w := data["w2_end"] - refined) > 0:
        ax3.add_patch(
            patches.Rectangle(
                (refined, rect_y),
                r2_w,
                rect_h,
                linewidth=1.5,
                edgecolor="black",
                facecolor="#F8E8E8",
            )
        )
        ax3.text(
            refined + r2_w / 2,
            text_y,
            f"Segment 2: {data['w2']}",
            ha="center",
            va="center",
            fontweight="bold",
            color="#333333",
        )

    if is_exact:
        ax0.axvline(refined, color="black", linestyle=":", alpha=0.5)
    ax1.axvline(refined, color="black", linestyle=":", alpha=0.5)

    ax3.set_xlabel("Time (seconds)", fontweight="bold", labelpad=10)
    ax3.legend(
        loc="upper right" if is_exact else (0.79, 0.8),
        frameon=True,
        edgecolor="#dddddd",
    )

    return fig


def generate_exact_sgpa_figure() -> plt.Figure:
    """Generates the synthetic exact SGPA figure."""
    fs, duration = 4000, 1.2
    t = np.linspace(0, duration, int(fs * duration))
    waveform = np.random.normal(0, 0.02, len(t))

    # Construct Words
    h_mask, ello_mask = (t > 0.1) & (t < 0.15), (t >= 0.15) & (t < 0.55)
    w_mask, d_mask = (t > 0.65) & (t < 0.95), (t >= 0.95) & (t < 1.0)

    waveform[h_mask] += np.random.normal(0, 0.3, sum(h_mask)) * np.hanning(sum(h_mask))
    waveform[ello_mask] += (
        (
            np.sin(2 * np.pi * 140 * t[ello_mask])
            + 0.5 * np.sin(2 * np.pi * 280 * t[ello_mask])
        )
        * np.exp(-6 * (t[ello_mask] - 0.15))
        * np.hanning(sum(ello_mask))
        * 0.9
    )
    waveform[w_mask] += (
        (
            np.sin(2 * np.pi * 120 * t[w_mask])
            + 0.3 * np.sin(2 * np.pi * 240 * t[w_mask])
        )
        * np.hanning(sum(w_mask))
        * 0.8
    )
    waveform[d_mask] += np.random.normal(0, 0.4, sum(d_mask)) * np.exp(
        -30 * (t[d_mask] - 0.95)
    )

    # Constants & Bounds
    t_L, t_R, gap_midpoint = 0.45, 0.65, 0.55
    search_start, search_end, refined_boundary = 0.41, 0.69, 0.59

    ctc_probs = 0.9 * np.exp(-((t - 0.28) ** 2) / 0.01) + 0.85 * np.exp(
        -((t - 0.80) ** 2) / 0.01
    )

    # Spectral Setup
    energy = np.clip(
        np.interp(t, [0, 0.41, refined_boundary, 0.69, 1.2], [0.8, 0.6, 0.05, 0.5, 0.8])
        + np.random.normal(0, 0.02, len(t)),
        0,
        1,
    )
    flux = np.clip(
        np.interp(t, [0, 0.41, refined_boundary, 0.69, 1.2], [0.5, 0.4, 0.08, 0.4, 0.5])
        + np.random.normal(0, 0.03, len(t)),
        0,
        1,
    )

    return _render_sgpa_panels(
        data={
            "t_wave": t,
            "waveform": waveform,
            "t_emissions": t,
            "char_prob": ctc_probs,
            "blank_prob": 1.0 - np.clip(ctc_probs * 1.5, 0, 1),
            "t_features": t,
            "energy": energy,
            "flux": flux,
            "cost": 0.8 * energy + 0.2 * flux,
            "w1": "HELLO",
            "w1_t": 0.28,
            "w1_start": 0.05,
            "w2": "WORLD",
            "w2_t": 0.80,
            "w2_end": 1.15,
            "t_L": t_L,
            "t_R": t_R,
            "gap_midpoint": gap_midpoint,
            "search_start": search_start,
            "search_end": search_end,
            "refined_boundary": refined_boundary,
            "passed_silence": True,
            "shift_ms": 140,
        },
        layout={
            "exact": True,
            "figsize": (12, 14),
            "stage1_ymax": 1.2,
            "stage2_text_y": 0.4,
            "stage4_ylim": (0, 1.2),
            "stage4_arrow_y": 0.85,
            "stage4_rect_y": 0.15,
            "stage4_rect_h": 0.5,
            "stage4_text_y": 0.4,
            "arrow_offset": 0.08,
        },
    )


def generate_real_sgpa_figure(
    aligner,
    prompt: str,
    audio_bytes: bytes,
    audio_handler: Any,
    audio_format: str = "wav",
    zoom_padding: float = 0.4,
) -> plt.Figure:
    """Generates the exact 4-panel SGPA paper figure using real internal states."""
    waveform, sr = audio_handler.from_bytes(audio_bytes, audio_format=audio_format)
    numpy_wave = waveform.cpu().numpy().squeeze()
    duration = len(numpy_wave) / sr
    t_wave = np.linspace(0, duration, len(numpy_wave))

    _, target_segments, _, valid_tokens = (
        aligner._SpectrogramGuidedAligner__prepare_transcript(prompt)
    )
    if len(target_segments) < 2:
        raise ValueError("Prompt must contain at least two words.")

    alignment_path, emissions_gpu = (
        aligner._SpectrogramGuidedAligner__perform_forced_alignment(
            waveform, sr, valid_tokens
        )
    )
    token_spans = aligner._SpectrogramGuidedAligner__merge_tokens(
        alignment_path, aligner.blank_id
    )

    w1_len = len(
        [c for c in aligner.normalize_text(target_segments[0]) if c in aligner.vocab]
    )
    w2_len = len(
        [c for c in aligner.normalize_text(target_segments[1]) if c in aligner.vocab]
    )

    ratio = waveform.size(1) / emissions_gpu.size(0)
    t_W1_start = (token_spans[0][1] * ratio) / sr
    t_L = (token_spans[w1_len - 1][2] * ratio) / sr
    t_R = (token_spans[w1_len][1] * ratio) / sr
    t_W2_end = (token_spans[w1_len + w2_len - 1][2] * ratio) / sr

    gap_midpoint = (t_L + t_R) / 2.0
    search_start = gap_midpoint - ((t_R - t_L) / 2.0 + 0.040)
    search_end = gap_midpoint + ((t_R - t_L) / 2.0 + 0.040)

    refined_boundary, passed_silence = (
        aligner._SpectrogramGuidedAligner__refine_boundary_smart(
            numpy_wave, sr, candidate_time=gap_midpoint, left_time=t_L, right_time=t_R
        )
    )
    refined_boundary = float(np.clip(refined_boundary, t_L, t_R))

    emissions_cpu = torch.exp(emissions_gpu).cpu().numpy()
    t_emissions = np.arange(emissions_cpu.shape[0]) * ratio / sr
    blank_prob = emissions_cpu[:, aligner.blank_id]

    hop_length = 64
    rms = librosa.feature.rms(y=numpy_wave, frame_length=256, hop_length=hop_length)[0]
    stft = np.abs(librosa.stft(numpy_wave, n_fft=256, hop_length=hop_length))
    flux = np.pad(
        np.sum(np.diff(stft, axis=1) ** 2, axis=0),
        (0, len(rms) - (stft.shape[1] - 1)),
        mode="constant",
    )
    t_features = np.arange(len(rms)) * hop_length / sr

    search_mask = (t_features >= search_start) & (t_features <= search_end)
    r_min, r_max = (
        (np.min(rms[search_mask]), np.max(rms[search_mask]))
        if np.any(search_mask)
        else (np.min(rms), np.max(rms))
    )
    f_min, f_max = (
        (np.min(flux[search_mask]), np.max(flux[search_mask]))
        if np.any(search_mask)
        else (np.min(flux), np.max(flux))
    )

    rms_norm = np.clip((rms - r_min) / (r_max - r_min + 1e-9), 0, 1)
    flux_norm = np.clip((flux - f_min) / (f_max - f_min + 1e-9), 0, 1)
    cost = (
        aligner.boundary_energy_weight * rms_norm
        + aligner.boundary_flux_weight * flux_norm
    )

    x_min, x_max = (
        max(0, search_start - zoom_padding),
        min(duration, search_end + zoom_padding),
    )

    _render_sgpa_panels(
        data={
            "t_wave": t_wave,
            "waveform": numpy_wave,
            "t_emissions": t_emissions,
            "char_prob": 1.0 - blank_prob,
            "blank_prob": blank_prob,
            "t_features": t_features,
            "energy": rms_norm,
            "flux": flux_norm,
            "cost": cost,
            "w1": target_segments[0].upper(),
            "w1_t": t_L - 0.1,
            "w1_start": max(x_min + 0.01, t_W1_start),
            "w2": target_segments[1].upper(),
            "w2_t": t_R + 0.1,
            "w2_end": min(x_max - 0.01, t_W2_end),
            "t_L": t_L,
            "t_R": t_R,
            "gap_midpoint": gap_midpoint,
            "search_start": search_start,
            "search_end": search_end,
            "refined_boundary": refined_boundary,
            "passed_silence": passed_silence,
            "shift_ms": (refined_boundary - t_L) * 1000,
        },
        layout={
            "exact": False,
            "figsize": (12, 9),
            "height_ratios": [0.8, 1, 1, 0.8],
            "x_min": x_min,
            "x_max": x_max,
            "stage1_ymax": np.max(np.abs(numpy_wave[int(x_min * sr) : int(x_max * sr)]))
            * 1.2,
            "w1_ha": "right",
            "w2_ha": "left",
            "stage2_text_y": 1.15,
            "stage4_ylim": (0.1, 0.45),
            "stage4_arrow_y": 0.37,
            "stage4_rect_y": 0.15,
            "stage4_rect_h": 0.2,
            "stage4_text_y": 0.25,
            "arrow_offset": 0.03,
        },
    )

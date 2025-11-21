"""Utilities and experiment runner for Neyman SHAP experiments.

This module contains helpers used by local experiment scripts that run the
Complementary Neyman SHAP explainer, compute metrics vs. exact baselines and
serialize run outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import secrets
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pandas as pd
import torch

from mllm_shap.connectors import LiquidAudio, ModelConfig
from mllm_shap.connectors.enums import (
    ModelHistoryTrackingMode,
    Role,
    SystemRolesSetup,
)
from mllm_shap.connectors.filters import ExcludePunctuationTokensFilter
from mllm_shap.shap import ComplementaryNeymanShapExplainer, PreciseShapExplainer
from mllm_shap.shap.normalizers import MinMaxNormalizer

# Ensure PyTorch allocation config is set before heavy imports if needed
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

NEYMAN_STAGE = "neyman"


def free_cuda(_tag: str = "") -> None:
    """Release Python refs and clear CUDA caches.

    The `_tag` argument is unused but kept for call-site compatibility.
    """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(f"torch.cuda.synchronize() failed: {exc}")
        torch.cuda.empty_cache()
        # NOTE: ipc_collect helps when many processes were used, harmless otherwise
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                getattr(torch.cuda, "ipc_collect")()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                warnings.warn(f"torch.cuda.ipc_collect() failed: {exc}")


def teardown_chat(chat: Any) -> None:
    """Best-effort removal of cached references on a chat object.

    This helps avoid accidental retention of large tensors between runs.
    """
    try:
        if hasattr(chat, "cache"):
            chat.cache = None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        warnings.warn(f"teardown_chat failed: {exc}")


def teardown_model(model: Any) -> None:
    """Attempt to clear model internals (processor/tokenizer) if present."""
    for attr in ("processor", "tokenizer"):
        try:
            obj = getattr(model, attr, None)
            if obj is not None and hasattr(obj, "clear"):
                obj.clear()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(f"teardown_model: clearing {attr} failed: {exc}")


def hard_kill(*objs: Any) -> None:
    """Delete provided objects and free CUDA resources."""
    for o in objs:
        try:
            del o
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(f"hard_kill: failed to del object: {exc}")
    free_cuda()


def build_model(device: torch.device, text_only: bool = True) -> Any:
    """Construct a LiquidAudio connector configured for text-only or text+audio."""
    mode = ModelHistoryTrackingMode.TEXT if text_only else ModelHistoryTrackingMode.TEXT_AUDIO
    return LiquidAudio(device=device, history_tracking_mode=mode)


def build_chat(model: Any, prompt: str, token_filter: Any = None, add_audio: bytes | None = None) -> Any:
    """Create and refresh a chat object with a system and user turn.

    The returned chat has had `refresh(full=True)` called and any temporary
    caches dropped (when applicable).
    """
    tf = token_filter or ExcludePunctuationTokensFilter()
    chat = model.get_new_chat(system_roles_setup=SystemRolesSetup.SYSTEM_ASSISTANT, token_filter=tf)
    chat.new_turn(Role.ASSISTANT)
    chat.add_text("You are a helpful assistant.")
    chat.end_turn()
    chat.new_turn(Role.USER)
    chat.add_text(prompt)
    if add_audio is not None:
        chat.add_audio(add_audio)
    chat.end_turn()
    chat.refresh(full=True)
    teardown_chat(chat)
    return chat


def exact_baseline(model: Any, chat: Any, generation_kwargs: Dict[str, Any]) -> torch.Tensor:
    """Compute an exact (precise) baseline normalized SHAP vector (CPU, float32).

    Returns the explainable-only normalized SHAP vector.
    """
    precise = PreciseShapExplainer(normalizer=MinMaxNormalizer())
    with torch.no_grad():
        resp = model.generate(chat=chat, keep_history=True, **generation_kwargs)
        _ = precise(
            model=model,
            source_chat=chat,
            response=resp,
            progress_bar=True,
            **generation_kwargs,
        )
    cache = resp.chat.cache
    vec = cache.normalized_values[: cache.n].detach().cpu().to(torch.float32).view(-1)
    teardown_chat(resp.chat)
    del precise, resp
    return cast(torch.Tensor, vec)


def clear_explainer_internal_caches(expl: Any) -> None:
    """Clear internal cached helper functions on explainer (best-effort)."""
    for name in ("_ComplementaryNeymanShapExplainer__get_start",):
        try:
            meth = getattr(expl, name)
            if hasattr(meth, "cache_clear"):
                meth.cache_clear()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(f"clear_explainer_internal_caches failed for {name}: {exc}")
    for name in ("_get_num_splits",):
        try:
            meth = getattr(expl, name)
            if hasattr(meth, "cache_clear"):
                meth.cache_clear()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            warnings.warn(f"clear_explainer_internal_caches failed for {name}: {exc}")


def run_neyman(
    model: Any,
    chat: Any,
    generation_kwargs: Dict[str, Any],
    *,
    use_standard_method: bool,
) -> List[Dict[str, Any]]:
    """Run ComplementaryNeymanShapExplainer and return the trajectory.

    This function returns a list of checkpoint dicts as produced by the explainer.
    """
    neyman = ComplementaryNeymanShapExplainer(
        fraction=1.0,
        normalizer=MinMaxNormalizer(),
        use_standard_method=use_standard_method,
    )
    clear_explainer_internal_caches(neyman)

    with torch.no_grad():
        resp = model.generate(chat=chat, keep_history=True, **generation_kwargs)
        # Cast to Dict because Mypy infers the explainer returns a List (default behavior)
        # but return_trajectory=True changes the return type to a Dict.
        out = cast(
            Dict[str, Any],
            neyman(
                model=model,
                source_chat=chat,
                response=resp,
                return_trajectory=True,
                progress_bar=True,
                **generation_kwargs,
            ),
        )
    traj = out["trajectory"]
    teardown_chat(resp.chat)
    del resp, neyman
    # Ensure Mypy knows this is the expected list structure
    return cast(List[Dict[str, Any]], traj)


def to_vec_cpu(x: Any) -> torch.Tensor:
    """Convert input to a 1-D CPU float32 tensor."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    # Explicit cast required because x is Any, making the chain return Any
    return cast(torch.Tensor, x.detach().cpu().to(torch.float32).view(-1))


def metrics_vs_baseline(  # pylint: disable=too-many-locals
    baseline: torch.Tensor, trajectory: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Compute per-checkpoint metrics comparing trajectory vectors to a baseline.

    Returns a DataFrame with columns: num_masks, stage, mae, rmae_pct,
    rmae_pct_masked, cosine, pearson, rel_l2, n_expl, phase_change_num_masks.
    """
    n = baseline.numel()
    eps, tau = 1e-8, 1e-4
    nz = baseline.abs() >= tau

    xs: List[int] = []
    stage: List[str] = []
    mae: List[float] = []
    rmae_pct: List[float] = []
    rmae_pct_masked: List[float] = []
    cosine: List[float] = []
    pearson: List[float] = []
    rel_l2: List[float] = []

    bn = baseline

    for step in trajectory:
        xs.append(int(step["num_masks"]))
        stage.append(str(step["stage"]))
        a = to_vec_cpu(step["normalized_shap"])
        m = min(n, a.numel())
        a = a[:m]
        b = bn[:m]
        nzm = nz[:m]
        valid = torch.isfinite(a) & torch.isfinite(b)
        av = a[valid]
        bv = b[valid]

        mae.append(float((a - b).abs().mean()))
        rmae_pct.append(float(100.0 * ((a - b).abs() / torch.clamp(b.abs(), min=eps)).mean()))
        rmae_pct_masked.append(
            float(100.0 * ((a[nzm] - b[nzm]).abs() / b[nzm].abs()).mean()) if nzm.any() else float("nan")
        )

        if av.numel() == 0:
            cosine.append(float("nan"))
            pearson.append(float("nan"))
            rel_l2.append(float("nan"))
        else:
            denom = (av.norm() * bv.norm()).item()
            cosine.append(float(torch.dot(av, bv) / (denom + eps)) if denom > 0 else float("nan"))
            a0, b0 = av - av.mean(), bv - bv.mean()
            denom_p = (a0.norm() * b0.norm()).item()
            pearson.append(float(torch.dot(a0, b0) / (denom_p + eps)) if denom_p > 0 else float("nan"))
            rel_l2.append(float((av - bv).norm() / (bv.norm().item() + eps)))

    df = pd.DataFrame(
        {
            "num_masks": xs,
            "stage": stage,
            "mae": mae,
            "rmae_pct": rmae_pct,
            "rmae_pct_masked": rmae_pct_masked,
            "cosine": cosine,
            "pearson": pearson,
            "rel_l2": rel_l2,
            "n_expl": n,
        }
    )
    first_neyman = next(
        (int(df.loc[i, "num_masks"]) for i in range(len(df)) if df.loc[i, "stage"] == NEYMAN_STAGE),
        None,
    )
    df["phase_change_num_masks"] = first_neyman
    return df


def main() -> None:  # pylint: disable=too-many-statements
    """Execute the Neyman stats experiment runner."""
    ap = argparse.ArgumentParser("Neyman SHAP statistical runner")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument(
        "--prompt",
        type=str,
        default="Explain photosynthesis occurrence in two sentences.",
    )
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="./neyman_runs")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = {
        "model_config": ModelConfig(text_temperature=float(args.temperature), text_top_k=int(args.top_k)),
        "max_new_tokens": int(args.max_new_tokens),
    }

    exp_id = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = out_root / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "meta.json").write_text(
        json.dumps(
            {
                "exp_id": exp_id,
                "runs": int(args.runs),
                "prompt": args.prompt,
                "gen": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                },
                "connector": ("LiquidAudio(TEXT)" if args.text_only else "LiquidAudio(TEXT_AUDIO)"),
                "methods": ["exact", "neyman_standard", "neyman_limited"],
            },
            indent=2,
        )
    )

    # Function to execute a Neyman variant; defined here to capture config
    def do_neyman(method_name: str, *, use_standard: bool, run_dir_path: Path, prompt_text: str) -> None:
        """Execute a specific Neyman configuration and save metrics."""
        mdir = run_dir_path / method_name
        (mdir / "vectors").mkdir(parents=True, exist_ok=True)

        model_inst = build_model(device=device, text_only=args.text_only)
        chat_inst = build_chat(model_inst, prompt=prompt_text)
        with torch.no_grad():
            traj = run_neyman(model_inst, chat_inst, gen, use_standard_method=use_standard)

        # save vectors and manifest
        rows: List[Dict[str, Any]] = []
        for step in traj:
            nm = int(step["num_masks"])
            vec = to_vec_cpu(step["normalized_shap"]).numpy()
            fn = f"vec_{nm:04d}.npy"
            np.save(mdir / "vectors" / fn, vec)
            rows.append({"num_masks": nm, "stage": step["stage"], "path": f"vectors/{fn}"})

        pd.DataFrame(rows).sort_values("num_masks").to_csv(mdir / "trajectory.csv", index=False)

        # also compute metrics vs exact of THIS run
        base = np.load(run_dir_path / "baseline_exact.npy")
        dfm = metrics_vs_baseline(torch.from_numpy(base), traj)
        dfm.to_csv(mdir / "metrics_vs_exact.csv", index=False)

        # cleanup hard
        teardown_chat(chat_inst)
        teardown_model(model_inst)
        hard_kill(model_inst, chat_inst, traj, rows)

    for r in range(args.runs):
        # fresh, cryptographically-sourced seeds per run
        seed = int(secrets.randbits(30))
        torch.manual_seed(seed)
        np.random.seed(seed)

        run_id = f"run_{r:03d}_{uuid.uuid4().hex[:8]}"
        run_dir = exp_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"➡️  Starting run {r + 1}/{args.runs}: {run_id}")

        # exact baseline
        model = build_model(device=device, text_only=args.text_only)
        chat = build_chat(model, prompt=args.prompt)
        with torch.no_grad():
            baseline = exact_baseline(model, chat, gen)
        np.save(run_dir / "baseline_exact.npy", baseline.numpy())

        # cleanup everything used by baseline
        teardown_chat(chat)
        teardown_model(model)
        hard_kill(model, chat, baseline)

        # standard and limited neyman
        do_neyman(
            "neyman_standard",
            use_standard=True,
            run_dir_path=run_dir,
            prompt_text=args.prompt,
        )
        do_neyman(
            "neyman_limited",
            use_standard=False,
            run_dir_path=run_dir,
            prompt_text=args.prompt,
        )

        (run_dir / "meta.json").write_text(json.dumps({"run_id": run_id, "prompt": args.prompt}, indent=2))
        hard_kill()


if __name__ == "__main__":
    main()

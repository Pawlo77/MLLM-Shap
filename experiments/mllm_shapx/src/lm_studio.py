"""LM Studio model lifecycle manager: download, load, configure, and unload."""

import logging
import os
import platform
import time
import torch
from dataclasses import dataclass, field
from typing import Any, Sequence
from lmstudio import LlmLoadModelConfig, LMStudioServerError
from lmstudio._sdk_models import GpuSetting
from lmstudio import llm as lmstudio_llm

LOGGER = logging.getLogger(__name__)

_CUDA_COMPAT_TYPE: str = "gguf"
"""Compatibility type for CUDA-compatible models, typically in GGUF format."""


@dataclass
class LmStudioConfig:
    """Configuration for LM Studio managed model lifecycle.

    Attributes:
        model_key: Model identifier (e.g. 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF').
            If the model is not downloaded, it will be searched and downloaded.
        context_length: Context window size. If None, computed as
            max_prompt_tokens + context_length_gap.
        context_length_gap: Extra tokens on top of max prompt tokens for context window.
        max_concurrency: Maximum parallel inference requests (maps to eval_batch_size).
        seed: RNG seed for reproducible inference.
        gpu_offload: Fraction of model to offload to GPU (0.0-1.0 or 'max').
            On CUDA defaults to 'max' (full offload). On Mac/MLX ignored.
        cpu_threads: Number of CPU threads for inference.
            On CUDA defaults to (os.cpu_count() - 2).
        flash_attention: Enable flash attention if available.
        ttl: Time-to-live in seconds for the loaded model instance.
        quantization_preference: Preferred quantization level for model search/download.
            On Mac defaults to '4bit', on CUDA defaults to '4bit'.
        api_host: LM Studio API host. None uses SDK default (localhost).
        keep_model_in_memory: Keep model weights pinned in RAM.
    """

    model_key: str
    """Model identifier (e.g. 'lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF')."""
    context_length: int | None = None
    """Context window size. None computes from max_prompt_tokens + context_length_gap."""
    context_length_gap: int = 64
    """Extra tokens on top of max prompt tokens for context window sizing."""
    max_concurrency: int = 1
    """Maximum parallel inference requests (maps to eval_batch_size)."""
    seed: int | None = None
    """RNG seed for reproducible inference. None uses non-deterministic."""
    gpu_offload: (float | str) | None = None
    """Fraction of model to offload to GPU (0.0-1.0 or 'max'). None uses platform default."""
    cpu_threads: int | None = None
    """Number of CPU threads for inference. None uses platform default."""
    flash_attention: bool = True
    """Enable flash attention if available."""
    ttl: int | None = 3600
    """Time-to-live in seconds for the loaded model instance."""
    quantization_preference: str | None = None
    """Preferred quantization level for model search/download (e.g. '4bit')."""
    api_host: str | None = None
    """LM Studio API host. None uses SDK default (localhost)."""
    keep_model_in_memory: bool = True
    """Keep model weights pinned in RAM between requests."""


def _is_mac() -> bool:
    """Detect if running on macOS. Used to determine compatibility types and defaults."""
    return platform.system() == "Darwin"


def _is_cuda_available() -> bool:
    """Detect if CUDA is available via PyTorch. Used to determine compatibility types and defaults."""
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


def _default_cpu_threads() -> int:
    """Default CPU thread count for LM Studio inference on CUDA systems."""
    n = os.cpu_count() or 4
    return max(1, n - 2)


def _resolve_compatibility_types(cfg: LmStudioConfig) -> list[str] | None:
    """Determine model compatibility types to search for based on platform."""
    if _is_cuda_available():
        return [_CUDA_COMPAT_TYPE]
    return None


def _pick_download_option(options: Sequence[Any], cfg: LmStudioConfig) -> Any | None:
    """Select best download option based on quantization preference."""
    if not options:
        return None
    if len(options) == 1:
        return options[0]

    pref = cfg.quantization_preference
    if pref is None:
        pref = "4bit" if _is_mac() else "4bit"

    # Try to match quantization in option info
    pref_lower = pref.lower().replace("-", "").replace("_", "")
    for opt in options:
        info = getattr(opt, "info", None)
        if info is None:
            continue
        info_str = str(info).lower().replace("-", "").replace("_", "")
        if pref_lower in info_str or "q4" in info_str:
            return opt

    # Fallback: first option
    return options[0]


def _build_load_config(cfg: LmStudioConfig) -> dict[str, Any]:
    """Build LlmLoadModelConfig kwargs from our config."""
    load_kwargs: dict[str, Any] = {}

    if cfg.context_length is not None:
        load_kwargs["context_length"] = cfg.context_length

    if cfg.seed is not None:
        load_kwargs["seed"] = cfg.seed

    load_kwargs["flash_attention"] = cfg.flash_attention
    load_kwargs["keep_model_in_memory"] = cfg.keep_model_in_memory

    if cfg.max_concurrency > 1:
        load_kwargs["eval_batch_size"] = cfg.max_concurrency

    # GPU settings
    if _is_cuda_available():
        gpu_ratio = cfg.gpu_offload if cfg.gpu_offload is not None else "max"
        gpu_setting = GpuSetting(ratio=gpu_ratio)
        load_kwargs["gpu"] = gpu_setting

    config = LlmLoadModelConfig(**load_kwargs)
    return {"config": config}


def _build_prediction_config(cfg: LmStudioConfig) -> dict[str, Any]:
    """Build LlmPredictionConfig kwargs for inference calls."""
    pred_kwargs: dict[str, Any] = {}

    cpu_threads = cfg.cpu_threads
    if cpu_threads is None and _is_cuda_available():
        cpu_threads = _default_cpu_threads()

    if cpu_threads is not None:
        pred_kwargs["cpu_threads"] = cpu_threads

    return pred_kwargs


@dataclass
class LmStudioManager:
    """Manages download, load, and unload of a model via LM Studio SDK.

    Usage:
        mgr = LmStudioManager(cfg)
        mgr.ensure_downloaded()
        mgr.load()
        # ... use the model via OpenAI-compat API ...
        mgr.unload()
    """

    cfg: LmStudioConfig

    _client: Any = field(default=None, init=False, repr=False)
    """LM Studio Client instance, initialized lazily on first use."""
    _model_handle: Any = field(default=None, init=False, repr=False)
    """Handle to the loaded model, initialized lazily on first use."""
    _loaded_model_key: str | None = field(default=None, init=False, repr=False)
    """Key of the loaded model, initialized lazily on first use."""

    def _get_client(self) -> Any:
        """Lazily initialize and return the LM Studio Client."""
        if self._client is None:
            from lmstudio import Client

            self._client = Client(api_host=self.cfg.api_host)
        return self._client

    def ensure_downloaded(self) -> str:
        """Ensure model is downloaded locally. Returns the model key.

        If model_key matches an already-downloaded model, skips download.
        Otherwise searches the repository and downloads the best match.
        """
        client = self._get_client()

        # Check if already downloaded
        downloaded = client.list_downloaded_models(namespace="llm")
        for model in downloaded:
            if self.cfg.model_key in (model.model_key, model.path):
                LOGGER.info("Model '%s' already downloaded.", self.cfg.model_key)
                self._loaded_model_key = model.model_key
                return model.model_key

        # Search and download
        LOGGER.info("Searching for model '%s' in repository...", self.cfg.model_key)
        compat_types = _resolve_compatibility_types(self.cfg)

        results = client.repository.search_models(
            search_term=self.cfg.model_key,
            limit=5,
            compatibility_types=compat_types,
        )

        if not results:
            raise RuntimeError(
                f"No models found matching '{self.cfg.model_key}' "
                f"with compatibility types {compat_types}."
            )

        # Pick best result and download option
        best_result = results[0]
        options = best_result.get_download_options()
        chosen = _pick_download_option(options, self.cfg)

        if chosen is None:
            raise RuntimeError(
                f"No download options available for model '{self.cfg.model_key}'."
            )

        LOGGER.info("Downloading model (option: %s)...", getattr(chosen, "info", "?"))

        _last_logged_pct: list[float] = [0.0]

        def _on_progress(update: Any) -> None:
            downloaded = getattr(update, "downloaded_bytes", 0)
            total = getattr(update, "total_bytes", 0)
            speed = getattr(update, "speed_bytes_per_second", 0.0)
            if total > 0:
                pct = downloaded / total * 100
                # Log every 10% to avoid spam
                if pct - _last_logged_pct[0] >= 10.0 or pct >= 99.9:
                    _last_logged_pct[0] = pct
                    speed_mb = speed / (1024 * 1024)
                    LOGGER.info(
                        "Download progress: %.0f%% (%.0f/%.0f MB, %.1f MB/s)",
                        pct,
                        downloaded / (1024 * 1024),
                        total / (1024 * 1024),
                        speed_mb,
                    )

        def _on_finalize() -> None:
            LOGGER.info("Download finalizing (verifying files)...")

        # The SDK default per-message timeout is 60s which is too short for
        # large model downloads (slow network can have >60s gaps between
        # progress messages). Temporarily disable the timeout.
        import lmstudio.sync_api as _sync_api

        _orig_timeout = _sync_api._DEFAULT_TIMEOUT
        _sync_api._DEFAULT_TIMEOUT = None  # type: ignore[assignment]
        try:
            model_key = chosen.download(
                on_progress=_on_progress, on_finalize=_on_finalize
            )
        except LMStudioServerError as exc:
            # Download already in progress (e.g. started via UI) — poll until done
            if "already" in str(exc).lower() or "cannot find" in str(exc).lower():
                LOGGER.info(
                    "Download appears already in progress, waiting for completion..."
                )
                model_key = self._poll_until_downloaded(client)
            else:
                raise
        finally:
            _sync_api._DEFAULT_TIMEOUT = _orig_timeout
        LOGGER.info("Model downloaded: %s", model_key)
        self._loaded_model_key = model_key
        return model_key

    def _poll_until_downloaded(
        self, client: Any, poll_interval: float = 5.0, timeout: float = 3600.0
    ) -> str:
        """Poll list_downloaded_models until the target model appears."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            downloaded = client.list_downloaded_models(namespace="llm")
            for model in downloaded:
                if self.cfg.model_key in (model.model_key, model.path):
                    return model.model_key
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Timed out waiting for model '{self.cfg.model_key}' download "
            f"to complete after {timeout:.0f}s."
        )

    def load(self) -> Any:
        """Load the model into LM Studio. Returns the LLM handle."""
        if self._model_handle is not None:
            LOGGER.warning("Model already loaded, returning existing handle.")
            return self._model_handle

        model_key = self._loaded_model_key or self.cfg.model_key
        load_kwargs = _build_load_config(self.cfg)
        load_kwargs["ttl"] = self.cfg.ttl

        LOGGER.info(
            "Loading model '%s' into LM Studio (context_length=%s, seed=%s)...",
            model_key,
            self.cfg.context_length,
            self.cfg.seed,
        )

        self._model_handle = lmstudio_llm(model_key, **load_kwargs)
        LOGGER.info("Model loaded successfully.")
        return self._model_handle

    def unload(self) -> None:
        """Unload the model from LM Studio, freeing resources."""
        if self._model_handle is None:
            LOGGER.debug("No model handle to unload.")
            return

        LOGGER.info("Unloading model from LM Studio...")
        try:
            self._model_handle.unload()
        except Exception as exc:
            LOGGER.warning("Error unloading model: %s", exc)
        finally:
            self._model_handle = None

    def close(self) -> None:
        """Unload model and close client connection."""
        self.unload()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def api_base_url(self) -> str:
        """Return the OpenAI-compatible base URL for the loaded model."""
        host = self.cfg.api_host or "127.0.0.1:1234"
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return f"{host.rstrip('/')}/v1"

    @property
    def model_identifier(self) -> str:
        """Return the model identifier for OpenAI-compat API calls."""
        return self._loaded_model_key or self.cfg.model_key

    @property
    def prediction_config(self) -> dict[str, Any]:
        """Return prediction config kwargs for LM Studio inference."""
        return _build_prediction_config(self.cfg)

    def __enter__(self) -> "LmStudioManager":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def build_lm_studio_config(
    experiment_cfg: Any,
    max_prompt_tokens: int | None = None,
) -> LmStudioConfig:
    """Build LmStudioConfig from experiment set's lm_studio section.

    Args:
        experiment_cfg: The ExperimentSet pydantic model (or its lm_studio sub-config).
        max_prompt_tokens: If provided and context_length is not set,
            computes context_length = max_prompt_tokens + gap.
    """
    lms_cfg = experiment_cfg.lm_studio

    context_length = lms_cfg.context_length
    if context_length is None and max_prompt_tokens is not None:
        # Add gap for generation output + safety margin
        gap = lms_cfg.context_length_gap
        gen_tokens = experiment_cfg.generation.max_new_tokens
        context_length = max_prompt_tokens + gen_tokens + gap

    seed = lms_cfg.seed
    if seed is None:
        seed = experiment_cfg.selection.shuffle_seed

    return LmStudioConfig(
        model_key=lms_cfg.model_key,
        context_length=context_length,
        context_length_gap=lms_cfg.context_length_gap,
        max_concurrency=lms_cfg.max_concurrency,
        seed=seed,
        gpu_offload=lms_cfg.gpu_offload,
        cpu_threads=lms_cfg.cpu_threads,
        flash_attention=lms_cfg.flash_attention,
        ttl=lms_cfg.ttl,
        quantization_preference=lms_cfg.quantization_preference,
        api_host=lms_cfg.api_host,
        keep_model_in_memory=lms_cfg.keep_model_in_memory,
    )

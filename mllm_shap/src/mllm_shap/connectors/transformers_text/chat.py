"""Chat state for text-only Transformers causal models."""

import warnings
from copy import deepcopy
from functools import cached_property

import torch
from torch import Tensor

from transformers import PreTrainedTokenizerBase

from ...utils.other import safe_mask
from ..base.chat import BaseMllmChat
from ..base.filters import TokenFilter
from ..enums import ModalityFlag, ModelHistoryTrackingMode, Role, SystemRolesSetup


class TransformersTextChat(BaseMllmChat):
    """
    Chat state for text-only causal LMs.

    Stores only TEXT token IDs. AUDIO is unsupported and will warn+no-op.
    """

    tokenizer: PreTrainedTokenizerBase
    _text_ids: Tensor
    _TWO_DIMS: int = 2
    _SINGLE_BATCH: int = 1
    _SHARED_ATTRIBUTES: frozenset[str] = frozenset({
        "tokenizer",  # Large read-only object, safe to share across copies
    })

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        device: torch.device,
        tokenizer: PreTrainedTokenizerBase,
        empty_turn_sequences: set[str] | None = None,
        token_filter: TokenFilter | None = None,
        system_roles_setup: SystemRolesSetup | None = None,
    ) -> None:
        empty_turn_sequences = empty_turn_sequences or set()
        self.tokenizer = tokenizer
        super().__init__(
            device=device,
            empty_turn_sequences=empty_turn_sequences,
            token_filter=token_filter,
            system_roles_setup=system_roles_setup,
        )
        self._text_ids = torch.empty(0, dtype=torch.long, device=device)

    def apply_text_mask(self, text_mask_relative: Tensor) -> None:
        """Apply a relative text mask to this chat instance (public on purpose)."""
        self._text_ids = safe_mask(self._text_ids, text_mask_relative)
        self.text_tokens_no_system_mask = safe_mask(
            self.text_tokens_no_system_mask, text_mask_relative
        )

    @classmethod
    def _set_new_instance(
        cls,
        full_mask: Tensor,
        text_mask_relative: Tensor,
        audio_mask_relative: Tensor,   # unused (no audio)
        chat: "TransformersTextChat",  # type: ignore[override]
    ) -> "TransformersTextChat":
        new_instance: "TransformersTextChat" = deepcopy(chat)
        new_instance.apply_text_mask(text_mask_relative)

        # full input token mask affects only text here
        # token_roles/turns are handled in BaseMllmChat._after_add via refresh()

        return new_instance

    @cached_property
    def input_tokens(self) -> list[Tensor]:
        # One Tensor per token, each shaped [1], to match BaseMllmChat expectations.
        return [tid.unsqueeze(0) for tid in self._text_ids]

    @cached_property
    def tokens_modality_flag(self) -> Tensor:
        # All tokens are TEXT
        return torch.full(
            (self._text_ids.shape[0],),
            ModalityFlag.TEXT,
            dtype=torch.long,
            device=self.torch_device,
        )

    @cached_property
    def text_tokens(self) -> Tensor:
        return self._text_ids

    @cached_property
    def audio_tokens(self) -> Tensor:
        # No audio tokens in this connector
        return torch.empty(0, dtype=torch.long, device=self.torch_device)

    def _decode_text(self, text_tokens: Tensor) -> str:
        # Accept shape [T] or [1]; tokenizer handles lists
        ids = text_tokens.detach().to("cpu").tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _decode_audio(self, audio_tokens: Tensor) -> Tensor | None:  # pragma: no cover - unsupported
        return None  # decoding audio is impossible here

    def _add_text(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return 0
        ids_t = torch.tensor(ids, dtype=torch.long, device=self.torch_device)
        start = self._text_ids.shape[0]
        self._text_ids = torch.cat([self._text_ids, ids_t], dim=0)
        return int(self._text_ids.shape[0] - start)

    def _add_audio(self, waveform: Tensor, sample_rate: int) -> int:  # pragma: no cover
        warnings.warn(
            "Audio input is not supported by the TransformersText connector. Ignoring provided audio.",
            stacklevel=2,
        )
        return 0

    def _append(
        self,
        text: Tensor,
        audio_out: Tensor,              # ignored for text-only
        modality_flag: Tensor,          # ignored for text-only
        history_tracking_mode: ModelHistoryTrackingMode,
    ) -> tuple[int, int]:
        if history_tracking_mode == ModelHistoryTrackingMode.AUDIO:
            warnings.warn(
                "Requested AUDIO-only history tracking is not supported by the text-only connector. "
                "No tokens appended.",
                stacklevel=2,
            )
            return 0, 0

        # Expect [1, T]; accept other simple shapes
        if text.dim() == self._TWO_DIMS and text.shape[0] == self._TWO_DIMS:
            text = text.squeeze(0)
        elif text.dim() == 0:
            text = text.unsqueeze(0)
        if text.dim() != 1:
            text = text.reshape(-1)

        text = text.to(dtype=torch.long, device=self.torch_device)
        start = self._text_ids.shape[0]
        self._text_ids = torch.cat([self._text_ids, text], dim=0)

        # IMPORTANT: In text-only path there are no audio tokens; BaseMllmChat won’t refresh caches.
        # Do it here to keep tokens_modality_flag / input_tokens in sync with masks.
        self.refresh(full=True)

        return int(self._text_ids.shape[0] - start), 0

    def _new_turn(self, speaker: Role) -> None:
        # No special turn markers injected for generic causal LMs.
        # Token accounting is handled in Base via _after_add.
        return

    def _end_turn(self) -> None:
        return

    def _get_tokens_sequences_to_exclude(self, phrases_to_exclude: set[str]) -> list[Tensor]:
        seqs: list[Tensor] = []
        for phrase in phrases_to_exclude:
            ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            seqs.append(torch.tensor(ids, dtype=torch.long, device=self.torch_device))
        return seqs

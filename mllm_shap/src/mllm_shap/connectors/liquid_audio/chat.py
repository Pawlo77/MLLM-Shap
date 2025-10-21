"""LiquidAudio chat state."""

from copy import deepcopy
from functools import cached_property
from logging import Logger
from typing import Any, Literal, cast

import torch
from liquid_audio import ChatState, LFMModality
from torch import Tensor

from ...utils.logger import get_logger
from ...utils.other import safe_mask
from .._base.chat import BaseChat
from ..enums import ModalityFlag, ModelHistoryTrackingMode, Role

logger: Logger = get_logger(__name__)


class LiquidAudioChat(BaseChat, ChatState):  # type: ignore[misc]
    """
    Chat state for LiquidAudio model.

    Fields:
        audio_empty_value: Value representing empty audio token.
        validate_from_chat: Whether to validate chat state when creating new instances.

        START_MARK: Token marking the start of text.
        EMPTY_SYSTEM_TURN: Token sequence for empty system turn.
        EMPTY_ASSISTANT_TURN: Token sequence for empty assistant turn.
        EMPTY_USER_TURN: Token sequence for empty user turn.
        AUDIO_IN_SHAPE: Number of audio in tokens.
        AUDIO_OUT_SHAPE: Number of audio out tokens.
    """

    audio_empty_value: float = torch.finfo(torch.float32).min
    validate_from_chat: bool

    START_MARK: str = "<|startoftext|>"
    EMPTY_SYSTEM_TURN: str = "<|im_start|>Role.SYSTEM\n<|im_end|>\n"
    EMPTY_ASSISTANT_TURN: str = "<|im_start|>Role.ASSISTANT\n<|im_end|>\n"
    EMPTY_USER_TURN: str = "<|im_start|>user\n<|im_end|>\n"

    AUDIO_IN_SHAPE: int = 128
    AUDIO_OUT_SHAPE: int = 8

    # for each element x in _audio_map:
    #     x > 0 -> index in audio_out + 1
    #     x < 0 -> -(index in audio_in + 1)
    _audio_map: Tensor

    # relies on ChatState.text, ChatState.audio_in, ChatState.audio_out, ChatState.modality_flag
    # both audio are in  (K, T) format

    def __init__(
        self,
        *args: Any,
        device: torch.device,
        liquid_kwargs: dict[str, Any] | None = None,
        validate_from_chat: bool = False,  # for testing purposes
        **kwargs: Any,
    ) -> None:
        ChatState.__init__(self, **(liquid_kwargs or {}))

        empty_turn_sequences = {
            LiquidAudioChat.EMPTY_SYSTEM_TURN,
            LiquidAudioChat.EMPTY_ASSISTANT_TURN,
            LiquidAudioChat.EMPTY_USER_TURN,
        }
        # Consider empty turns with start mark as well
        for e in empty_turn_sequences.copy():
            empty_turn_sequences.add(LiquidAudioChat.START_MARK + e)

        added_vocab_tokens = set(self.proc.text.get_added_vocab().values())

        added_vocab_tokens = added_vocab_tokens.union(kwargs.pop("added_vocab_tokens", set()))
        empty_turn_sequences = empty_turn_sequences.union(kwargs.pop("empty_turn_sequences", set()))
        BaseChat.__init__(
            self,
            *args,
            added_vocab_tokens=added_vocab_tokens,
            device=device,
            empty_turn_sequences=empty_turn_sequences,
            **kwargs,
        )  # type: ignore[misc]

        # mark starting tokens as system
        self.speaker = Role.SYSTEM
        self._after_add(1, text_added=True, refresh=True)
        self.speaker = None

        self.validate_from_chat = validate_from_chat
        self._audio_map = torch.empty((0,), dtype=torch.long, device=self.torch_device)

    # assume `_{}` are protected methods from BaseChat
    # pylint: disable=too-many-locals,protected-access
    @classmethod
    def _set_new_instance(
        cls: type["LiquidAudioChat"],
        full_mask: Tensor,
        text_mask_relative: Tensor,
        audio_mask_relative: Tensor,
        chat: "LiquidAudioChat",  # type: ignore[override]
    ) -> "LiquidAudioChat":
        new_instance: "LiquidAudioChat" = deepcopy(chat)

        # filter out text tokens based on the text_mask
        # masking done on new_instance as it can mutate the tensors
        new_instance.text = safe_mask(new_instance.text, text_mask_relative)
        new_instance.text_tokens_no_system_mask = safe_mask(new_instance.text_tokens_no_system_mask, text_mask_relative)

        # split audio mask into input and output parts
        # masks relative to audio tokens
        # this is calculated before filtering out audio tokens
        audio_in_mask_relative, audio_out_mask_relative = chat._get_relative_audio_masks()

        # audio map is a list of indices in audio_in and audio_out
        # after removing some audio tokens, we need to update the audio map accordingly
        final_audio_in_relative = audio_mask_relative[audio_in_mask_relative]
        final_audio_out_relative = audio_mask_relative[audio_out_mask_relative]

        # calculate index shifts due to removed tokens, make it
        # relative to new audio map and token type (i.e., audio in or out)
        removed_audio_in_relative_shift = torch.cumsum((~final_audio_in_relative).to(torch.long), dim=0)[
            final_audio_in_relative
        ]
        removed_audio_out_relative_shift = torch.cumsum((~final_audio_out_relative).to(torch.long), dim=0)[
            final_audio_out_relative
        ]

        # pick < 0 --> audio in, by final_audio_in_relative - what to keep, and adjust indices
        new_audio_map_in = (
            chat._audio_map[chat._audio_map < 0][final_audio_in_relative] + removed_audio_in_relative_shift
        )
        # pick > 0 --> audio out, by final_audio_out_relative - what to keep, and adjust indices
        new_audio_map_out = (
            chat._audio_map[chat._audio_map > 0][final_audio_out_relative] - removed_audio_out_relative_shift
        )

        new_instance._audio_map = torch.cat([new_audio_map_in, new_audio_map_out], dim=0)

        # repeat each audio in token LiquidAudioChat.AUDIO_OUT_SHAPE times for LiquidAudioChat.AUDIO_OUT_SHAPE codebooks
        final_audio_in_relative_codebooks = torch.repeat_interleave(
            final_audio_in_relative,
            repeats=LiquidAudioChat.AUDIO_OUT_SHAPE,
        )
        new_instance.audio_in = safe_mask(
            new_instance.audio_in,
            final_audio_in_relative_codebooks,
        )
        new_instance.audio_out = safe_mask(new_instance.audio_out, final_audio_out_relative)

        # update audio in lens - tensor of audio tokens (codebook - aware)
        # for each added audio sample
        s = 0
        for i in range(new_instance.audio_in_lens.shape[0]):
            original_len = int(new_instance.audio_in_lens[i].item())
            # count how many audio in tokens are kept for this sample
            kept_tokens = final_audio_in_relative_codebooks[s : s + original_len].sum().item()  # noqa: E203
            new_instance.audio_in_lens[i] = kept_tokens
            s += original_len
        new_instance.audio_in_lens = new_instance.audio_in_lens[new_instance.audio_in_lens > 0]

        # safety checks
        if chat.validate_from_chat:
            if new_instance._audio_map.shape[0] != audio_mask_relative.sum().item():
                raise ValueError("audio_map shape does not match the number of audio tokens after filtering.")
            if new_audio_map_in.shape[0] != final_audio_in_relative.sum().item():
                raise ValueError("audio_map shape does not match the number of audio in tokens after filtering.")
            if new_audio_map_out.shape[0] != final_audio_out_relative.sum().item():
                raise ValueError("audio_map shape does not match the number of audio out tokens after filtering.")
            indices = -new_instance._audio_map[new_instance._audio_map < 0] - 1
            if indices.numel() > 0 and indices.max() >= new_instance.audio_in.shape[1]:
                raise ValueError("audio_in indices in audio_map are out of bounds after filtering.")
            indices = new_instance._audio_map[new_instance._audio_map > 0] - 1
            if indices.numel() > 0 and indices.max() >= new_instance.audio_out.shape[1]:
                raise ValueError("audio_out indices in audio_map are out of bounds after filtering.")

        # filter out modality flag based on masks
        new_instance.modality_flag = safe_mask(new_instance.modality_flag, full_mask)

        return new_instance

    @cached_property
    def input_tokens(self) -> list[Tensor]:
        # Move everything to CPU and convert to lists
        text_mask = self.text_tokens_mask
        audio_mask = self.audio_tokens_mask

        # Total number of tokens
        total_len = len(text_mask)
        result: list[Tensor] = [torch.empty(0)] * total_len

        a_idx = t_idx = 0
        for i, is_audio in enumerate(audio_mask):
            if is_audio:
                token_idx = self.audio_tokens[a_idx]
                if token_idx < 0:  # audio in
                    result[i] = self.audio_in[..., -token_idx - 1].unsqueeze(-1)
                else:  # audio out
                    result[i] = self.audio_out[..., token_idx - 1].unsqueeze(-1)
                a_idx += 1
            else:
                result[i] = self.text_tokens[t_idx].unsqueeze(-1)
                t_idx += 1

        return result

    @cached_property
    def tokens_modality_flag(self) -> Tensor:
        modality_flag = torch.full_like(self.modality_flag[0], ModalityFlag.AUDIO)
        modality_flag[self.modality_flag[0] == LFMModality.TEXT] = ModalityFlag.TEXT
        return modality_flag

    @cached_property
    def text_tokens(self) -> Tensor:
        return cast(Tensor, self.text[0])

    @cached_property
    def audio_tokens(self) -> Tensor:  # return audio tokens map
        return self._audio_map

    def _decode_text(self, text_tokens: Tensor) -> str:
        return self.proc.text.decode(text_tokens)

    def _decode_audio(self, audio_tokens: Tensor) -> Tensor | None:
        if len(audio_tokens.shape) == 1:
            logger.debug("Decoding audio tokens based on indices from _audio_map.")

            sign = torch.sign(audio_tokens)
            if sign.all():  # audio in
                audio_tokens = self.audio_in[audio_tokens - 1]
            elif not sign.any():  # audio out
                audio_tokens = self.audio_out[-audio_tokens - 1]
            else:
                raise ValueError("audio_tokens should contain either only audio in or only audio out tokens.")

        # input tokens
        if audio_tokens.shape[0] == LiquidAudioChat.AUDIO_IN_SHAPE:
            logger.debug("Decoding audio in...")
            # logger.warning("Decoding audio in tokens is not supported.")
            return None
        # audio out tokens
        if audio_tokens.shape[0] == LiquidAudioChat.AUDIO_OUT_SHAPE:
            logger.debug("Decoding audio out...")

            # expects shape (B, K, T)
            mimi_codes = audio_tokens.unsqueeze(0)
            return cast(Tensor, self.proc.mimi.decode(mimi_codes).squeeze(0))

        raise ValueError(
            f"audio tokens first dimension should be either {LiquidAudioChat.AUDIO_OUT_SHAPE} "
            f"(audio out) or {LiquidAudioChat.AUDIO_IN_SHAPE} (audio in)."
        )

    def _add_text(self, text: str) -> int:
        starting_tokens_num = self.text.shape[1]
        ChatState.add_text(self, text)
        return int(self.text.shape[1] - starting_tokens_num)

    def _add_audio(self, waveform: Tensor, sample_rate: int) -> int:
        starting_tokens_num = self.audio_in.shape[1]
        ChatState.add_audio(self, waveform, sample_rate)
        # LiquidAudioChat.AUDIO_OUT_SHAPE codebooks
        added_tokens_num = int(self.audio_in.shape[1] - starting_tokens_num) // LiquidAudioChat.AUDIO_OUT_SHAPE

        # update audio map
        self._audio_map = torch.cat(
            [
                self._audio_map,
                -(
                    torch.arange(
                        starting_tokens_num // LiquidAudioChat.AUDIO_OUT_SHAPE,
                        starting_tokens_num // LiquidAudioChat.AUDIO_OUT_SHAPE + added_tokens_num,
                        dtype=torch.long,
                        device=self.torch_device,
                    )
                    + 1
                ),
            ],
            dim=0,
        )

        return added_tokens_num

    def _append(
        self,
        text: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
        history_tracking_mode: ModelHistoryTrackingMode,
    ) -> tuple[int, int]:
        starting_text_tokens_num = self.text[0].shape[0]
        starting_audio_tokens_num = self.audio_out[0].shape[0]

        if history_tracking_mode == ModelHistoryTrackingMode.TEXT:
            audio_out = torch.empty((self.codebooks, 0), dtype=audio_out.dtype, device=audio_out.device)
            modality_flag = modality_flag[modality_flag == LFMModality.TEXT].unsqueeze(0)
        elif history_tracking_mode == ModelHistoryTrackingMode.AUDIO:
            text = torch.empty((1, 0), dtype=text.dtype, device=text.device)
            modality_flag = modality_flag[modality_flag != LFMModality.TEXT].unsqueeze(0)

        # else: keep both text and audio_out as is
        ChatState.append(self, text, audio_out, modality_flag)

        # update audio map
        self._audio_map = torch.cat(
            [
                self._audio_map,
                torch.arange(
                    starting_audio_tokens_num,
                    self.audio_out.shape[1],
                    dtype=torch.long,
                    device=self.torch_device,
                )
                + 1,
            ],
            dim=0,
        )

        return (
            self.text[0].shape[0] - starting_text_tokens_num,
            self.audio_out[0].shape[0] - starting_audio_tokens_num,
        )

    def _new_turn(self, speaker: Role) -> None:
        role: Literal["system", "user", "assistant"]
        if speaker == Role.SYSTEM:
            role = "system"
        elif speaker == Role.USER:
            role = "user"
        else:  # Role.ASSISTANT
            role = "assistant"
        ChatState.new_turn(self, role)

    def _end_turn(self) -> None:
        ChatState.end_turn(self)

    def _get_tokens_sequences_to_exclude(self, phrases_to_exclude: set[str]) -> list[Tensor]:
        token_sequences_to_exclude: list[Tensor] = []
        for phrase in phrases_to_exclude:
            token_ids = self.proc.text.encode(phrase, add_special_tokens=False)
            token_sequences_to_exclude.append(torch.tensor(token_ids, device=self.torch_device))
        return token_sequences_to_exclude

    def _get_relative_audio_masks(self) -> tuple[Tensor, Tensor]:
        """
        Get relative audio in and out masks based on the modality flag
        (relative to audio tokens only).

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the relative audio in mask and audio out mask
        """

        audio_in_mask = self.modality_flag[0] == LFMModality.AUDIO_IN
        audio_out_mask = self.modality_flag[0] == LFMModality.AUDIO_OUT
        audio_mask = audio_in_mask | audio_out_mask

        audio_in_mask_relative = audio_mask[audio_mask].clone()
        audio_in_mask_relative[audio_out_mask[audio_mask]] = False

        audio_out_mask_relative = audio_mask[audio_mask].clone()
        audio_out_mask_relative[audio_in_mask[audio_mask]] = False

        return audio_in_mask_relative, audio_out_mask_relative

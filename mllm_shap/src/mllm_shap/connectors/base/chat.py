# pylint: disable=too-many-lines
"""Base class for chat state management."""

from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from logging import Logger
from typing import Any, cast

import torch
from torch import Tensor

from mllm_shap.utils.audio import TorchAudioHandler

from ...utils.logger import get_logger
from ...utils.other import raise_connector_error
from ..enums import ModalityFlag, ModelHistoryTrackingMode, Role, SystemRolesSetup
from ..filters import KeepAllTokens
from ._validators import BaseChatConfig
from .chat_entry import ChatEntry
from .explainer_cache import ExplainerCache
from .filters import TokenFilter
from .model_response import ModelResponse

logger: Logger = get_logger(__name__)


class AllTextTokensFilteredOutError(ValueError):
    """Raised when all text tokens are filtered out from the chat."""


# pylint: disable=too-many-public-methods,too-many-instance-attributes
class BaseMllmChat(ABC):
    """Base class for chat state management."""

    token_filter: TokenFilter
    """The token filtering strategy."""
    # renamed from device not to conflict with MRO
    # as some chats can have device property/method
    torch_device: torch.device
    """The device on which tensors are stored."""
    system_roles_setup: SystemRolesSetup
    """
    A set of roles that are considered system roles. If Role.ASSISTANT
    is added, multi-turn assistant messages will also be marked as system and therefore
    excluded from shapley value calculations.
    """

    speaker: Role | None = None
    """The role of the current speaker in the chat."""
    turn_number: int
    """The current turn number in the chat."""
    token_turns: Tensor
    """A tensor indicating the turn structure of the chat."""
    token_roles: Tensor
    """A tensor indicating the role of each token in the chat."""

    empty_turn_sequences: list[Tensor]
    """A tensor indicating the empty turn sequence."""
    text_tokens_no_system_mask: Tensor
    """A boolean tensor indicating which text tokens are user-generated."""
    audio_tokens_no_system_mask: Tensor
    """A boolean tensor indicating which audio tokens are user-generated."""
    token_sequences_to_exclude: list[Tensor]
    """A list of token IDs to exclude from processing."""

    _system_roles: set[Role]
    __shap: ExplainerCache | None = None

    __external_shap_values_mask: Tensor | None = None
    __external_group_ids: Tensor | None = None
    _SHARED_ATTRIBUTES: frozenset[str] = frozenset({
        "system_roles_setup",
        "_system_roles",
        "empty_turn_sequences",
        "token_sequences_to_exclude",
    })

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        device: torch.device,
        empty_turn_sequences: set[str],
        token_filter: TokenFilter | None = None,
        system_roles_setup: SystemRolesSetup | None = None,
    ) -> None:
        """
        Initialize the chat state.

        Args:
            device: The device on which tensors are stored.
            empty_turn_sequences: A list of empty turn sequences (their text representations).
            token_filter: The token filtering class.
            system_roles_setup: The setup for system roles.
                Defaults to SystemRolesSetup.SYSTEM.
        """
        # validation
        __config = BaseChatConfig(
            device=device,
            token_filter=token_filter if token_filter is not None else KeepAllTokens(),
            system_roles_setup=system_roles_setup if system_roles_setup is not None else SystemRolesSetup.SYSTEM,
            empty_turn_sequences=empty_turn_sequences,
        )

        self.torch_device = __config.device
        self.token_filter = __config.token_filter
        self.system_roles_setup = __config.system_roles_setup

        if self.system_roles_setup == SystemRolesSetup.SYSTEM:
            self._system_roles = {Role.SYSTEM}
        elif self.system_roles_setup == SystemRolesSetup.SYSTEM_ASSISTANT:
            self._system_roles = {Role.SYSTEM, Role.ASSISTANT}
        else:  # SystemRolesSetup.NONE
            self._system_roles = set()

        self.text_tokens_no_system_mask = torch.zeros(0, dtype=torch.bool, device=device)
        self.audio_tokens_no_system_mask = torch.zeros(0, dtype=torch.bool, device=device)

        self.turn_number = 0
        self.token_turns = torch.zeros(0, dtype=torch.int16, device=device)
        self.token_roles = torch.zeros(0, dtype=torch.int8, device=device)

        self.token_sequences_to_exclude = self._get_tokens_sequences_to_exclude(self.token_filter.phrases_to_exclude)
        self.empty_turn_sequences = self._get_tokens_sequences_to_exclude(__config.empty_turn_sequences)

    @classmethod
    def from_chat(
        cls,
        mask: Tensor,
        chat: "BaseMllmChat",
    ) -> "BaseMllmChat":
        """
        Create a new chat instance from an existing chat and a mask.

        Args:
            mask: A boolean tensor indicating which messages to include.
            chat: The existing chat instance to copy.
        Returns:
            An instance of BaseMllmChat.
        Raises:
            ValueError: If the mask size does not match the number of tokens in the chat,
                or if the mask is all False,
                or if all text tokens are filtered out.
        """
        logger.debug("Creating new chat instance from existing chat with masks.")

        if mask.shape[0] != chat.input_tokens_num:
            raise ValueError("Mask size does not match the number of tokens in the chat.")
        if not mask.any():
            raise ValueError("Mask cannot be all False.")

        # translate mask back - for group ids only first entry in mask is correct
        mask = chat.translate_groups_ids_mask(mask)

        text_mask_relative = mask[chat.text_tokens_mask]
        audio_mask_relative = mask[chat.audio_tokens_mask]

        # filter out empty turn sequences
        new_chat_text_tokens = chat.text_tokens[text_mask_relative]
        new_chat_text_tokens_mask = torch.ones_like(new_chat_text_tokens, dtype=torch.bool, device=chat.torch_device)
        for seq_tensor in chat.empty_turn_sequences:
            seq_len = seq_tensor.shape[0]
            # assume _detect is protected not private
            match_indices = chat._detect(  # pylint: disable=protected-access
                tokens=new_chat_text_tokens,
                seq_tensor=seq_tensor,
                mark=False,
            )

            for idx in match_indices:
                turn_number = chat.token_turns[idx].item()
                turn_mask = chat.token_turns == turn_number

                audio_turn_mask = turn_mask[chat.audio_tokens_mask]
                if audio_mask_relative[audio_turn_mask].any():  # there is still used audio in the turn
                    continue

                new_chat_text_tokens_mask[idx : idx + seq_len] = False  # noqa: E203

        # update masks
        text_mask_relative[text_mask_relative.clone()] = new_chat_text_tokens_mask
        mask[chat.text_tokens_mask] = text_mask_relative

        if not text_mask_relative.any():
            raise AllTextTokensFilteredOutError("Resulting chat cannot have all text tokens filtered out.")

        new_instance = cls._set_new_instance(
            full_mask=mask,
            text_mask_relative=text_mask_relative,
            audio_mask_relative=audio_mask_relative,
            chat=chat,
        )
        new_instance.refresh(full=True)

        # dont propagate external masks / group ids
        del new_instance.external_shap_values_mask
        del new_instance.external_group_ids

        return new_instance

    @property
    def cache(self) -> ExplainerCache | None:
        """Access for the explainer cache if set."""
        return self.__shap

    @cache.setter
    def cache(self, value: ExplainerCache | None) -> None:
        """
        Set the explainer cache.

        Args:
            value: The ExplainerCache instance to set.
        Raises:
            ValueError: If the explainer cache is for a different chat instance.
        """
        if self.__shap is not None and value is not None and self.__shap.chat != value.chat:
            raise ValueError("Cannot set explainer cache for a different chat instance.")
        self.__shap = value

    @cache.deleter
    def cache(self) -> None:
        """Delete the explainer cache."""
        if self.__shap is not None:
            del self.__shap
            self.__shap = None

    @property
    def input_tokens_num(self) -> int:
        """Total number of input tokens (text + audio)."""
        return len(self.input_tokens)

    @cached_property
    @abstractmethod
    def input_tokens(self) -> list[Tensor]:
        """Combined input tensor (text + audio)."""

    @cached_property
    @abstractmethod
    def tokens_modality_flag(self) -> Tensor:
        """The modality flag tensor indicating token types according to :class:`ModalityFlag` enum."""

    @cached_property
    def text_tokens_mask(self) -> Tensor:
        """Boolean mask indicating positions of text tokens in the input (:attr:`input_tokens` tensor)."""
        return self.tokens_modality_flag == ModalityFlag.TEXT

    @cached_property
    def audio_tokens_mask(self) -> Tensor:
        """Boolean mask indicating positions of audio tokens in the input (:attr:`input_tokens` tensor)."""
        return ~self.text_tokens_mask

    @cached_property
    @abstractmethod
    def text_tokens(self) -> Tensor:
        """Input text tensor (tokens)."""

    @cached_property
    @abstractmethod
    def audio_tokens(self) -> Tensor:
        """Input audio tensor (tokens) in shape (T, K)"""

    @cached_property
    def text_tokens_no_system_mask_filtered(self) -> Tensor:
        """
        Boolean mask indicating which text tokens are not system
        generated, after filtering out specified sequences.
        Relative to :attr:`text_tokens_mask`.
        """
        mask = self.text_tokens_no_system_mask.clone()

        # filter out sequences to exclude
        for seq_tensor in self.token_sequences_to_exclude:
            mask = self._detect(tokens=self.text_tokens, seq_tensor=seq_tensor, mask=mask, mark=True)

        return mask

    @property
    def audio_tokens_no_system_mask_filtered(self) -> Tensor:
        """
        Boolean mask indicating which audio tokens are not system
        generated. Relative to :attr:`audio_tokens_mask`.
        """
        return self.audio_tokens_no_system_mask  # no filtering

    @property
    def external_shap_values_mask(self) -> Tensor | None:
        """
        An optional external SHAP values mask. Should be a boolean tensor
        with size equal to the number of tokens in the chat.

        If provided, :attr:`shap_values_mask` will be and-ed with this mask.

        Forbids adding new tokens to the chat while set.
        """
        return self.__external_shap_values_mask

    @external_shap_values_mask.setter
    def external_shap_values_mask(self, value: Tensor) -> None:
        """
        Set the external SHAP values mask.

        Args:
            value: The external SHAP values mask tensor.
        Raises:
            ValueError: If the external SHAP values mask size does not match the number of tokens in the chat.
        """
        if self.input_tokens_num != value.shape[0]:
            raise ValueError("External SHAP values mask size does not match the number of tokens in the chat.")
        self.__external_shap_values_mask = value
        self.refresh(shap=True)

    @external_shap_values_mask.deleter
    def external_shap_values_mask(self) -> None:
        """
        Delete the external SHAP values mask.
        """
        self.__external_shap_values_mask = None
        self.refresh(shap=True)

    @property
    def external_group_ids(self) -> Tensor | None:
        """
        An optional external group IDs for tokens. Should be an integer tensor
        with size equal to the number of tokens in the chat and set directly before
        the explanation process.

        All entries > 0 will be treated as belonging to the same group for
        SHAP value calculations, 0 entries will be ignored from calculations
        (shap_values_mask for them will be set to False). Takes precedence over
        :attr:`external_shap_values_mask`.

        Forbids adding new tokens to the chat while set.
        """
        return self.__external_group_ids

    @external_group_ids.setter
    def external_group_ids(self, value: Tensor) -> None:
        """
        Set the external group IDs for tokens.

        Args:
            value: The external group IDs tensor.
        Raises:
            ValueError: If the external group IDs size does not match the number of tokens in the chat.
        """
        if self.input_tokens_num != value.shape[0]:
            raise ValueError("External group IDs size does not match the number of tokens in the chat.")
        self.__external_group_ids = value
        self.refresh(shap=True)

    @external_group_ids.deleter
    def external_group_ids(self) -> None:
        """
        Delete the external group IDs for tokens.
        """
        self.__external_group_ids = None
        self.refresh(shap=True)

    @cached_property
    def external_group_ids_first_positions(self) -> Tensor:
        """
        Get the positions (indices) of the first occurrences of each
        consecutive non-zero group ID in `external_group_ids`.

        Returns:
            Tensor of positions (indices) of the first occurrence of
            each non-zero group ID, or None if not set.
        Raises:
            ValueError: If external_group_ids is not set.
        """
        if self.external_group_ids is None:
            raise ValueError("external_group_ids is not set.")

        ids = self.external_group_ids

        # Get unique consecutive IDs and counts
        unique_ids, counts = torch.unique_consecutive(ids, return_counts=True)

        # Compute start indices for each run
        start_positions = torch.cat([torch.tensor([0], device=ids.device), counts.cumsum(0)[:-1]])

        # Filter out group ID == 0
        mask = unique_ids != 0
        first_positions = start_positions[mask]

        return first_positions

    @cached_property
    def external_group_ids_positive_mask(self) -> Tensor:
        """
        Boolean mask indicating which tokens have positive group IDs.

        Returns:
            Boolean mask indicating which tokens have positive group IDs.
        Raises:
            ValueError: If external_group_ids is not set.
        """
        if self.external_group_ids is None:
            raise ValueError("external_group_ids is not set.")

        return self.external_group_ids > 0

    @cached_property
    def shap_values_mask(self) -> Tensor:
        """
        Boolean mask indicating which tokens should be considered
        for SHAP value calculations (i.e., non-system text tokens).

        Raises:
            ValueError: If the external SHAP values mask size does not match the number of tokens in the chat.
        """
        mask = torch.zeros(self.input_tokens_num, dtype=torch.bool, device=self.torch_device)

        # set text tokens
        text_mask = self.text_tokens_mask
        mask[text_mask] = self.text_tokens_no_system_mask_filtered

        # set audio tokens
        audio_mask = self.audio_tokens_mask
        mask[audio_mask] = self.audio_tokens_no_system_mask_filtered

        if self.external_group_ids is not None:
            # mark only tokens within positive groups ids, just first token in each group
            new_mask = torch.zeros_like(mask, dtype=torch.bool, device=self.torch_device)
            new_mask[self.external_group_ids_first_positions] = True
            mask = new_mask & mask
        elif self.external_shap_values_mask is not None:
            mask &= self.external_shap_values_mask

        return mask

    def refresh(self, full: bool = False, shap: bool = False) -> None:
        """
        Refresh cached properties, that is:

        - input_tokens
        - tokens_modality_flag
        - text_tokens_mask
        - text_tokens
        - audio_tokens_mask
        - audio_tokens

        If `full` is True, also refresh:

        - audio_tokens_no_system_mask_filtered
        - text_tokens_no_system_mask_filtered
        - shap_values_mask
        - external_group_ids_first_positions
        - external_group_ids_positive_mask

        If `shap` is True, will only refresh:
        - shap_values_mask
        - external_group_ids_first_positions
        - external_group_ids_positive_mask

        Args:
            full: If True, refreshes all cached properties.
            shap: If True, refreshes shap-related cached properties.
        """
        logger.debug("Refreshing cached properties (full=%s, shap=%s).", full, shap)

        if shap:
            self.__dict__.pop("shap_values_mask", None)
            self.__dict__.pop("external_group_ids_first_positions", None)
            self.__dict__.pop("external_group_ids_positive_mask", None)
            return

        self.__dict__.pop("input_tokens", None)
        self.__dict__.pop("tokens_modality_flag", None)

        self.__dict__.pop("text_tokens_mask", None)
        self.__dict__.pop("text_tokens", None)

        self.__dict__.pop("audio_tokens_mask", None)
        self.__dict__.pop("audio_tokens", None)
        if full:
            self.__dict__.pop("audio_tokens_no_system_mask_filtered", None)
            self.__dict__.pop("text_tokens_no_system_mask_filtered", None)
            self.__dict__.pop("shap_values_mask", None)
            self.__dict__.pop("external_group_ids_first_positions", None)
            self.__dict__.pop("external_group_ids_positive_mask", None)

    def new_turn(self, speaker: Role) -> None:
        """
        Start a new turn in the chat state.

        Warning:
            `For Developers:` This method assumes cached property refresh is handled in :func:`_new_turn` or
            by calling add_text/add_audio methods.
        Args:
            speaker: The role of the speaker for the new turn.
        Raises:
            ValueError: If a turn is already active.
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if self.speaker is not None:
            raise ValueError("Cannot start a new turn while another turn is active. Please end the current turn first.")
        self.turn_number += 1

        # mark system messages
        self.speaker = Role.SYSTEM
        raise_connector_error(self._new_turn, speaker)
        self.speaker = speaker

        logger.debug("New turn started with speaker: %s", speaker)

    def end_turn(self) -> None:
        """
        End the current turn in the chat state.

        Warning:
            `For Developers:` This method assumes cached property refresh is handled in :func:`_new_turn` or
            by calling add_text/add_audio methods.
        Raises:
            ValueError: If no turn is active.
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if self.speaker is None:
            raise ValueError("No active turn to end. Please start a turn first.")
        current_speaker = self.speaker

        # mark system messages
        self.speaker = Role.SYSTEM
        raise_connector_error(self._end_turn)
        self.speaker = None

        logger.debug("Turn ended for speaker: %s", current_speaker)

    def add_text(self, text: str) -> None:
        """
        Add text to the chat state.

        Args:
            text: The text to add.
        Raises:
            ValueError: If text is not a non-empty string.
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if not isinstance(text, str) or not text:
            raise ValueError(f"text must be a non-empty string, got {type(text)}")
        self._before_add()

        n_tokens_added = raise_connector_error(self._add_text, text)
        self._after_add(n_tokens_added, text_added=True)

        logger.debug(
            "Added text: '%s' (is_system=%s, speaker=%s)",
            text.replace("\n", "\\n"),
            self.is_system_turn,
            self.speaker,
        )

    def add_audio(
        self,
        audio_content: bytes,
        audio_format: str = "mp3",
    ) -> None:
        """
        Add audio content to the chat state.

        Args:
            audio_content: The audio content in bytes.
            audio_format: The format of the audio content (default is "mp3").
        Raises:
            ValueError: If audio_content is not non-empty bytes.
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if not isinstance(audio_content, bytes) or not audio_content:
            raise ValueError(f"audio_content must be non-empty bytes, got {type(audio_content)}")
        self._before_add()

        waveform, sample_rate = TorchAudioHandler.from_bytes(audio_content, audio_format=audio_format)
        n_tokens_added = raise_connector_error(self._add_audio, waveform, sample_rate)
        self._after_add(n_tokens_added, text_added=False)

        logger.debug(
            "Added audio content of format: '%s' (speaker=%s)",
            audio_format,
            self.speaker,
        )

    def append(
        self,
        text: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
        history_tracking_mode: ModelHistoryTrackingMode,
    ) -> None:
        """
        Append text and audio tokens along with modality flags to the chat state.
        Assumes that entry data is correct and non system.

        Warning:
            This method does not validate input data.
        Args:
            text: The text tokens to append.
            audio_out: The audio tokens to append (intended for model's output).
            modality_flag: The modality flags corresponding to the tokens.
            history_tracking_mode: The mode for tracking chat history.
        Raises:
            ValueError: If length mismatch occurs after appending.
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        self._before_add()

        text_tokens_added, audio_tokens_added = raise_connector_error(
            self._append,
            text,
            audio_out,
            modality_flag,
            history_tracking_mode=history_tracking_mode,
        )

        self._after_add(text_tokens_added, text_added=True, refresh=False)
        self._after_add(audio_tokens_added, text_added=False, refresh=True)

        logger.debug(
            "Appended text tokens: %d, audio tokens: %d",
            text_tokens_added,
            audio_tokens_added,
        )

    def decode_text(
        self,
        text_tokens: list[Tensor] | Tensor | None = None,
    ) -> str:
        """
        Decode the generated text tokens.

        Warning:
            This method does not validate input data.
        Args:
            text_tokens: The generated text tokens.
        Returns:
            The decoded text.
        Raises:
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if text_tokens is None:
            text_tokens = self.text_tokens
        if isinstance(text_tokens, list):
            text_tokens = torch.cat(text_tokens)

        with torch.no_grad():
            logger.debug("Decoding text tokens (%d): %s", text_tokens.shape[0], text_tokens)
            return cast(str, raise_connector_error(self._decode_text, text_tokens))

    def decode_audio(
        self,
        audio_tokens: list[Tensor] | Tensor | None = None,
        sample_rate: int = 24_000,
        audio_format: str = "mp3",
    ) -> bytes:
        """
        Decode the generated audio tokens.

        Warning:
            This method does not validate input data.
        Args:
            audio_tokens: Audio tokens to decode in format (T, K).
            sample_rate: The sample rate for the decoded audio (default is 24,000 Hz).
            audio_format: The desired output audio format (default is "mp3").
        Returns:
            The decoded audio content in bytes. Empty bytes if decoding is not available.
        Raises:
            RuntimeError: If an error occurs in the underlying connector implementation.
        """
        if audio_tokens is None:
            audio_tokens = self.audio_tokens
        if isinstance(audio_tokens, list):
            audio_tokens = torch.cat(audio_tokens)

        with torch.no_grad():
            logger.debug("Decoding audio tokens (%d): %s", audio_tokens.shape[0], audio_tokens)
            waveform = raise_connector_error(self._decode_audio, audio_tokens)
        if waveform is None:  # unable to decode
            return b""
        return TorchAudioHandler.to_bytes(waveform.cpu(), sample_rate=sample_rate, audio_format=audio_format)

    # pylint: disable=too-many-locals
    def get_conversation(self) -> list[list[ChatEntry]]:
        """
        Serialize the chat state to a JSON-compatible dictionary.

        Returns:
            A list of turns, where each turn is a list of ChatEntry objects. `shap_values=None`
                indicates that SHAP values are not yet available.
        Example:
        ::

            [
                [
                    ChatEntry(
                        content_type=0,
                        roles=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                        content='<|im_start|>, system, \\n, You, are, a helpful assistant that answers questions briefly...', # noqa: E501 # pylint: disable=line-too-long
                        shap_values=None
                    )
                ],
                [
                    ChatEntry(
                        content_type=0,
                        roles=[2, 2, 2, 0, 0, 0, 0, 2, 2],
                        content='<|im_start|>, user, \\n, Who, are, you, ?, <|im_end|>, \\n',
                        shap_values=None
                    )
                ]
            ]
        """
        conversation: list[list[ChatEntry]] = []

        for turn in range(1, self.turn_number + 1):
            turn_conversation: list[ChatEntry] = []

            # Mask for current turn
            turn_mask = self.token_turns == turn
            text_turn_tokens_mask = turn_mask & self.text_tokens_mask
            text_turn_tokens_mask_relative = text_turn_tokens_mask[turn_mask]

            r = torch.where(turn_mask)
            min_, max_ = int(r[0].min().item()), int(r[0].max().item())

            # Extract per-turn data
            # input_tokens is list, turns are contiguous
            turn_tokens = self.input_tokens[min_ : max_ + 1]  # noqa: E203
            turn_roles = self.token_roles[turn_mask]
            shap_values_normalized: Tensor | None = self.cache.normalized_values[turn_mask] if self.cache else None

            decoded: str | bytes
            last_modality: int | None = None
            roles: list[int] = []
            content: list[str | bytes] = []
            shap_values: list[float | None] = []

            for i, (token, role) in enumerate(zip(turn_tokens, turn_roles)):
                is_text = bool(text_turn_tokens_mask_relative[i])
                current_modality = ModalityFlag.TEXT if is_text else ModalityFlag.AUDIO

                # New segment when modality changes
                if last_modality is not None and current_modality != last_modality:
                    turn_conversation.append(
                        ChatEntry(
                            content_type=cast(ModalityFlag, last_modality).value,
                            roles=roles,
                            content=content,
                            shap_values=shap_values if shap_values_normalized is not None else None,
                        )
                    )
                    content = []
                    shap_values = []
                    roles = []

                # Decode token depending on modality
                if current_modality == ModalityFlag.TEXT:
                    decoded = self.decode_text(text_tokens=token)
                else:
                    decoded = self.decode_audio(audio_tokens=token)
                content.append(decoded)
                roles.append(role.item())

                # SHAP value for token
                if shap_values_normalized is not None:
                    shap_values.append(float(shap_values_normalized[i].item()))

                last_modality = current_modality

            # Append final segment
            if content:
                turn_conversation.append(
                    ChatEntry(
                        content_type=cast(ModalityFlag, last_modality).value,
                        roles=roles,
                        content=content,
                        shap_values=shap_values if shap_values_normalized is not None else None,
                    )
                )

            conversation.append(turn_conversation)

        return conversation

    def translate_groups_ids_mask(self, mask: Tensor) -> Tensor:
        """
        Translate a mask over group IDs to a mask over all tokens.

        Args:
            mask: A boolean tensor indicating which group IDs to include.
        Returns:
            A boolean tensor indicating which tokens to include.
        """
        if self.external_group_ids is not None:
            # select groups ids included within a mask and mark all their tokens to True
            groups_included = torch.where(mask[self.external_group_ids_first_positions])[0] + 1
            mask[torch.isin(self.external_group_ids, groups_included)] = True
            # remaining group ids are to be excluded, if they are in shap_values_mask
            groups_excluded = torch.where(~mask[self.external_group_ids_first_positions])[0] + 1
            mask[torch.isin(self.external_group_ids, groups_excluded)] = False
        return mask

    @classmethod
    @abstractmethod
    def _set_new_instance(
        cls: type["BaseMllmChat"],
        full_mask: Tensor,
        text_mask_relative: Tensor,
        audio_mask_relative: Tensor,
        chat: "BaseMllmChat",
    ) -> "BaseMllmChat":
        """
        Create a new chat instance from an existing chat and a mask.

        Args:
            full_mask: A boolean tensor indicating which messages to include.
            text_mask_relative: A boolean tensor indicating which text tokens to keep.
            audio_mask_relative: A boolean tensor indicating which audio tokens to keep.
            chat: The existing chat instance to copy.
        Returns:
            An instance of BaseMllmChat.
        """

    @property
    def is_system_turn(self) -> bool:
        """Flag indicating whether the current turn is a system turn."""
        return self.speaker in self._system_roles

    @abstractmethod
    def _decode_text(self, text_tokens: Tensor) -> str:
        """
        Decode the generated text tokens.

        Args:
            text_tokens: The generated text tokens.
        Returns:
            The decoded text.
        """

    @abstractmethod
    def _decode_audio(self, audio_tokens: Tensor) -> Tensor | None:
        """
        Decode the generated audio tokens.

        Args:
            audio_tokens: The generated audio tokens in format (T, K).
        Returns:
            The decoded audio waveform tensor or None if decoding is not possible.
        """

    @abstractmethod
    def _add_text(self, text: str) -> int:
        """
        Add text to the chat state.

        Args:
            text: The text to add.
        Returns:
            The number of tokens added.
        """

    @abstractmethod
    def _add_audio(self, waveform: Tensor, sample_rate: int) -> int:
        """
        Add audio content to the chat state.

        Args:
            waveform: The audio waveform tensor.
            sample_rate: The sample rate of the audio.
        Returns:
            The number of tokens added.
        """

    @abstractmethod
    def _append(
        self,
        text: Tensor,
        audio_out: Tensor,
        modality_flag: Tensor,
        history_tracking_mode: ModelHistoryTrackingMode,
    ) -> tuple[int, int]:
        """
        Append text and audio tokens along with modality flags to the chat state.

        Args:
            text: The text tokens to append.
            audio_out: The audio tokens to append.
            modality_flag: The modality flags corresponding to the tokens.
            history_tracking_mode: The mode for tracking chat history.
        Returns:
            A tuple containing the number of text tokens and audio tokens added.
        """

    @abstractmethod
    def _new_turn(self, speaker: Role) -> None:
        """
        Prepare for a new turn in the chat state.

        Args:
            speaker: The role of the speaker for the new turn.
        """

    @abstractmethod
    def _end_turn(self) -> None:
        """Finalize the current turn in the chat state."""

    @abstractmethod
    def _get_tokens_sequences_to_exclude(self, phrases_to_exclude: set[str]) -> list[Tensor]:
        """
        Get the list of tensors representing token sequences to exclude from processing.

        Args:
            phrases_to_exclude: The set of phrases to exclude.
        Returns:
            A list of token sequences to exclude.
        """

    def _extend_text_tokens_no_system_mask(self, num_tokens: int, is_user: bool) -> None:
        """
        Extend the text_tokens_user_flags tensor.

        Args:
            num_tokens: Number of tokens to add.
            is_user: Whether the new tokens are from the user or system.
        """
        logger.debug("Extending text tokens (no system mask): %d", num_tokens)
        self.__extend_tensor(
            num_tokens=num_tokens,
            fill_value=is_user,
            tensor_name="text_tokens_no_system_mask",
        )

    def _extend_audio_tokens_no_system_mask(self, num_tokens: int, is_user: bool) -> None:
        """
        Extend the audio_tokens_user_flags tensor.

        Args:
            num_tokens: Number of tokens to add.
            is_user: Whether the new tokens are from the user or system.
        """
        logger.debug("Extending audio tokens (no system mask): %d", num_tokens)
        self.__extend_tensor(
            num_tokens=num_tokens,
            fill_value=is_user,
            tensor_name="audio_tokens_no_system_mask",
        )

    def _extend_token_turns(self, num_tokens: int) -> None:
        """
        Extend the text_tokens_user_flags tensor.

        Args:
            num_tokens: Number of tokens to add.
        """
        logger.debug("Extending token turns: %d", num_tokens)
        self.__extend_tensor(
            num_tokens=num_tokens,
            fill_value=self.turn_number,
            tensor_name="token_turns",
        )

    def _extend_token_roles(self, num_tokens: int) -> None:
        """
        Extend the token_roles tensor.

        Args:
            num_tokens: Number of tokens to add.
        Raises:
            ValueError: If there is no active speaker.
        """
        if self.speaker is None:
            raise ValueError("Cannot extend token roles when there is no active speaker.")

        logger.debug("Extending token roles: %d", num_tokens)
        self.__extend_tensor(
            num_tokens=num_tokens,
            fill_value=self.speaker.value,
            tensor_name="token_roles",
        )

    def _before_add(self) -> None:
        """
        Prepare for adding new tokens to the chat state.

        Raises:
            ValueError: If external_group_ids or external_shap_values_mask is set.
        """
        if self.external_group_ids is not None:
            raise ValueError("Cannot add tokens when external_group_ids is set.")
        if self.external_shap_values_mask is not None:
            raise ValueError("Cannot add tokens when external_shap_values_mask is set.")

    def _after_add(self, num_tokens: int, text_added: bool = True, refresh: bool = True) -> None:
        """
        Extend the masks tensor.

        Args:
            num_tokens: Number of tokens to add.
            text_added: Whether text was added (True) or audio (False).
            refresh: Whether to refresh the cached property.
        """
        if num_tokens == 0:
            return

        self._extend_token_turns(num_tokens)
        self._extend_token_roles(num_tokens)

        if text_added:
            self._extend_text_tokens_no_system_mask(num_tokens, not self.is_system_turn)
        else:
            self._extend_audio_tokens_no_system_mask(num_tokens, not self.is_system_turn)

        # refresh cached property
        if refresh:
            self.refresh(full=True)

    def _detect(
        self,
        tokens: Tensor,
        seq_tensor: Tensor,
        mask: Tensor | None = None,
        mark: bool = True,
    ) -> Tensor:
        """
        Detect occurrences of seq_tensor in self.text_tokens
        and update the mask to filter them out.

        Args:
            tokens: The input tokens tensor.
            seq_tensor: The sequence tensor to detect.
            mask: The boolean mask to update. Required if mark is True.
            mark: Whether to mark detected sequences as
                False in the mask. If False, returns match positions.
        Returns:
            The updated boolean mask if mark is True,
            otherwise the indices of matches.
        Raises:
            ValueError: If mask is None when mark is True.
        """
        if mask is None and mark:
            raise ValueError("Mask must be provided when mark is True.")

        seq_len = seq_tensor.shape[0]
        if seq_len == 0 or seq_len > tokens.shape[0]:
            if not mark:
                return torch.tensor([], dtype=torch.long, device=self.torch_device)
            return cast(Tensor, mask)

        # Slide a window over tokens
        # Create a rolling view of tokens (shape [L - seq_len + 1, seq_len])
        windows = tokens.unfold(0, seq_len, 1)

        # Compare windows with seq_tensor
        matches = (windows == seq_tensor).all(dim=1)  # shape: [L - seq_len + 1]
        match_indices = matches.nonzero(as_tuple=False).flatten()
        logger.debug(
            "Detected %d occurrences of sequence: %s",
            match_indices.shape[0],
            seq_tensor,
        )
        if not mark:
            return match_indices

        # Mark matching positions as False
        for idx in match_indices:
            cast(Tensor, mask)[idx : idx + seq_len] = False  # noqa: E203
        return cast(Tensor, mask)

    def __extend_tensor(self, num_tokens: int, fill_value: Any, tensor_name: str) -> None:
        """
        Extend the specified tensor.

        Args:
            num_tokens: Number of tokens to add.
            fill_value: The value to use for extension.
            tensor_name: The name of the tensor to extend.
        """
        if num_tokens == 0:
            return

        tensor_to_extend = getattr(self, tensor_name)

        new_flags = torch.full(
            (num_tokens,),
            fill_value,
            dtype=tensor_to_extend.dtype,
            device=self.torch_device,
        )
        try:
            setattr(self, tensor_name, torch.cat([tensor_to_extend, new_flags]))
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to extend tensor '{tensor_name}' with num_tokens={num_tokens}."
                f"Shapes involved: {tensor_to_extend.shape}, {new_flags.shape}."
            ) from e

    def __repr__(self) -> str:
        """Represent the chat state."""
        return (
            f"{type(self).__name__}("
            f"text_tokens_len={self.text_tokens_mask.sum()}, "
            f"audio_tokens_len={self.audio_tokens_mask.sum()}, "
            f"tokens_len={self.input_tokens_num}, "
            f"last_speaker={self.speaker})"
        )

    def __deepcopy__(self, memo: Any) -> "BaseMllmChat":
        """
        Create a deep copy of the chat instance.
        Shares read-only attributes and updates reference to copied chat within :attr:`cache`.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Get shared attributes from class hierarchy
        shared_attrs = set()
        for klass in cls.__mro__:
            if hasattr(klass, "_SHARED_ATTRIBUTES"):
                shared_attrs.update(klass._SHARED_ATTRIBUTES)

        for k, v in self.__dict__.items():
            if k == "_BaseMllmChat__shap":  # pylint: disable=magic-value-comparison
                shap = self.cache
                if shap is not None:
                    shap.chat = None  # type: ignore[assignment]
                    new_shap = deepcopy(shap, memo)
                    new_shap.chat = result
                    setattr(result, k, new_shap)
                else:
                    setattr(result, k, None)
            elif k in shared_attrs:
                # Share reference instead of copying
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


# rebuild model after definition of BaseShapExplainer
ExplainerCache.model_rebuild()
ModelResponse.model_rebuild()

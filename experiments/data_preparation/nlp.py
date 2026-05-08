"""NLP related utilities."""

import asyncio
import os
from asyncio import Lock, Semaphore
from typing import Any, cast

import nltk
import pandas as pd
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from google.cloud.texttospeech import (
    AudioConfig,
    AudioEncoding,
    SsmlVoiceGender,
    SynthesisInput,
    TextToSpeechAsyncClient,
    VoiceSelectionParams,
)
from tqdm.asyncio import tqdm_asyncio

from .constants import TTSConfig

_ = load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


NO_WORD_CHARS: str = ".,!?:;\"'()[]{}—–-\n\t "


def split_into_sentences(text: str) -> list[str]:
    """
    Split the given text into a list of sentences.

    Args:
        text: The input text to split.
    Returns:
        A list of sentences extracted from the text.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    return [str(x) for x in nltk.sent_tokenize(text) if str(x).strip(NO_WORD_CHARS)]


def count_sentences(text: str) -> int:
    """
    Count the number of sentences in a given text.

    Args:
        text: The input text to analyze.
    Returns:
        The number of sentences in the text.
    """
    return len(split_into_sentences(text))


class SynthesisError(Exception):
    """Custom exception for synthesis errors."""


class SynthesisQuotaExceeded(Exception):
    """Custom exception for synthesis quota exceeded errors."""


class TTS:
    """A class for text-to-speech synthesis using Google Cloud Text-to-Speech API."""

    client: TextToSpeechAsyncClient = TextToSpeechAsyncClient()
    limit_per_period: int = 500
    period_duration_seconds: int = 60
    semaphore_size: int = 25

    def __init__(self) -> None:
        """Initialize the TTS class with a rate limiter."""
        self._call_times: list[float] = []
        self._lock: Lock = Lock()
        self._semaphore: Semaphore = Semaphore(self.semaphore_size)

    async def synthesize_text(
        self,
        text: str,
        language_code: str,
        gender: SsmlVoiceGender,
        voice_name: str | None = None,
        retries: int = 3,
    ) -> bytes:
        """
        Synthesizes speech from the input string of text asynchronously.

        Args:
            text: The text to be synthesized.
            language_code: The language code of the voice.
            gender: The gender of the voice.
            voice_name: The name of the voice (optional).
            retries: Number of retries in case of failure (default is 3).
        Returns:
            The synthesized audio content in bytes.
        """
        await self._check_rate_limit()
        kw = {}
        if voice_name is not None:
            kw["name"] = voice_name

        text = text.strip(NO_WORD_CHARS)
        synthesis_input = SynthesisInput(text=text)
        voice = VoiceSelectionParams(
            language_code=language_code, ssml_gender=gender, **kw
        )

        audio_config = AudioConfig(audio_encoding=AudioEncoding.MP3)

        for attempt in range(retries):
            try:
                response = await self.client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                return response.audio_content
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(self.period_duration_seconds)
                else:
                    raise SynthesisError(f"Error during synthesis of {text}") from e

    async def synthesize_sentences(
        self, sentences: list[str], **kwargs: Any
    ) -> list[bytes]:
        """
        Synthesize a list of sentences into audio content asynchronously.

        Args:
            sentences: A list of sentences to be synthesized.
            **kwargs: Additional keyword arguments for the synthesize_text method.
        Returns:
            A list of synthesized audio content in bytes.
        """
        return await asyncio.gather(
            *(self.synthesize_text(sentence, **kwargs) for sentence in sentences)
        )

    async def synthesize_df(
        self,
        df: pd.DataFrame,
        column_to_synthesize: str = "sentences",
        target_column: str = "audio",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Synthesize a DataFrame column containing lists of sentences into audio content asynchronously.

        Args:
            df: The DataFrame to synthesize with a specified column.
            column_to_synthesize: The column to synthesize (default is 'sentences').
            target_column: The column to store the synthesized audio (default is 'audio').
            kwargs: Additional keyword arguments to pass to the synthesis methods.

        Returns:
            The DataFrame with the synthesized audio in the target column.
        """

        # Wrapped task to respect semaphore
        async def wrapped_task(
            idx: int, dt_to_synthesize: list[dict[str, list[str]]] | list[str] | str
        ) -> tuple[int, Any]:
            result: bytes | list[bytes] | list[dict[str, Any]]
            async with self._semaphore:
                if isinstance(dt_to_synthesize, str):
                    result = await self.synthesize_text(text=dt_to_synthesize, **kwargs)
                elif isinstance(dt_to_synthesize, list) and all(
                    isinstance(i, str) for i in dt_to_synthesize
                ):
                    result = await self.synthesize_sentences(
                        sentences=cast(list[str], dt_to_synthesize), **kwargs
                    )
                else:
                    result = []
                    for i, entry_dict in enumerate(dt_to_synthesize):
                        if "value" not in entry_dict:
                            raise ValueError(
                                f"Missing 'value' key in entry at index {idx}, sub-index {i}. Entry: {entry_dict}"
                            )
                        entry = cast(dict[str, Any], entry_dict).copy()
                        sentences = entry.pop("value")
                        if not isinstance(sentences, list) or not all(
                            isinstance(s, str) for s in sentences
                        ):
                            raise ValueError(
                                f"'value' must be a list of strings in entry at index {idx}, "
                                f"sub-index {i}. Found: {type(sentences)}"
                            )
                        entry["value"] = await self.synthesize_sentences(
                            sentences=sentences, **kwargs
                        )
                        result.append(entry)
                return idx, result

        # Create tasks
        tasks = [
            asyncio.create_task(wrapped_task(i, x))
            for i, x in enumerate(df[column_to_synthesize])
        ]
        results = [None] * len(tasks)

        try:
            for finished in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="Synthesizing"
            ):
                idx, res = await finished
                results[idx] = res
        except ResourceExhausted as e:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise SynthesisQuotaExceeded(
                f"Quota exceeded. Consider reducing the limit_per_minute or semaphore_size. "
                f"Number of requests in last minute: {len(self._call_times)}"
            ) from e
        except Exception:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise

        df[target_column] = [r if isinstance(r, list) else [r] for r in results]

        return df

    async def synthesize_df_from_config(
        self, df: pd.DataFrame, config: TTSConfig, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Wrapper around self.synthesize_df to use TTSConfig.

        Args:
            df: The DataFrame to synthesize.
            config: The TTSConfig to use for synthesis.
            **kwargs: Additional keyword arguments to pass to self.synthesize_df.
        Returns:
            The DataFrame with synthesized audio.
        """
        return await self.synthesize_df(
            df,
            gender=config.gender,
            language_code=config.language_code,
            voice_name=config.voice_name,
            **kwargs,
        )

    async def _check_rate_limit(self) -> None:
        """Check and enforce the rate limit."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            self._call_times = [
                t
                for t in self._call_times
                if current_time - t < self.period_duration_seconds
            ]

            if len(self._call_times) >= self.limit_per_period:
                # Wait until the oldest call is more than 20 seconds old
                wait_time = self.period_duration_seconds - (
                    current_time - self._call_times[0]
                )
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                self._call_times = self._call_times[1:]

            self._call_times.append(current_time)

    @staticmethod
    def display_audio(audio_content: bytes) -> None:
        """
        Display audio content in a Jupyter notebook.

        Args:
            audio_content: The audio content in bytes.
        """
        # Import here to avoid dependency if not used in notebook
        from IPython.display import Audio, display

        display(Audio(data=audio_content, autoplay=True))

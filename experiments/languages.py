"""Utility functions for language detection and filtering."""

import pandas as pd
from googletrans import Translator
from lingua import Language, LanguageDetector, LanguageDetectorBuilder  # pylint: disable=no-name-in-module
from tqdm.asyncio import tqdm_asyncio
from transformers import pipeline
from transformers.pipelines.text_classification import TextClassificationPipeline


class LanguageClassifier:
    """A language classifier that uses Lingua and a transformer model for detection."""

    _english_detector: LanguageDetector | None = None
    _lang_detector: TextClassificationPipeline | None = None

    @property
    def english_detector(self) -> LanguageDetector:
        """Lazy load and return the English language detector."""
        if self._english_detector is None:
            self._english_detector = (
                LanguageDetectorBuilder.from_languages(Language.ENGLISH).with_preloaded_language_models().build()
            )
        return self._english_detector

    @property
    def lang_detector(self) -> TextClassificationPipeline:
        """Lazy load and return the transformer-based language detector."""
        if self._lang_detector is None:
            self._lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        return self._lang_detector

    def is_language(self, text: str, label: str) -> bool:
        """
        Check if the given text is in the specified language with a confidence above the threshold.

        Args:
            text: The text to check.
            label: The language label to check against (e.g., "en" for English).
        Returns:
            True if the text is in the specified language, False otherwise.
        """
        return self.classify_language(text) == label

    def is_english(self, text: str) -> bool:
        """
        Check if the given text is in English with a confidence above the threshold.

        Args:
            text: The text to check.
        Returns:
            True if the text is in English, False otherwise.
        """
        # Fast check with Lingua
        if self.english_detector.detect_language_of(text):
            return True
        # Fallback to transformer-based detector
        return self.is_language(text, "en")

    def is_spanish(self, text: str) -> bool:
        """
        Check if the given text is in Spanish with a confidence above the threshold.

        Args:
            text: The text to check.
        Returns:
            True if the text is in Spanish, False otherwise.
        """
        return self.is_language(text, "es")

    def is_french(self, text: str) -> bool:
        """
        Check if the given text is in French with a confidence above the threshold.

        Args:
            text: The text to check.
        Returns:
            True if the text is in French, False otherwise.
        """
        return self.is_language(text, "fr")

    def classify_language(self, text: str) -> str:
        """
        Classify the language of the given text.

        Args:
            text: The text to classify.
        Returns:
            The language label of the text.
        """
        return str(
            sorted(self.lang_detector(text), key=lambda x: x["score"], reverse=True)[0][  # pylint: disable=not-callable
                "label"
            ]
        )


class LanguageTranslator:
    """A placeholder for a language translator class."""

    _translator: Translator = Translator()

    async def translate(self, text: str, target_language: str) -> str:
        """
        Translate the given text to the target language.

        Args:
            text: The text to translate.
            target_language: The target language code (e.g., "en" for English).
        Returns:
            The translated text.
        """
        return str((await self._translator.translate(text, dest=target_language)).text)

    async def to_english(self, text: str) -> str:
        """
        Translate the given text to English.

        Args:
            text: The text to translate.
        Returns:
            The translated text in English.
        """
        return await self.translate(text, target_language="en")

    async def to_spanish(self, text: str) -> str:
        """
        Translate the given text to Spanish.

        Args:
            text: The text to translate.
        Returns:
            The translated text in Spanish.
        """
        return await self.translate(text, target_language="es")

    async def to_french(self, text: str) -> str:
        """
        Translate the given text to French.

        Args:
            text: The text to translate.
        Returns:
            The translated text in French.
        """
        return await self.translate(text, target_language="fr")

    async def translate_df(
        self, df: pd.DataFrame, target_language: str, column_to_translate: str = "conversation__joined"
    ) -> pd.DataFrame:
        """
        Translate the specified column of the DataFrame to the target language.

        Args:
            df: The DataFrame to translate with a specified column.
            target_language: The target language code (e.g., 'en' for English).
            column_to_translate: The column to translate (default is 'conversation__joined').
        Returns:
            The DataFrame with the translated specified column.
        """
        tasks = [self.translate(x, target_language=target_language) for x in df[column_to_translate]]
        results = await tqdm_asyncio.gather(*tasks)
        df[column_to_translate] = results
        return df

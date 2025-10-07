"""Utility functions for language detection and filtering."""

from typing import cast

from lingua import Language, LanguageDetector, LanguageDetectorBuilder  # pylint: disable=no-name-in-module
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
            self._lang_detector = cast(
                TextClassificationPipeline,
                pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection"),
            )
        return self._lang_detector

    def is_english(self, text: str, label: str = "en") -> bool:
        """
        Check if the given text is in English with a confidence above the threshold.

        Args:
            text: The text to check.
            label: The language label to check against (default is "en" for English).
        Returns:
            True if the text is in English, False otherwise.
        """
        # Fast check with Lingua
        if not self.english_detector.detect_language_of(text):
            # Fallback to transformer-based detector
            if self.classify_language(text) != label:
                return False
        return True

    def is_spanish(self, text: str, label: str = "es") -> bool:
        """
        Check if the given text is in Spanish with a confidence above the threshold.

        Args:
            text: The text to check.
            label: The language label to check against (default is "es" for Spanish).
        Returns:
            True if the text is in Spanish, False otherwise.
        """
        if self.classify_language(text) != label:
            return False
        return True

    def is_french(self, text: str, label: str = "fr") -> bool:
        """
        Check if the given text is in French with a confidence above the threshold.

        Args:
            text: The text to check.
            label: The language label to check against (default is "fr" for French).
        Returns:
            True if the text is in French, False otherwise.
        """
        if self.classify_language(text) != label:
            return False
        return True

    def classify_language(self, text: str) -> str:
        """
        Classify the language of the given text.

        Args:
            text: The text to classify.
        Returns:
            The language label of the text.
        """
        return str(sorted(self.lang_detector(text), key=lambda x: x["score"], reverse=True)[0]["label"])


if __name__ == "__main__":
    # test the language classifier
    classifier = LanguageClassifier()
    print(classifier.is_english("This is a test sentence."))
    print(classifier.is_english("C'est une phrase de test."))
    print(classifier.is_spanish("Esta es una frase de prueba."))
    print(classifier.is_spanish("This is a test sentence."))
    print(classifier.is_french("C'est une phrase de test."))
    print(classifier.is_french("This is a test sentence."))

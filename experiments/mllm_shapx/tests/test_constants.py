"""Tests for constants module — enums, frozensets, and helper predicates."""

from ..src.constants import (
    AUDIO_MODALITIES,
    INTERLEAVED_AUDIO_FIRST_MODALITIES,
    INTERLEAVED_MODALITIES,
    INTERLEAVED_TEXT_FIRST_MODALITIES,
    AudioCol,
    DatasetSource,
    InputModality,
    audio_column_for,
    is_text_only_modality,
    needs_audio,
)


class TestInputModality:
    def test_text_only(self) -> None:
        assert is_text_only_modality(InputModality.TEXT) is True

    def test_audio_not_text_only(self) -> None:
        assert is_text_only_modality(InputModality.AUDIO_MALE) is False
        assert is_text_only_modality(InputModality.AUDIO_FEMALE) is False
        assert is_text_only_modality(InputModality.AUDIO_ORIGINAL) is False

    def test_interleaved_not_text_only(self) -> None:
        for m in INTERLEAVED_MODALITIES:
            assert is_text_only_modality(m) is False


class TestNeedsAudio:
    def test_text_does_not_need_audio(self) -> None:
        assert needs_audio(InputModality.TEXT) is False

    def test_all_audio_modalities_need_audio(self) -> None:
        for m in AUDIO_MODALITIES:
            assert needs_audio(m) is True

    def test_all_interleaved_need_audio(self) -> None:
        for m in INTERLEAVED_MODALITIES:
            assert needs_audio(m) is True


class TestAudioColumnFor:
    def test_text_returns_none(self) -> None:
        assert audio_column_for(InputModality.TEXT) is None

    def test_original(self) -> None:
        assert audio_column_for(InputModality.AUDIO_ORIGINAL) == AudioCol.ORIGINAL

    def test_male_audio(self) -> None:
        assert audio_column_for(InputModality.AUDIO_MALE) == AudioCol.MALE

    def test_female_audio(self) -> None:
        assert audio_column_for(InputModality.AUDIO_FEMALE) == AudioCol.FEMALE

    def test_interleaved_male(self) -> None:
        assert (
            audio_column_for(InputModality.INTERLEAVED_TEXT_FIRST_MALE) == AudioCol.MALE
        )
        assert (
            audio_column_for(InputModality.INTERLEAVED_AUDIO_FIRST_MALE)
            == AudioCol.MALE
        )

    def test_interleaved_female(self) -> None:
        assert (
            audio_column_for(InputModality.INTERLEAVED_TEXT_FIRST_FEMALE)
            == AudioCol.FEMALE
        )
        assert (
            audio_column_for(InputModality.INTERLEAVED_AUDIO_FIRST_FEMALE)
            == AudioCol.FEMALE
        )


class TestModalityGroupings:
    def test_interleaved_is_union(self) -> None:
        assert INTERLEAVED_MODALITIES == (
            INTERLEAVED_TEXT_FIRST_MODALITIES | INTERLEAVED_AUDIO_FIRST_MODALITIES
        )

    def test_no_overlap_audio_and_interleaved(self) -> None:
        assert AUDIO_MODALITIES.isdisjoint(INTERLEAVED_MODALITIES)


class TestDatasetSource:
    def test_all_sources_are_strings(self) -> None:
        for ds in DatasetSource:
            assert isinstance(ds, str)

    def test_expected_values(self) -> None:
        assert DatasetSource.HF_PARQUET == "hf_parquet"
        assert DatasetSource.HF_DATASETS == "hf_datasets"
        assert DatasetSource.LOCAL_PARQUET == "local_parquet"
        assert DatasetSource.LOCAL_CSV == "local_csv"

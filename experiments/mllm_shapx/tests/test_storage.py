"""Tests for storage module — JSON I/O, checkpoints, run directories."""

import json
from pathlib import Path


from ..src.constants import CHECKPOINT_VERSION
from ..src.storage import (
    _default_checkpoint,
    existing_completed_from_disk,
    load_checkpoint,
    make_run_dir,
    save_json,
    update_checkpoint,
)


class TestMakeRunDir:
    def test_creates_expected_structure(self, tmp_path: Path) -> None:
        run_dir = make_run_dir(str(tmp_path), "exp_001", "variant_a")
        assert run_dir.exists()
        assert (run_dir / "samples").exists()
        assert (run_dir / "summary").exists()
        assert run_dir == tmp_path / "exp_001" / "variant_a"

    def test_idempotent(self, tmp_path: Path) -> None:
        d1 = make_run_dir(str(tmp_path), "exp", "run")
        d2 = make_run_dir(str(tmp_path), "exp", "run")
        assert d1 == d2


class TestSaveJson:
    def test_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        save_json(path, {"key": "value"})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data == {"key": "value"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "file.json"
        save_json(path, [1, 2, 3])
        assert path.exists()
        assert json.loads(path.read_text()) == [1, 2, 3]

    def test_overwrites_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        save_json(path, {"first": True})
        save_json(path, {"second": True})
        data = json.loads(path.read_text())
        assert data == {"second": True}

    def test_handles_special_types(self, tmp_path: Path) -> None:
        import numpy as np
        import torch

        path = tmp_path / "special.json"
        save_json(
            path,
            {
                "np_int": np.int64(42),
                "np_float": np.float32(3.14),
                "tensor": torch.tensor([1, 2, 3]),
                "binary": b"hello",
            },
        )
        data = json.loads(path.read_text())
        assert data["np_int"] == 42
        assert abs(data["np_float"] - 3.14) < 0.01
        assert data["tensor"] == [1, 2, 3]
        assert data["binary"]["_binary"] is True
        assert data["binary"]["num_bytes"] == 5


class TestDefaultCheckpoint:
    def test_has_version(self) -> None:
        ckpt = _default_checkpoint()
        assert ckpt["version"] == CHECKPOINT_VERSION

    def test_has_empty_completed(self) -> None:
        ckpt = _default_checkpoint()
        assert ckpt["completed_indices"] == []

    def test_has_timestamps(self) -> None:
        ckpt = _default_checkpoint()
        assert "created_at" in ckpt
        assert "updated_at" in ckpt


class TestLoadCheckpoint:
    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        ckpt = load_checkpoint(tmp_path / "nonexistent.json")
        assert ckpt["version"] == CHECKPOINT_VERSION
        assert ckpt["completed_indices"] == []

    def test_loads_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.json"
        save_json(
            path,
            {
                "version": CHECKPOINT_VERSION,
                "completed_indices": [1, 2, 3],
                "next_index": 4,
                "created_at": 1.0,
                "updated_at": 2.0,
            },
        )
        ckpt = load_checkpoint(path)
        assert ckpt["completed_indices"] == [1, 2, 3]
        assert ckpt["next_index"] == 4

    def test_migrates_old_version(self, tmp_path: Path) -> None:
        path = tmp_path / "old_ckpt.json"
        save_json(
            path,
            {
                "version": 1,
                "completed_indices": [5],
                "next_index": 6,
                "created_at": 1.0,
                "updated_at": 2.0,
            },
        )
        ckpt = load_checkpoint(path)
        assert ckpt["version"] == CHECKPOINT_VERSION

    def test_malformed_json_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        ckpt = load_checkpoint(path)
        assert ckpt["version"] == CHECKPOINT_VERSION
        assert ckpt["completed_indices"] == []


class TestUpdateCheckpoint:
    def test_appends_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.json"
        ckpt = _default_checkpoint()
        update_checkpoint(path, ckpt, just_completed=7)
        assert 7 in ckpt["completed_indices"]
        # Verify persisted
        loaded = json.loads(path.read_text())
        assert 7 in loaded["completed_indices"]

    def test_no_duplicate_completed(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.json"
        ckpt = _default_checkpoint()
        update_checkpoint(path, ckpt, just_completed=5)
        update_checkpoint(path, ckpt, just_completed=5)
        assert ckpt["completed_indices"].count(5) == 1

    def test_updates_next_index(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.json"
        ckpt = _default_checkpoint()
        update_checkpoint(path, ckpt, next_index=10)
        assert ckpt["next_index"] == 10

    def test_stamps_version(self, tmp_path: Path) -> None:
        path = tmp_path / "ckpt.json"
        ckpt = {
            "version": 1,
            "completed_indices": [],
            "next_index": 0,
            "created_at": 1.0,
            "updated_at": 1.0,
        }
        update_checkpoint(path, ckpt, just_completed=0)
        assert ckpt["version"] == CHECKPOINT_VERSION


class TestExistingCompletedFromDisk:
    def test_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "samples").mkdir()
        done = existing_completed_from_disk(tmp_path)
        assert done == set()

    def test_finds_sample_files(self, tmp_path: Path) -> None:
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        (samples_dir / "sample_00003_result.json").write_text("{}")
        (samples_dir / "sample_00010_result.json").write_text("{}")
        (samples_dir / "other_file.json").write_text("{}")
        done = existing_completed_from_disk(tmp_path)
        assert done == {3, 10}

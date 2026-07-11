"""Tests for CLI module — argument parsing and command dispatch."""

from ..src.cli import OutputOptions, build_argparser, _resolve_configs


class TestBuildArgparser:
    def test_global_output_defaults(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["validate", "--config", "test.json"])
        assert args.quiet is False
        assert args.verbose is False
        assert args.json is False

    def test_global_output_json_flag(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["--json", "plan", "--config", "plan.json"])
        assert args.json is True

    def test_global_output_quiet_flag(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["--quiet", "run", "--config", "run.json"])
        assert args.quiet is True

    def test_global_output_verbose_flag(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["--verbose", "run", "--config", "run.json"])
        assert args.verbose is True

    def test_global_output_quiet_and_verbose_conflict(self) -> None:
        ap = build_argparser()
        from pytest import raises

        with raises(SystemExit):
            ap.parse_args(["--quiet", "--verbose", "validate", "--config", "x.json"])

    def test_validate_subcommand(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["validate", "--config", "test.json"])
        assert args.cmd == "validate"
        assert args.config == "test.json"
        assert args.check_dataset is False

    def test_validate_with_check_dataset(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["validate", "--config", "c.json", "--check-dataset"])
        assert args.check_dataset is True

    def test_plan_subcommand(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["plan", "--config", "plan.json"])
        assert args.cmd == "plan"
        assert args.config == "plan.json"

    def test_plan_skip_data(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["plan", "--config", "x.json", "--skip-data"])
        assert args.skip_data is True

    def test_run_subcommand(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["run", "--config", "run.json"])
        assert args.cmd == "run"
        assert args.config == "run.json"
        assert args.resume is False

    def test_run_with_resume(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["run", "--config", "r.json", "--resume"])
        assert args.resume is True

    def test_run_with_sharding(self) -> None:
        ap = build_argparser()
        args = ap.parse_args([
            "run",
            "--config",
            "r.json",
            "--shard-index",
            "2",
            "--num-shards",
            "8",
        ])
        assert args.shard_index == 2
        assert args.num_shards == 8

    def test_run_with_max_samples(self) -> None:
        ap = build_argparser()
        args = ap.parse_args(["run", "--config", "r.json", "--max-samples", "50"])
        assert args.max_samples == 50


class TestResolveConfigs:
    def test_single_config(self) -> None:
        from argparse import Namespace

        args = Namespace(config="path/to/config.json", config_list=None)
        result = _resolve_configs(args, OutputOptions())
        assert result == ["path/to/config.json"]

    def test_config_list(self, tmp_path) -> None:
        from argparse import Namespace

        list_file = tmp_path / "configs.txt"
        list_file.write_text("a.json\nb.json\n# comment\n\nc.json\n")
        args = Namespace(config=None, config_list=str(list_file))
        result = _resolve_configs(args, OutputOptions())
        assert result == ["a.json", "b.json", "c.json"]

    def test_glob_pattern(self, tmp_path) -> None:
        from argparse import Namespace

        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")
        (tmp_path / "c.txt").write_text("")
        args = Namespace(config=str(tmp_path / "*.json"), config_list=None)
        result = _resolve_configs(args, OutputOptions())
        assert len(result) == 2
        assert all(r.endswith(".json") for r in result)

"""W5 run-registry tests: provenance sidecars, manifests, re-run stability."""

import hashlib
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)

from npdl.experiments import (  # noqa: E402
    ExperimentRun,
    compute_config_hash,
    create_run,
    verify_manifest,
)


def _write(run, rel_path, content):
    path = run.artifact_path(*rel_path.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as f:
        f.write(content)
    return path


class TestConfigHash:
    def test_stable_across_key_order(self):
        first = {"b": [1, 2], "a": 1}
        second = {"a": 1, "b": [1, 2]}
        assert compute_config_hash(first) == compute_config_hash(second)

    def test_changes_with_value(self):
        assert compute_config_hash({"a": 1}) != compute_config_hash({"a": 2})

    def test_is_sha256_hex(self):
        assert re.fullmatch(r"[0-9a-f]{64}", compute_config_hash({"seed": 0}))


class TestCreateRun:
    def test_writes_seed_hash_command_and_config(self, tmp_path):
        config = {"num_agents": 10, "nested": {"beta": 0.3}}
        argv = ["run_parameter_sweep.py", "--config", "x.json", "--seed", "7"]
        run = create_run(str(tmp_path), "sweep", config, 7, argv=argv)

        assert isinstance(run, ExperimentRun)
        assert run.seed == 7
        assert run.config_hash == compute_config_hash(config)
        assert run.command == argv
        assert os.path.basename(run.run_dir).startswith("sweep_seed7_")
        assert os.path.basename(run.run_dir).endswith(compute_config_hash(config)[:8])

        with open(os.path.join(run.run_dir, "run_info.json")) as f:
            info = json.load(f)
        assert info["experiment"] == "sweep"
        assert info["seed"] == 7
        assert info["config_hash"] == compute_config_hash(config)
        assert info["command"] == argv

        with open(os.path.join(run.run_dir, "resolved_config.json")) as f:
            assert json.load(f) == config

    def test_run_dir_name_is_deterministic(self, tmp_path):
        config = {"a": 1}
        first = create_run(str(tmp_path), "exp", config, 3)
        second = create_run(str(tmp_path), "exp", config, 3)
        assert first.run_dir == second.run_dir

    def test_rerun_same_config_reuses_dir(self, tmp_path):
        first = create_run(str(tmp_path), "exp", {"a": 1}, 3, run_name="out")
        with open(os.path.join(first.run_dir, "result.csv"), "w") as f:
            f.write("x\n")
        second = create_run(str(tmp_path), "exp", {"a": 1}, 3, run_name="out")
        assert second.run_dir == first.run_dir

    def test_rerun_refuses_to_record_stale_artifacts(self, tmp_path):
        first = create_run(str(tmp_path), "exp", {"a": 1}, 3, run_name="out")
        for name in ("kept.csv", "stale.csv"):
            with open(os.path.join(first.run_dir, name), "w") as f:
                f.write("x\n")
            # Age the files so a rewrite is visible at any mtime resolution.
            os.utime(os.path.join(first.run_dir, name), ns=(10**9, 10**9))
        first.finalize()
        second = create_run(str(tmp_path), "exp", {"a": 1}, 3, run_name="out")
        with open(os.path.join(second.run_dir, "kept.csv"), "w") as f:
            f.write("x\n")
        with pytest.raises(RuntimeError, match="stale.csv"):
            second.finalize()
        os.remove(os.path.join(second.run_dir, "stale.csv"))
        manifest = second.finalize()
        assert [e["path"] for e in manifest["artifacts"]] == [
            "kept.csv",
            "resolved_config.json",
            "run_info.json",
        ]

    def test_refuses_dir_holding_a_different_run(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 3, run_name="out")
        with open(os.path.join(run.run_dir, "result.csv"), "w") as f:
            f.write("x\n")
        with pytest.raises(FileExistsError):
            create_run(str(tmp_path), "exp", {"a": 1}, 4, run_name="out")
        with pytest.raises(FileExistsError):
            create_run(str(tmp_path), "exp", {"a": 2}, 3, run_name="out")

    def test_refuses_non_run_dir_with_files(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "old_dump.csv").write_text("x\n")
        with pytest.raises(FileExistsError):
            create_run(str(tmp_path), "exp", {}, 0, run_name="out")

    def test_allows_dir_with_only_logs(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "run.log").write_text("started\n")
        run = create_run(str(tmp_path), "exp", {}, 0, run_name="out")
        assert run.run_dir == str(out)

    def test_defaults_command_to_sys_argv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--seed", "1"])
        run = create_run(str(tmp_path), "exp", {}, 1)
        assert run.command == ["prog", "--seed", "1"]


class TestFinalize:
    def test_manifest_lists_artifacts_with_hashes(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 0, argv=["cmd"])
        _write(run, "results.csv", b"a,b\n1,2\n")
        _write(run, "nested/summary.json", '{"x": 1}')

        manifest = run.finalize()

        by_path = {entry["path"]: entry for entry in manifest["artifacts"]}
        assert set(by_path) >= {
            "run_info.json",
            "resolved_config.json",
            "results.csv",
            "nested/summary.json",
        }
        assert "manifest.json" not in by_path
        expected = hashlib.sha256(b"a,b\n1,2\n").hexdigest()
        assert by_path["results.csv"]["sha256"] == expected
        assert by_path["results.csv"]["size_bytes"] == len(b"a,b\n1,2\n")
        assert manifest["seed"] == 0
        assert manifest["config_hash"] == compute_config_hash({"a": 1})

        with open(os.path.join(run.run_dir, "manifest.json")) as f:
            assert json.load(f) == manifest

    def test_manifest_excludes_log_files(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {}, 0, argv=["cmd"])
        _write(run, "sweep.log", "2026-01-01 INFO hello\n")
        manifest = run.finalize()
        assert "sweep.log" not in {e["path"] for e in manifest["artifacts"]}
        assert os.path.isfile(os.path.join(run.run_dir, "sweep.log"))

    def test_rerun_with_same_seed_reproduces_hashes(self, tmp_path):
        config = {"num_agents": 4, "rounds": 5}

        def _produce(root):
            run = create_run(root, "exp", config, 42, argv=["cmd", "--seed", "42"])
            _write(run, "out/data.csv", "i,v\n0,1.5\n1,2.5\n")
            _write(run, "out/summary.json", json.dumps({"mean": 2.0}, sort_keys=True))
            return run.finalize()

        first = _produce(str(tmp_path / "run_a"))
        second = _produce(str(tmp_path / "run_b"))
        first_hashes = {e["path"]: e["sha256"] for e in first["artifacts"]}
        second_hashes = {e["path"]: e["sha256"] for e in second["artifacts"]}
        assert first_hashes
        assert first_hashes == second_hashes


class TestVerifyManifest:
    def test_clean_run_verifies(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 0, argv=["cmd"])
        _write(run, "out.txt", "data")
        run.finalize()
        assert verify_manifest(run.run_dir) == []

    def test_detects_tampered_artifact(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 0, argv=["cmd"])
        path = _write(run, "out.txt", "data")
        run.finalize()
        with open(path, "w") as f:
            f.write("tampered")
        problems = verify_manifest(run.run_dir)
        assert problems == ["hash mismatch: out.txt"]

    def test_detects_missing_artifact(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 0, argv=["cmd"])
        path = _write(run, "out.txt", "data")
        run.finalize()
        os.remove(path)
        assert verify_manifest(run.run_dir) == ["missing artifact: out.txt"]

    def test_detects_unregistered_file(self, tmp_path):
        run = create_run(str(tmp_path), "exp", {"a": 1}, 0, argv=["cmd"])
        run.finalize()
        _write(run, "sneaky.txt", "unregistered")
        assert verify_manifest(run.run_dir) == ["unregistered file: sneaky.txt"]

    def test_missing_manifest_reported(self, tmp_path):
        assert verify_manifest(str(tmp_path)) == ["missing manifest.json"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

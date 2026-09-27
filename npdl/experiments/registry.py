"""Run-directory registry: provenance sidecars plus an artifact manifest."""

import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

RUN_INFO_FILENAME = "run_info.json"
CONFIG_FILENAME = "resolved_config.json"
MANIFEST_FILENAME = "manifest.json"

# Log files carry wall-clock timestamps, so they can never be reproducible
# across re-runs; they stay in the run dir but out of the manifest.
EXCLUDED_SUFFIXES = (".log",)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Return the sha256 hex digest of the resolved config.

    The config is canonicalized with sorted keys and compact separators so
    the hash is stable regardless of insertion order. Non-JSON-native values
    (numpy scalars, paths) fall back to ``str()``; JSON-native values hash
    exactly.
    """
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_experiment_name(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("._") or "experiment"


class ExperimentRun:
    """A single registered experiment run rooted at ``run_dir``."""

    def __init__(
        self,
        run_dir: str,
        experiment: str,
        seed: int,
        config_hash: str,
        command: List[str],
    ) -> None:
        self.run_dir = run_dir
        self.experiment = experiment
        self.seed = seed
        self.config_hash = config_hash
        self.command = command

    def artifact_path(self, *parts: str) -> str:
        """Return a path for an output artifact inside the run dir."""
        return os.path.join(self.run_dir, *parts)

    def _manifest_entries(
        self, exclude_suffixes: Sequence[str] = EXCLUDED_SUFFIXES
    ) -> List[Dict[str, Any]]:
        entries = []
        for dirpath, dirnames, filenames in os.walk(self.run_dir):
            dirnames.sort()
            for filename in sorted(filenames):
                if filename == MANIFEST_FILENAME:
                    continue
                if filename.endswith(tuple(exclude_suffixes)):
                    continue
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, self.run_dir).replace(os.sep, "/")
                entries.append(
                    {
                        "path": rel_path,
                        "size_bytes": os.path.getsize(full_path),
                        "sha256": _sha256_file(full_path),
                    }
                )
        entries.sort(key=lambda entry: entry["path"])
        return entries

    def finalize(
        self, exclude_suffixes: Sequence[str] = EXCLUDED_SUFFIXES
    ) -> Dict[str, Any]:
        """Hash every artifact in the run dir and write ``manifest.json``.

        Returns the manifest dict. Re-running the same seed/config produces
        identical artifact hashes when the runner is deterministic.
        """
        manifest = {
            "experiment": self.experiment,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "artifacts": self._manifest_entries(exclude_suffixes),
        }
        with open(os.path.join(self.run_dir, MANIFEST_FILENAME), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        return manifest


def create_run(
    root: str,
    experiment: str,
    config: Dict[str, Any],
    seed: int,
    argv: Optional[Sequence[str]] = None,
    run_name: Optional[str] = None,
) -> ExperimentRun:
    """Create a registered run dir with provenance sidecars.

    The directory name ``{experiment}_seed{seed}_{config_hash8}`` is fully
    deterministic: re-running the same seed/config lands in the same place.
    Writes ``resolved_config.json`` and ``run_info.json`` (no timestamps, so
    both are byte-stable across re-runs) and returns the :class:`ExperimentRun`.
    """
    config_hash = compute_config_hash(config)
    safe_experiment = _safe_experiment_name(experiment)
    if run_name is None:
        run_name = f"{safe_experiment}_seed{seed}_{config_hash[:8]}"
    run_dir = os.path.join(root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    command = list(argv) if argv is not None else list(sys.argv)

    with open(os.path.join(run_dir, CONFIG_FILENAME), "w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
        f.write("\n")

    run_info = {
        "experiment": experiment,
        "seed": seed,
        "config_hash": config_hash,
        "command": command,
        "config_file": CONFIG_FILENAME,
    }
    with open(os.path.join(run_dir, RUN_INFO_FILENAME), "w") as f:
        json.dump(run_info, f, indent=2, sort_keys=True)
        f.write("\n")

    return ExperimentRun(
        run_dir=run_dir,
        experiment=experiment,
        seed=seed,
        config_hash=config_hash,
        command=command,
    )


def verify_manifest(
    run_dir: str, exclude_suffixes: Sequence[str] = EXCLUDED_SUFFIXES
) -> List[str]:
    """Re-hash the artifacts listed in ``manifest.json``; return problems.

    An empty list means every listed artifact exists with a matching hash and
    no unregistered files (other than the manifest itself and excluded
    suffixes) are present.
    """
    problems: List[str] = []
    manifest_path = os.path.join(run_dir, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        return [f"missing {MANIFEST_FILENAME}"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    listed = set()
    for entry in manifest.get("artifacts", []):
        rel_path = entry["path"]
        listed.add(rel_path)
        full_path = os.path.join(run_dir, *rel_path.split("/"))
        if not os.path.isfile(full_path):
            problems.append(f"missing artifact: {rel_path}")
        elif _sha256_file(full_path) != entry["sha256"]:
            problems.append(f"hash mismatch: {rel_path}")

    for dirpath, _, filenames in os.walk(run_dir):
        for filename in filenames:
            if filename == MANIFEST_FILENAME:
                continue
            if filename.endswith(tuple(exclude_suffixes)):
                continue
            rel_path = os.path.relpath(
                os.path.join(dirpath, filename), run_dir
            ).replace(os.sep, "/")
            if rel_path not in listed:
                problems.append(f"unregistered file: {rel_path}")

    return sorted(problems)

"""The package version must match the one declared in pyproject.toml."""

import os
import re

import npdl

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version():
    with open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8") as fh:
        text = fh.read()
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert match, "no version in [project] table of pyproject.toml"
    return match.group(1)


def test_package_version_matches_pyproject():
    assert npdl.__version__ == _pyproject_version()

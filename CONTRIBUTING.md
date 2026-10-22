# Contributing

This is published research code. Do not change published results, numbers,
experiment definitions, parameters, or simulation behaviour as part of packaging
or test-infrastructure work.

## Test environment

Use Python 3.13 to match CI. From the repository root, create a fresh environment
and install the declared development extra. On Windows, Python 3.14 can require
a pygame source build; use 3.13 for this setup. In Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest tests/ -q
```

On Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
```

For the coverage command used by CI:

```bash
python -m pytest tests/ -q --cov=npdl --cov-report=xml --cov-report=term
```

On Windows, use `.venv\Scripts\python.exe` instead of `python` when the
environment is not activated. More test commands are in [tests/README.md](tests/README.md).

## Dependency strategy

[pyproject.toml](pyproject.toml) is the source of dependency declarations.
The `dev` extra supports pytest 8.4 and 9.x, includes pytest-cov, and inherits all
runtime dependencies, including Plotly, Dash, SciPy, and the other visualization
libraries. CI installs this same extra; no ad hoc test packages are required.
Development tooling needs a newer Python than the package's historical runtime
minimum; Python 3.13 is the CI reference environment.

[requirements.txt](requirements.txt) is a compatibility entry point for the same
editable development install; run it from the repository root.
[requirements.lock](requirements.lock) is a historical, partial Windows/Python
3.13 snapshot, with some unpinned entries. It is retained for reference, is not a
complete cross-platform lock, and is not used by the current test gate. The gate
resolves the declared version ranges rather than promising identical versions
across dates or platforms.

The old fixture mark incompatibility was fixed in commit `8d23262e`:
`tests/conftest.py` uses `pytest.fixture(params=...)` with `request.param` for
the same five network variants. Keep fixture parametrization in this form for
pytest 9 compatibility; do not skip tests to work around collection errors.

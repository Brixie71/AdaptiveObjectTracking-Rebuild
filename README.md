# Adaptive Object Tracking Rebuild

This folder is the active rebuild workspace for the thesis system.

`ThesisV2` remains outside this folder and is treated as read-only historical/source-of-truth material.

Start with `REBUILD_PREREQUISITES.md` before writing implementation code.

## Gate 1 development environment

The initial acceptance platform is Windows 10 or 11 x64 with Python 3.11 x64. Use a project virtual environment rather than the system Python.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Select and install the correct `torch` and `torchvision` pair for the actual machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Do not guess a CUDA wheel. Install that pair before the remaining project requirements, then install the package in editable mode:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The committed examples contain no credentials. Copy `.env.example` to `.env` only when local RTSP sources need values; camera configuration stores environment-variable names, not URLs or passwords.

## Gate 1 verification

Run the focused automated suite and dependency consistency check:

```powershell
python -m pytest tests/unit/config tests/unit/telemetry tests/integration/test_check_environment.py -v
python -m pip check
```

Then run the machine diagnostic from the repository root:

```powershell
python scripts/check_environment.py
python scripts/check_environment.py --json
```

The diagnostic validates Windows/Python architecture, installed distributions, configuration, the model manifest, compute capability, and structured-log writability. A machine without usable CUDA may report a CPU-fallback warning, but startup must not crash. A missing or unloadable PyTorch installation is a failure because neither CPU nor GPU inference would be usable.

Gate 1 is accepted only when the test suite and `pip check` pass and both diagnostic commands exit successfully on the intended machine. Sanitized JSON may then be recorded as Gate 1 evidence.

# =============================================================================
# startup_diagnostics.py
# Purpose: Pre-flight environment snapshot — Python prefix, GPU VRAM, zombie detection
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
startup_diagnostics.py — Run immediately on startup, before any heavy imports.

Captures:
  - Timestamp, sys.executable, sys.prefix / sys.base_prefix
  - VIRTUAL_ENV / CONDA_PREFIX / CONDA_DEFAULT_ENV env vars
  - Whether sys.executable lives under the expected venv prefix
  - nvidia-smi VRAM summary (total / free per GPU)
  - Per-process GPU memory consumers (zombie detection)
  - pip show output for critical packages (confirms install location)

All operations are wrapped in try/except so this module NEVER raises.
Uses stdlib only — no torch, no transformers, no psutil required.

Usage
-----
    import startup_diagnostics
    startup_diagnostics.run()                      # writes startup.log
    startup_diagnostics.run(log_file="my.log")     # custom path
    startup_diagnostics.run(skip=True)             # no-op (CI bypass)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

EXPECTED_VENV_PREFIX = "/venv/main"


def run(log_file: str = "startup.log", *, skip: bool = False) -> None:
    """Run startup diagnostics: write log file and print compact summary.

    Args:
        log_file: Path for the written log (relative to CWD or absolute).
        skip:     If True, silently return without doing anything. Used for
                  CI/automated runs where the check adds no value.
    """
    if skip:
        return

    try:
        _run_impl(log_file)
    except Exception as exc:
        # Startup diagnostics must NEVER crash the pipeline.
        print(f"[startup] diagnostics failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Implementation — internal helpers
# ---------------------------------------------------------------------------


def _run_impl(log_file: str) -> None:
    """Collect diagnostics, write log, print summary."""
    lines: list[str] = []  # log file lines

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    _append(lines, f"# startup_diagnostics — {now_iso}")
    _append(lines, "")

    # -------------------------------------------------------------------
    # 1. Python environment
    # -------------------------------------------------------------------
    executable = sys.executable
    prefix = getattr(sys, "prefix", "?")
    base_prefix = getattr(sys, "base_prefix", "?")
    real_prefix = getattr(sys, "real_prefix", None)  # set by virtualenv

    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    in_venv_main = EXPECTED_VENV_PREFIX in executable

    _append(lines, "## Python Environment")
    _append(lines, f"  executable    : {executable}")
    _append(lines, f"  sys.prefix    : {prefix}")
    _append(lines, f"  sys.base_prefix: {base_prefix}")
    if real_prefix:
        _append(lines, f"  sys.real_prefix: {real_prefix}")
    _append(lines, f"  VIRTUAL_ENV   : {virtual_env or '(not set)'}")
    _append(lines, f"  CONDA_PREFIX  : {conda_prefix or '(not set)'}")
    _append(lines, f"  CONDA_DEFAULT_ENV: {conda_env or '(not set)'}")
    _append(lines, f"  in /venv/main : {in_venv_main}")
    _append(lines, "")

    # Detect prefix mismatch
    prefix_mismatch = False
    active_prefix = virtual_env or conda_prefix
    if active_prefix and active_prefix not in executable and active_prefix not in prefix:
        prefix_mismatch = True
        _append(lines, f"  WARNING: sys.executable is NOT under activated prefix ({active_prefix})")
        _append(lines, "  WARNING: pip installs will go to the wrong location!")
        _append(lines, "")

    # -------------------------------------------------------------------
    # 2. nvidia-smi — total/free VRAM per GPU
    # -------------------------------------------------------------------
    _append(lines, "## GPU VRAM (nvidia-smi)")
    gpu_summary, zombie_pids = _collect_gpu_info(lines)

    # -------------------------------------------------------------------
    # 3. pip show — package install locations
    # -------------------------------------------------------------------
    _append(lines, "## Package Locations (pip show)")
    _collect_pip_show(lines)

    # -------------------------------------------------------------------
    # Write log file
    # -------------------------------------------------------------------
    try:
        log_path = Path(log_file)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        # Can't write log; still print summary
        print(f"[startup] WARNING: could not write {log_file}: {exc}")

    # -------------------------------------------------------------------
    # Print compact summary to stdout
    # -------------------------------------------------------------------
    _print_summary(
        executable=executable,
        prefix=prefix,
        active_prefix=active_prefix,
        prefix_mismatch=prefix_mismatch,
        in_venv_main=in_venv_main,
        gpu_summary=gpu_summary,
        zombie_pids=zombie_pids,
    )


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _collect_gpu_info(lines: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Run nvidia-smi; parse VRAM and zombie processes.

    Returns:
        gpu_summary: Human-readable VRAM string for the compact summary.
        zombie_pids: List of (pid, used_mb) tuples for processes on GPU.
    """
    zombie_pids: list[tuple[int, int]] = []
    gpu_summary = "N/A (nvidia-smi not available)"

    # --- Main query: memory per GPU ---
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            gpu_lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
            if gpu_lines:
                summaries = []
                for gpu_line in gpu_lines:
                    parts = [p.strip() for p in gpu_line.split(",")]
                    if len(parts) >= 5:
                        idx, name, used_mb, total_mb, free_mb = parts[:5]
                        try:
                            used_gb = int(used_mb) / 1024
                            total_gb = int(total_mb) / 1024
                            summaries.append(f"GPU {idx} ({name}): {used_gb:.1f}/{total_gb:.1f} GB used")
                        except ValueError:
                            summaries.append(gpu_line)
                    else:
                        summaries.append(gpu_line)
                    _append(lines, f"  {gpu_line}")
                gpu_summary = "; ".join(summaries)
            else:
                _append(lines, "  (no GPUs detected)")
                gpu_summary = "no GPUs detected"
        else:
            _append(lines, f"  nvidia-smi exited with code {result.returncode}")
            if result.stderr:
                _append(lines, f"  stderr: {result.stderr.strip()[:200]}")
    except FileNotFoundError:
        _append(lines, "  nvidia-smi not found (CPU-only environment)")
        gpu_summary = "N/A (nvidia-smi not found)"
    except subprocess.TimeoutExpired:
        _append(lines, "  nvidia-smi timed out")
    except Exception as exc:
        _append(lines, f"  error running nvidia-smi: {exc}")

    _append(lines, "")

    # --- Per-process query: find zombies ---
    _append(lines, "## GPU Compute Processes (nvidia-smi --query-compute-apps)")
    try:
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc_result.returncode == 0:
            proc_lines = [ln.strip() for ln in proc_result.stdout.strip().splitlines() if ln.strip()]
            if proc_lines:
                for pline in proc_lines:
                    _append(lines, f"  {pline}")
                    parts = [p.strip() for p in pline.split(",")]
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[0])
                            used_mb = int(parts[1])
                            zombie_pids.append((pid, used_mb))
                        except ValueError:
                            pass
            else:
                _append(lines, "  (no compute processes)")
        else:
            _append(lines, "  (query-compute-apps not supported or no processes)")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
        _append(lines, f"  error: {exc}")

    _append(lines, "")
    return gpu_summary, zombie_pids


# ---------------------------------------------------------------------------
# pip show helper
# ---------------------------------------------------------------------------

_PIP_SHOW_PACKAGES = ["transformers", "datasets", "accelerate"]


def _collect_pip_show(lines: list[str]) -> None:
    """Run `pip show` for critical packages and record install locations."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show"] + _PIP_SHOW_PACKAGES,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            for pip_line in result.stdout.splitlines():
                _append(lines, f"  {pip_line}")
        else:
            _append(lines, "  (pip show failed or packages not installed)")
            if result.stderr:
                _append(lines, f"  stderr: {result.stderr.strip()[:200]}")
    except Exception as exc:
        _append(lines, f"  error running pip show: {exc}")
    _append(lines, "")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(
    *,
    executable: str,
    prefix: str,
    active_prefix: str,
    prefix_mismatch: bool,
    in_venv_main: bool,
    gpu_summary: str,
    zombie_pids: list[tuple[int, int]],
) -> None:
    """Print a compact startup summary to stdout."""
    print(f"[startup] Python  : {executable}")

    if prefix_mismatch:
        print(
            f"[startup] WARNING : sys.executable NOT under {active_prefix!r} — "
            "pip will install to wrong prefix!"
        )
    elif active_prefix:
        match_sym = "✓" if (active_prefix in executable or active_prefix in prefix) else "~"
        print(f"[startup] Prefix  : {prefix}  ← {active_prefix} {match_sym}")
    else:
        print(f"[startup] Prefix  : {prefix}")

    # GPU line
    if zombie_pids:
        biggest_pid, biggest_mb = max(zombie_pids, key=lambda t: t[1])
        biggest_gb = biggest_mb / 1024
        all_used_gb = sum(mb for _, mb in zombie_pids) / 1024
        print(
            f"[startup] GPU VRAM: {all_used_gb:.1f} GB used — "
            f"ZOMBIE PROCESS DETECTED (pid {biggest_pid}, {biggest_gb:.1f} GB)"
        )
        print(f"[startup] FATAL   : Kill zombie before running: kill {biggest_pid}")
    else:
        print(f"[startup] GPU VRAM: {gpu_summary}")
        print("[startup] Zombies : none")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _append(lines: list[str], text: str) -> None:
    """Append a line to the log accumulator."""
    lines.append(text)


# ---------------------------------------------------------------------------
# Allow running standalone for debugging
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()

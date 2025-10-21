#!/usr/bin/env python3
"""Portable runner for AI Co-Scientist.

This replaces the previous bash implementation so the script can be executed
via `python3 run.sh ...` without syntax issues. It mirrors the original
behaviour: managing the virtual environment and delegating to the package
entrypoints.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR
PARENT_DIR = PKG_DIR.parent
VENV_DIR = PKG_DIR / ".venv"
SCRIPT_NAME = Path(sys.argv[0] or "run.sh").name


def print_usage() -> None:
    usage = f"""\
Usage: {SCRIPT_NAME} <command> [args]

Commands:
  setup                       Create venv and install requirements
  interactive [args...]       Start REPL (passes args to CLI)
  batch [args...]             Run non-interactive (passes args to CLI)
  help                        Show this help

Common args (passed through to CLI):
  --goal "..."                Research goal
  --max-iterations N          Supervisor iteration cap (default 20)
  --model NAME                Supervisor model (default gpt-5)
  --temperature T             Supervisor temperature (default 0.2)
  --worker-model NAME         Worker model for non-supervisor nodes
  --worker-temperature T      Worker temperature
  --input-json PATH           Initial state JSON
  --literature-file PATH      Literature text file
  --lit-chronology-file PATH  Chronology text file (articles_with_reasoning_text)

Examples:
  {SCRIPT_NAME} setup
  {SCRIPT_NAME} interactive --goal "ALS mechanism" --max-iterations 5
  {SCRIPT_NAME} batch --goal "Plant growth vs sunlight" --max-iterations 3
"""
    print(textwrap.dedent(usage).strip())


def resolve_python_bin() -> str:
    override = os.environ.get("PYTHON_BIN")
    if override:
        return override
    if sys.executable:
        return sys.executable
    return "python3"


def ensure_python() -> str:
    python_bin = resolve_python_bin()
    if os.path.isabs(python_bin) or os.sep in python_bin:
        python_path = Path(python_bin)
        if not python_path.exists():
            print(
                f"ERROR: Python interpreter not found at {python_bin}. "
                "Set PYTHON_BIN or install Python 3.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return str(python_path)
    resolved = shutil.which(python_bin)
    if resolved is None:
        print(
            f"ERROR: {python_bin} not found. Set PYTHON_BIN or install Python 3.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return resolved


def ensure_venv(python_bin: str) -> None:
    if not VENV_DIR.exists():
        print(f"[setup] Creating virtual environment at {VENV_DIR}")
        run_cmd([python_bin, "-m", "venv", str(VENV_DIR)])


def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def install_requirements() -> None:
    print("[setup] Installing/upgrading pip and dependencies")
    venv_python = venv_python_path()
    if not venv_python.exists():
        print(
            f"ERROR: Expected virtualenv python at {venv_python} but it was missing.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    run_cmd(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        env=env,
    )
    run_cmd([str(venv_python), "-m", "pip", "install", "-r", str(PKG_DIR / "requirements.txt")], env=env)


def cmd_setup() -> None:
    python_bin = ensure_python()
    ensure_venv(python_bin)
    install_requirements()
    print("[setup] Done. Create a .env from ENV.sample and fill your keys if needed.")


def run_cli(mode: str, args: Iterable[str]) -> None:
    python_bin = ensure_python()
    ensure_venv(python_bin)
    venv_python = venv_python_path()
    if not venv_python.exists():
        print(
            f"ERROR: Virtual environment at {VENV_DIR} looks incomplete. "
            "Re-run the setup command.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    cmd = [str(venv_python), "-m", "co_scientist_langgraph"]
    if mode == "interactive":
        cmd.append("--interactive")
    cmd.extend(args)
    run_cmd(cmd, cwd=PARENT_DIR)


def run_cmd(cmd: Iterable[str], cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    cmd_list = list(cmd)
    try:
        subprocess.run(cmd_list, cwd=str(cwd) if cwd else None, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def main(argv: list[str]) -> int:
    if not argv:
        print_usage()
        return 0
    cmd, *args = argv
    if cmd in {"help", "-h", "--help"}:
        print_usage()
        return 0
    if cmd == "setup":
        cmd_setup()
        return 0
    if cmd in {"interactive", "repl"}:
        run_cli("interactive", args)
        return 0
    if cmd in {"batch", "run"}:
        run_cli("batch", args)
        return 0
    print(f"Unknown command: {cmd}", file=sys.stderr)
    print_usage()
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

#!/usr/bin/env bash
set -euo pipefail

# Portable runner for AI Co-Scientist
# - Creates/uses a local venv
# - Installs requirements when needed
# - Runs the package from its parent directory so module imports resolve

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$SCRIPT_DIR"
PARENT_DIR="$(dirname "$PKG_DIR")"
VENV_DIR="$PKG_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

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
  $(basename "$0") setup
  $(basename "$0") interactive --goal "ALS mechanism" --max-iterations 5
  $(basename "$0") batch --goal "Plant growth vs sunlight" --max-iterations 3
EOF
}

ensure_python() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Set PYTHON_BIN or install Python 3." >&2
    exit 1
  fi
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

install_requirements() {
  echo "[setup] Installing/upgrading pip and dependencies"
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  python -m pip install -r "$PKG_DIR/requirements.txt"
}

cmd_setup() {
  ensure_python
  ensure_venv
  activate_venv
  install_requirements
  echo "[setup] Done. Create a .env from ENV.sample and fill your keys if needed."
}

run_cli() {
  local mode="$1"; shift || true
  ensure_python
  ensure_venv
  activate_venv
  # Run from parent so 'python -m co_scientist_langgraph' resolves the package
  pushd "$PARENT_DIR" >/dev/null
  if [[ "$mode" == "interactive" ]]; then
    python -m co_scientist_langgraph --interactive "$@"
  else
    python -m co_scientist_langgraph "$@"
  fi
  popd >/dev/null
}

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    setup)
      cmd_setup
      ;;
    interactive|repl)
      run_cli interactive "$@"
      ;;
    batch|run)
      run_cli batch "$@"
      ;;
    help|-h|--help)
      print_usage
      ;;
    *)
      echo "Unknown command: $cmd" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"



#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3 first." >&2
  exit 1
fi

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Error: requirements file not found at '$REQ_FILE'." >&2
  exit 1
fi

echo "Creating virtual environment at '$VENV_DIR'..."
python3 -m venv "$VENV_DIR"

echo "Activating venv and installing requirements from '$REQ_FILE'..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo
echo "Done."
echo "Activate with:"
echo "  source \"$VENV_DIR/bin/activate\""

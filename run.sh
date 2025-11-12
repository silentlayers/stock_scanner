#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the Streamlit app
# - Creates a local venv in .venv if missing
# - Installs/updates requirements
# - Runs Streamlit with any extra args you pass to this script

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python3"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+." >&2
  exit 1
fi

# Create venv if needed
if [ ! -d .venv ]; then
  echo "Creating virtual environment at .venv"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip tooling and install deps
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# Enable insecure transport for localhost OAuth (handled in config.py as well, but set here defensively)
export OAUTHLIB_INSECURE_TRANSPORT=1

# Run Streamlit (forward any extra args, e.g. --server.port 8502)
exec streamlit run streamlit_app.py "$@"

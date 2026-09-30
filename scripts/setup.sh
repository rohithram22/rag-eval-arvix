#!/usr/bin/env bash
# Creates .venv and installs dependencies (CPU-only PyTorch).
# First run: resolves requirements.txt and writes requirements.lock.txt.
# Later runs / fresh clones: installs the exact pinned versions from the lock file.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

if [ ! -d .venv ]; then
  echo ">> Creating virtual environment (.venv) with $($PY --version)"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel --quiet

if [ -f requirements.lock.txt ]; then
  echo ">> Installing pinned versions from requirements.lock.txt"
  pip install -r requirements.lock.txt --extra-index-url "$TORCH_CPU_INDEX"
else
  echo ">> Installing CPU-only torch (avoids multi-GB CUDA wheels)"
  pip install torch --index-url "$TORCH_CPU_INDEX"
  echo ">> Installing project requirements"
  pip install -r requirements.txt
  echo ">> Writing requirements.lock.txt"
  pip freeze --exclude-editable > requirements.lock.txt
fi

echo ">> Installing the rag_eval package in editable mode"
pip install -e . --quiet

echo
echo "Done. Activate with:  source .venv/bin/activate"

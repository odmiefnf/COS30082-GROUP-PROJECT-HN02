#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY=""
if [ -x ".venv-gpu/bin/python" ]; then
  PY=".venv-gpu/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  echo "[Setup] No virtual environment found. Creating .venv..."
  SYS_PY=""
  for cmd in python3.11 python3.12 python3.13 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
      SYS_PY="$cmd"
      break
    fi
  done
  if [ -z "$SYS_PY" ]; then
    echo "[Error] Python 3.10+ was not found. Install Python 3.11 or 3.12 and try again." >&2
    exit 2
  fi
  "$SYS_PY" -m venv .venv
  PY=".venv/bin/python"
fi

if ! "$PY" -c "import torch, torchvision, cv2, PIL, numpy" >/dev/null 2>&1; then
  echo "[Setup] Installing required packages from requirements-webcam.txt..."
  echo "This only runs once and may take a few minutes."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements-webcam.txt
fi

exec "$PY" -m attendance_app.web --open "$@"

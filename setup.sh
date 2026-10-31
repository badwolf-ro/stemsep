#!/usr/bin/env bash
# Build stemsep's venv: python 3.12 + torch 2.11 + MSST, pinned to the set in requirements.lock.
# Needs uv (https://docs.astral.sh/uv/). Model weights are not installed here - stemsep
# downloads each one on first use, or pre-fetch with: ./stemsep models --pull leap bandit-v2
#
#   ./setup.sh                 # torch build matching your NVIDIA driver (uv --torch-backend=auto)
#   TORCH_BACKEND=cu128 ./setup.sh   # force one (cu128 is required for RTX 50-series / sm_120)
set -euo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v uv >/dev/null || { echo "setup: uv not found - https://docs.astral.sh/uv/" >&2; exit 1; }
command -v ffmpeg >/dev/null || echo "setup: warning - ffmpeg not on PATH (needed, with libsoxr)" >&2
uv venv --python 3.12 "$S/.venv"
uv pip install --python "$S/.venv/bin/python" --torch-backend="${TORCH_BACKEND:-auto}" \
  -r "$S/requirements.lock"
echo "done. try: $S/stemsep models"

#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 app.py "$@"

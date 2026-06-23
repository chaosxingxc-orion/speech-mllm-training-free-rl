#!/usr/bin/env bash
set -euo pipefail
source "${SPEECHRL_VENV:-$HOME/.venvs/speechrl}/bin/activate"

echo "=== hf CLI ==="
command -v hf || echo "NOT installed"

echo "=== modelscope ==="
command -v modelscope || echo "NOT installed"

echo "=== hf_transfer ==="
python3 -c "import hf_transfer; print('installed:', hf_transfer.__version__)" 2>/dev/null || echo "NOT installed"

echo "=== aria2c ==="
command -v aria2c || echo "NOT installed"

echo "=== huggingface_hub ==="
python3 -c "import huggingface_hub; print('installed:', huggingface_hub.__version__)" 2>/dev/null || echo "NOT installed"

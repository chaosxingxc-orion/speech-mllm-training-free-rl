#!/usr/bin/env bash
set -euo pipefail
# Activate the shared WSL2 venv (see ../../docs/setup.md).
source "${SPEECHRL_VENV:-$HOME/.venvs/speechrl}/bin/activate"
cd "$(dirname "$0")/.."
python -m training_free_rl.main "$@"

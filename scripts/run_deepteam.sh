#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-deepteam"
MODE="${1:-python}"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "python3.13 is required for deepteam (supports Python <3.14)."
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3.13 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install -r "${ROOT_DIR}/requirements.txt"

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

if [ -z "${OPENROUTER_KEY:-}" ]; then
  echo "OPENROUTER_KEY is missing. Copy .env.example to .env and set OPENROUTER_KEY."
  exit 1
fi

if [ "${MODE}" = "cli" ]; then
  deepteam run "${ROOT_DIR}/deepteam.yaml"
else
  python "${ROOT_DIR}/red_team_llm.py"
fi

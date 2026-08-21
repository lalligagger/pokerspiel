#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/app/cfg/solve_config_debug.json}"

if [[ -f "$CONFIG_PATH" ]]; then
  eval "$(python3 /app/config_env.py --config "$CONFIG_PATH" --format shell)"
fi

exec uvicorn api.app:app --host 0.0.0.0 --port 8080

#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/app/cfg/solve_config_local.json}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"

if [[ -f "$CONFIG_PATH" ]]; then
  echo "==> as-run settings"
  python3 -m json.tool "$CONFIG_PATH"
  eval "$(python3 /app/config_env.py --config "$CONFIG_PATH" --format shell)"
fi

python3 /app/solver_live_dashboard.py \
  --api-base-url http://0.0.0.0:8080 \
  --interval 300 \
  --request-timeout 120 \
  --serve-host 0.0.0.0 \
  --serve-port "$DASHBOARD_PORT" &
DASHBOARD_PID=$!
trap 'kill "$DASHBOARD_PID" >/dev/null 2>&1 || true' EXIT

printf '\nsummary: http://0.0.0.0:8080/summary\n'
printf 'api docs: http://0.0.0.0:8080/docs\n'
printf 'dashboard: http://0.0.0.0:%s\n' "$DASHBOARD_PORT"

exec uvicorn api.app:app --host 0.0.0.0 --port 8080

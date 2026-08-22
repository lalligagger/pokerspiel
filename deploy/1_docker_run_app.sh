#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$(python3 - "${1:-${CONFIG_PATH:-}}" <<'PY'
import json, sys
from pathlib import Path
path = sys.argv[1]
if not path or not Path(path).exists():
    print('pokerspiel')
    raise SystemExit
cfg = json.loads(Path(path).read_text(encoding='utf-8'))
print(cfg.get('project', 'pokerspiel'))
PY
)}"
ZONE="${ZONE:-$(python3 - "${1:-${CONFIG_PATH:-}}" <<'PY'
import json, sys
from pathlib import Path
path = sys.argv[1]
if not path or not Path(path).exists():
    print('us-west1-b')
    raise SystemExit
cfg = json.loads(Path(path).read_text(encoding='utf-8'))
print(cfg.get('zone', 'us-west1-b'))
PY
)}"
INSTANCE_NAME="${INSTANCE_NAME:-$(python3 - "${1:-${CONFIG_PATH:-}}" <<'PY'
import json, sys
from pathlib import Path
path = sys.argv[1]
if not path or not Path(path).exists():
    print('instance-20260818-234442')
    raise SystemExit
cfg = json.loads(Path(path).read_text(encoding='utf-8'))
print(cfg.get('instance', cfg.get('instance_name', 'instance-20260818-234442')))
PY
)}"
APP_PORT="${APP_PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"
REPO_DIR="${REPO_DIR:-$HOME/pokerspiel}"
BRANCH="${BRANCH:-$(python3 - "${1:-${CONFIG_PATH:-}}" <<'PY'
import json, sys
from pathlib import Path
path = sys.argv[1]
if not path or not Path(path).exists():
    print('postflop-redux')
    raise SystemExit
cfg = json.loads(Path(path).read_text(encoding='utf-8'))
print(cfg.get('branch', cfg.get('git_branch', 'postflop-redux')))
PY
)}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
CONFIG_PATH="${1:-${CONFIG_PATH:-}}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config.json>" >&2
  echo "Example: $0 cfg/solve_config_gce.json" >&2
  exit 1
fi

DOCKER_ENV_ARGS="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
env_map = {
    'POKERSPIEL_SOLVER': cfg.get('solver'),
    'POKERSPIEL_PRESET': cfg.get('preset'),
    'POKERSPIEL_RANGE_SAMPLES': cfg.get('range_samples'),
    'POKERSPIEL_POSTFLOP_SAMPLES': cfg.get('postflop_samples'),
    'POKERSPIEL_MIN_ITERATIONS': cfg.get('min_iterations'),
    'POKERSPIEL_CHECKPOINT_EVERY': cfg.get('checkpoint_every') if cfg.get('checkpoint_every') is not None else cfg.get('stability_checkpoint'),
    'POKERSPIEL_MEMORY_THRESHOLD': cfg.get('memory_threshold'),
    'POKERSPIEL_OUTPUT_JSON': cfg.get('output_json'),
}
parts = []
for key, value in env_map.items():
    if value is None:
        continue
    parts.append(f"-e {key}={value}")
print(' '.join(parts))
PY
)"

echo "==> Launching app on GCE instance: $INSTANCE_NAME"
echo "==> Target branch: $BRANCH"

ssh_ready=0
for attempt in $(seq 1 20); do
  if gcloud compute ssh "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --command='true' >/dev/null 2>&1; then
    ssh_ready=1
    break
  fi
  echo "==> Waiting for SSH on $INSTANCE_NAME (attempt $attempt/20)..."
  sleep 10
done

if [[ "$ssh_ready" != "1" ]]; then
  echo "SSH did not become available for $INSTANCE_NAME" >&2
  exit 1
fi

remote_script=$(cat <<REMOTE
set -eux

BRANCH="${BRANCH:-postflop-redux}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"
APP_PORT="${APP_PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
export BRANCH REMOTE_NAME IMAGE_NAME APP_PORT DASHBOARD_PORT

sudo apt-get update
sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "\$USER"

newgrp docker <<'REMOTE_BLOCK'
set -eux
cd "\$HOME"

REPO_DIR="\${REPO_DIR:-\$HOME/pokerspiel}"

# Force-reset any stale repo state from prior deployments / solver runs while keeping the VM and IP intact.
sudo find "\$HOME" -maxdepth 2 -name "pokerspiel*" -exec rm -rf {} +
sudo find /tmp -maxdepth 2 -iname "pokerspiel*" -exec rm -rf {} +
if [ -d "\$REPO_DIR" ]; then
  sudo rm -rf "\$REPO_DIR"
fi

git clone https://github.com/lalligagger/pokerspiel.git "\$REPO_DIR"
cd "\$REPO_DIR"

git fetch "\$REMOTE_NAME" --prune
git checkout -B "\$BRANCH" "\$REMOTE_NAME/\$BRANCH"
git reset --hard "\$REMOTE_NAME/\$BRANCH"
git pull --ff-only "\$REMOTE_NAME" "\$BRANCH"

docker build -t "\$IMAGE_NAME" .

docker rm -f "\$IMAGE_NAME" >/dev/null 2>&1 || true

# Remove any other container already publishing these ports so the app can restart cleanly.
docker ps --filter "publish=\${APP_PORT}" -q | while read -r cid; do
  [ -n "\$cid" ] && docker rm -f "\$cid" >/dev/null 2>&1 || true
done
docker ps --filter "publish=\${DASHBOARD_PORT}" -q | while read -r cid; do
  [ -n "\$cid" ] && docker rm -f "\$cid" >/dev/null 2>&1 || true
done

if ss -lnt 2>/dev/null | grep -Eq "(:|\[::\]):\${APP_PORT} "; then
  echo "Port \${APP_PORT} is occupied by another process. Stop it or set APP_PORT to a free port." >&2
  exit 1
fi
if ss -lnt 2>/dev/null | grep -Eq "(:|\[::\]):\${DASHBOARD_PORT} "; then
  echo "Port \${DASHBOARD_PORT} is occupied by another process. Stop it or set DASHBOARD_PORT to a free port." >&2
  exit 1
fi

docker run -d \
  --name "\$IMAGE_NAME" \
  --restart unless-stopped \
  -p "\$APP_PORT:\$APP_PORT" \
  -p "\$DASHBOARD_PORT:\$DASHBOARD_PORT" \
  ${DOCKER_ENV_ARGS} \
  -v "\$HOME/pokerspiel:/app" \
  -w /app \
  "\$IMAGE_NAME" \
  bash -lc "set -eux; CONFIG_PATH=\${CONFIG_PATH:-./cfg/solve_config_local.json}; if [ -f \"\$CONFIG_PATH\" ]; then echo '==> as-run settings'; python3 -m json.tool \"\$CONFIG_PATH\"; eval \"\$(python3 /app/config_env.py --config \"\$CONFIG_PATH\" --format shell)\"; fi; python3 /app/solver_live_dashboard.py --api-base-url http://0.0.0.0:${APP_PORT} --interval 300 --request-timeout 120 --serve-host 0.0.0.0 --serve-port ${DASHBOARD_PORT} > /tmp/pokerspiel-dashboard.log 2>&1 & DASHBOARD_PID=\$!; trap 'kill \"\$DASHBOARD_PID\" >/dev/null 2>&1 || true' EXIT; printf '\nsummary: http://0.0.0.0:${APP_PORT}/summary\n'; printf 'api docs: http://0.0.0.0:${APP_PORT}/docs\n'; printf 'dashboard: http://0.0.0.0:${DASHBOARD_PORT}\n'; exec uvicorn api.app:app --host 0.0.0.0 --port ${APP_PORT}"

echo "==> container started on localhost:\$APP_PORT"
echo "==> check: curl http://127.0.0.1:\$APP_PORT/status"
echo "==> dashboard: http://127.0.0.1:\$DASHBOARD_PORT"
REMOTE_BLOCK
REMOTE
)

gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="$remote_script"

EXTERNAL_IP="$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"

echo
printf 'SUMMARY_URL=http://%s:%s/summary\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'STATUS_URL=http://%s:%s/status\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'DOCS_URL=http://%s:%s/docs\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'DASHBOARD_URL=http://%s:%s\n' "$EXTERNAL_IP" "$DASHBOARD_PORT"

echo
echo "==> App started. Next step: ./deploy/2_api_ip_config.sh"
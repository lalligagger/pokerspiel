#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-pokerspiel}"
ZONE="${ZONE:-us-west1-b}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260818-234442}"
APP_PORT="${APP_PORT:-8080}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"
REPO_DIR="${REPO_DIR:-$HOME/pokerspiel}"
BRANCH="${BRANCH:-postflop-redux}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
CONFIG_PATH="${1:-${CONFIG_PATH:-}}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "Usage: $0 <config.json>" >&2
  echo "Example: $0 cfg/solve_config_light.json" >&2
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
    'POKERSPIEL_STABILITY_THRESHOLD': cfg.get('stability_threshold'),
    'POKERSPIEL_STOP_PATIENCE': cfg.get('stop_patience'),
    'POKERSPIEL_MIN_ITERATIONS': cfg.get('min_iterations'),
    'POKERSPIEL_CHECKPOINT_EVERY': cfg.get('checkpoint_every') if cfg.get('checkpoint_every') is not None else cfg.get('stability_checkpoint'),
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
export BRANCH REMOTE_NAME IMAGE_NAME APP_PORT

sudo apt-get update
sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "\$USER"

newgrp docker <<'REMOTE_BLOCK'
set -eux
cd "\$HOME"

if [ ! -d pokerspiel ]; then
  git clone --branch "\$BRANCH" --single-branch https://github.com/lalligagger/pokerspiel pokerspiel
fi

cd pokerspiel

git fetch "\$REMOTE_NAME" --prune
if git rev-parse --verify "\$BRANCH" >/dev/null 2>&1; then
  git checkout "\$BRANCH"
else
  git checkout -b "\$BRANCH" "\$REMOTE_NAME/\$BRANCH"
fi
git reset --hard "\$REMOTE_NAME/\$BRANCH"
git pull --ff-only "\$REMOTE_NAME" "\$BRANCH"

docker build -t "\$IMAGE_NAME" .

docker rm -f "\$IMAGE_NAME" >/dev/null 2>&1 || true

# Remove any other container already publishing this port so the app can restart cleanly.
docker ps --filter "publish=\${APP_PORT}" -q | while read -r cid; do
  [ -n "\$cid" ] && docker rm -f "\$cid" >/dev/null 2>&1 || true
done

if ss -lnt 2>/dev/null | grep -Eq "(:|\[::\]):\${APP_PORT} "; then
  echo "Port \${APP_PORT} is occupied by another process. Stop it or set APP_PORT to a free port." >&2
  exit 1
fi

docker run -d \
  --name "\$IMAGE_NAME" \
  --restart unless-stopped \
  -p "\$APP_PORT:\$APP_PORT" \
  ${DOCKER_ENV_ARGS} \
  -v "\$HOME/pokerspiel:/app" \
  -w /app \
  "\$IMAGE_NAME" \
  uvicorn api.app:app --host 0.0.0.0 --port "\$APP_PORT"

echo "==> container started on localhost:\$APP_PORT"
echo "==> check: curl http://127.0.0.1:\$APP_PORT/status"
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
printf 'STATUS_URL=http://%s:%s/status\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'DOCS_URL=http://%s:%s/docs\n' "$EXTERNAL_IP" "$APP_PORT"

echo
echo "==> App started. Next step: ./deploy/2_api_ip_config.sh"
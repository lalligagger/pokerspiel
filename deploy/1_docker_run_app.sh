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
MIN_ITERATIONS="${MIN_ITERATIONS:-1000}"

echo "==> Launching app on GCE instance: $INSTANCE_NAME"
echo "==> Target branch: $BRANCH"

gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command='
set -eux

BRANCH="${BRANCH:-postflop-redux}"
REMOTE_NAME="${REMOTE_NAME:-origin}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"
APP_PORT="${APP_PORT:-8080}"
MIN_ITERATIONS="${MIN_ITERATIONS:-1000}"
export BRANCH REMOTE_NAME IMAGE_NAME APP_PORT MIN_ITERATIONS

sudo apt-get update
sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

newgrp docker <<'"'"'REMOTE'"'"'
set -eux
cd "$HOME"

if [ ! -d pokerspiel ]; then
  git clone --branch "$BRANCH" --single-branch https://github.com/lalligagger/pokerspiel pokerspiel
fi

cd pokerspiel

git fetch "$REMOTE_NAME" --prune
if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout "$BRANCH"
else
  git checkout -b "$BRANCH" "$REMOTE_NAME/$BRANCH"
fi
git reset --hard "$REMOTE_NAME/$BRANCH"
git pull --ff-only "$REMOTE_NAME" "$BRANCH"

docker build -t "$IMAGE_NAME" .

docker rm -f "$IMAGE_NAME" >/dev/null 2>&1 || true

# Remove any other container already publishing this port so the app can restart cleanly.
docker ps --filter "publish=${APP_PORT}" -q | while read -r cid; do
  [ -n "$cid" ] && docker rm -f "$cid" >/dev/null 2>&1 || true
done

if ss -lnt 2>/dev/null | grep -Eq "(:|\[::\]):${APP_PORT} "; then
  echo "Port ${APP_PORT} is occupied by another process. Stop it or set APP_PORT to a free port." >&2
  exit 1
fi

docker run -d \
  --name "$IMAGE_NAME" \
  --restart unless-stopped \
  -p "$APP_PORT:$APP_PORT" \
  -e POKERSPIEL_RANGE_SAMPLES=250 \
  -e POKERSPIEL_MIN_ITERATIONS="$MIN_ITERATIONS" \
  -v "$HOME/pokerspiel:/app" \
  -w /app \
  "$IMAGE_NAME" \
  uvicorn api.app:app --host 0.0.0.0 --port "$APP_PORT"

echo "==> container started on localhost:$APP_PORT"
echo "==> check: curl http://127.0.0.1:$APP_PORT/status"
REMOTE
'

EXTERNAL_IP="$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"

echo
printf 'STATUS_URL=http://%s:%s/status\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'DOCS_URL=http://%s:%s/docs\n' "$EXTERNAL_IP" "$APP_PORT"

echo
echo "==> App started. Next step: ./deploy/2_api_ip_config.sh"
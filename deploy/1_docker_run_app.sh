#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-pokerspiel}"
ZONE="${ZONE:-us-west1-b}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260818-234442}"
APP_PORT="${APP_PORT:-8080}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"
REPO_DIR="${REPO_DIR:-$HOME/pokerspiel}"

echo "==> Launching app on GCE instance: $INSTANCE_NAME"

gcloud compute ssh "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command='
set -eux

sudo apt-get update
sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

newgrp docker <<'"'"'REMOTE'"'"'
set -eux
cd "$HOME"

if [ ! -d pokerspiel ]; then
  git clone https://github.com/lalligagger/pokerspiel pokerspiel
fi

cd pokerspiel
git pull --ff-only || true

docker build -t "$IMAGE_NAME" .

docker rm -f "$IMAGE_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$IMAGE_NAME" \
  --restart unless-stopped \
  -p "$APP_PORT:$APP_PORT" \
  -e POKERSPIEL_RANGE_SAMPLES=250 \
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
echo "==> App started. Next step: ./deploy/2_deploy_docker_w_ipcfg.sh"
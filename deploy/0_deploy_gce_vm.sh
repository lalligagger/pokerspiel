#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-pokerspiel}"
ZONE="${ZONE:-us-west1-b}"
INSTANCE="${INSTANCE:-instance-20260818-234442}"
REMOTE_DIR="${REMOTE_DIR:-/home/\$USER/pokerspiel}"
APP_PORT="${APP_PORT:-8080}"
IMAGE_NAME="${IMAGE_NAME:-pokerspiel-live}"

echo "==> Using project: $PROJECT"
echo "==> Using zone: $ZONE"
echo "==> Using instance: $INSTANCE"

# Create VM if it does not exist
if ! gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" >/dev/null 2>&1; then
  echo "==> Creating VM: $INSTANCE"
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type=e2-standard-2 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --tags=http-server,https-server
fi

# Firewall rule for the app
gcloud compute firewall-rules create allow-pokerspiel-app \
  --project="$PROJECT" \
  --allow=tcp:"$APP_PORT" \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server,https-server \
  --direction=INGRESS \
  2>/dev/null || true

# SSH into the VM and install Docker + repo + app
gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command="
    set -eux

    sudo apt-get update
    sudo apt-get install -y docker.io git

    sudo systemctl enable --now docker
    sudo usermod -aG docker \$USER
    newgrp docker <<'REMOTE'
set -eux
cd \$HOME

if [ ! -d pokerspiel ]; then
  git clone https://github.com/lalligagger/pokerspiel pokerspiel
fi

cd pokerspiel
git pull --ff-only || true

docker build -t $IMAGE_NAME .

docker stop $IMAGE_NAME >/dev/null 2>&1 || true
docker rm $IMAGE_NAME >/dev/null 2>&1 || true

docker run -d \
  --name $IMAGE_NAME \
  --restart unless-stopped \
  -p $APP_PORT:$APP_PORT \
  -v \$HOME/pokerspiel:/app \
  $IMAGE_NAME \
  uvicorn api.app:app --host 0.0.0.0 --port $APP_PORT

echo 'Container started'
echo 'Check with: curl http://127.0.0.1:$APP_PORT/status'
REMOTE
  "

echo
IP=$(gcloud compute instances describe "$INSTANCE" --project "$PROJECT" --zone "$ZONE" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
echo "==> App should be live on:"
echo "http://${IP}:$APP_PORT/status"
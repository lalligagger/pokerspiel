#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-pokerspiel}"
ZONE="${ZONE:-us-west1-b}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260818-234442}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-2}"
APP_PORT="${APP_PORT:-8080}"
BRANCH="${BRANCH:-postflop-redux}"

echo "==> Using project: $PROJECT"
echo "==> Using zone: $ZONE"
echo "==> Using instance: $INSTANCE_NAME"
echo "==> Using git branch: $BRANCH"

INSTANCE_ID="$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format='value(id)' 2>/dev/null || true)"

if [[ -z "$INSTANCE_ID" ]]; then
  echo "==> Creating VM: $INSTANCE_NAME"
  gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --tags=allow-pokerspiel-app

  INSTANCE_ID="$(gcloud compute instances describe "$INSTANCE_NAME" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --format='value(id)')"
else
  echo "==> VM already exists: $INSTANCE_NAME (instance id: $INSTANCE_ID)"
fi

EXTERNAL_IP="$(gcloud compute instances describe "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"

echo
printf 'INSTANCE_NAME=%s\n' "$INSTANCE_NAME"
printf 'INSTANCE_ID=%s\n' "$INSTANCE_ID"
printf 'EXTERNAL_IP=%s\n' "$EXTERNAL_IP"
printf 'GIT_BRANCH=%s\n' "$BRANCH"
printf 'STATUS_URL=http://%s:%s/status\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'DOCS_URL=http://%s:%s/docs\n' "$EXTERNAL_IP" "$APP_PORT"

echo
echo "==> VM ready. Next step: ./deploy/1_docker_run_app.sh"
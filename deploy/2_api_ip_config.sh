#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-pokerspiel}"
ZONE="${ZONE:-us-west1-b}"
INSTANCE_NAME="${INSTANCE_NAME:-instance-20260818-234442}"
APP_PORT="${APP_PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"

echo "==> Ensuring firewall allows public access to FastAPI on port $APP_PORT and dashboard on port $DASHBOARD_PORT"

gcloud compute instances add-tags "$INSTANCE_NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --tags=allow-pokerspiel-app 2>/dev/null || true

gcloud compute firewall-rules create allow-pokerspiel-api \
  --project="$PROJECT" \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:"$APP_PORT" \
  --source-ranges=0.0.0.0/0 \
  --target-tags=allow-pokerspiel-app \
  2>/dev/null || true

gcloud compute firewall-rules create allow-pokerspiel-dashboard \
  --project="$PROJECT" \
  --direction=INGRESS \
  --priority=1001 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:"$DASHBOARD_PORT" \
  --source-ranges=0.0.0.0/0 \
  --target-tags=allow-pokerspiel-app \
  2>/dev/null || true

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
printf 'FastAPI docs are now public on %s:%s\n' "$EXTERNAL_IP" "$APP_PORT"
printf 'Dashboard is now public on %s:%s\n' "$EXTERNAL_IP" "$DASHBOARD_PORT"

echo "==> You can verify with: curl -sS http://$EXTERNAL_IP:$APP_PORT/status"
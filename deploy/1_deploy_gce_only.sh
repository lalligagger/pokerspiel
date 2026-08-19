export PROJECT="pokerspiel"
export ZONE="us-west1-b"
export INSTANCE="instance-20260818-234442"
export APP_PORT="8080"

gcloud compute instances create "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server

gcloud compute firewall-rules create allow-pokerspiel-readonly \
  --project="$PROJECT" \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --allow=tcp:"$APP_PORT" \
  --source-ranges=0.0.0.0/0 \
  --target-tags=http-server,https-server \
  2>/dev/null || true
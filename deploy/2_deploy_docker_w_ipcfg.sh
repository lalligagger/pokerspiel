gcloud compute ssh "$INSTANCE" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --command='
set -eux
sudo apt-get update
sudo apt-get install -y docker.io git

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

docker build -t pokerspiel-live .

docker stop pokerspiel-live >/dev/null 2>&1 || true
docker rm pokerspiel-live >/dev/null 2>&1 || true

docker run -d \
  --name pokerspiel-live \
  --restart unless-stopped \
  -p 8080:8080 \
  pokerspiel-live \
  uvicorn api.app:app --host 0.0.0.0 --port 8080

echo "Container started"
echo "Check: curl http://127.0.0.1:8080/health"
REMOTE
'
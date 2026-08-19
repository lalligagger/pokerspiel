#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL_CONFIG_PATH="$ROOT_DIR/cfg/solve_config_light.json"
LEGACY_CONFIG_PATH="$ROOT_DIR/solve_config_light.json"
CONFIG_PATH="${1:-$CANONICAL_CONFIG_PATH}"
DEPLOY_TARGET="${2:-local}"

# Only the runner/CLI uses JSON profiles. The live FastAPI app does not read
# these config files directly; it owns its runtime state in memory.
if [[ ! -f "$CONFIG_PATH" && -f "$LEGACY_CONFIG_PATH" ]]; then
  CONFIG_PATH="$LEGACY_CONFIG_PATH"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

python3 - "$CONFIG_PATH" "$DEPLOY_TARGET" <<'PY'
import json
import os
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
deploy_target = sys.argv[2]

with config_path.open("r", encoding="utf-8") as fh:
    config = json.load(fh)

if deploy_target not in {"local", "gce"}:
    raise SystemExit(f"Unsupported deploy target: {deploy_target}")

if config.get("deploy") not in {None, deploy_target}:
    raise SystemExit(
        f"Config deploy target '{config.get('deploy')}' does not match requested target '{deploy_target}'"
    )

# Merge the config-driven solver CLI flags. The env-backed values that should
# be materialized at process startup live under `solver_env` and are generated
# at runtime by this runner.
args = [
    "python", "app_solver.py", config.get("mode", "hulh"),
    "--iterations", str(config.get("iterations", 100)),
    "--preset", config.get("preset", "hulh-preflop"),
    "--range-samples", str(config.get("range_samples", 1000)),
    "--stability-threshold", str(config.get("stability_threshold", 0.01)),
    "--stop-patience", str(config.get("stop_patience", 3)),
    "--min-iterations", str(config.get("min_iterations", 0)),
    "--solver", config.get("solver", "outcome"),
    "--report-mode", config.get("report_mode", "summary"),
    "--artifact-mode", config.get("artifact_mode", "lightweight"),
    "--range-last-n", str(config.get("range_last_n", 2000)),
    "--output-json", str(config.get("output_json", "./overnight_runs/runner/report.json")),
]

checkpoint_every = config.get("checkpoint_every")
stability_checkpoint = config.get("stability_checkpoint")
if checkpoint_every not in (None, 0):
    if stability_checkpoint in (None, 0):
        stability_checkpoint = checkpoint_every
    elif int(stability_checkpoint) != int(checkpoint_every):
        stability_checkpoint = checkpoint_every
args.extend(["--stability-checkpoint", str(stability_checkpoint)]) if stability_checkpoint not in (None, 0) else None
if config.get("checkpoint_history_limit") not in (None, 0):
    args.extend(["--checkpoint-history-limit", str(config["checkpoint_history_limit"])])

# Convert to JSON-serializable value for shell usage.
print(json.dumps({"deploy": deploy_target, "config": config, "argv": args}, separators=(",", ":")))
PY

JSON_OUT="$(python3 - "$CONFIG_PATH" "$DEPLOY_TARGET" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
deploy_target = sys.argv[2]
with config_path.open("r", encoding="utf-8") as fh:
    config = json.load(fh)

if deploy_target not in {"local", "gce"}:
    raise SystemExit(f"Unsupported deploy target: {deploy_target}")

if config.get("deploy") not in {None, deploy_target}:
    raise SystemExit(f"Config deploy target '{config.get('deploy')}' does not match requested target '{deploy_target}'")

args = [
    "python", "app_solver.py", config.get("mode", "hulh"),
    "--iterations", str(config.get("iterations", 100)),
    "--preset", config.get("preset", "hulh-preflop"),
    "--range-samples", str(config.get("range_samples", 1000)),
    "--postflop-samples", str(config.get("postflop_samples", 32)),
    "--stability-threshold", str(config.get("stability_threshold", 0.01)),
    "--stop-patience", str(config.get("stop_patience", 3)),
    "--min-iterations", str(config.get("min_iterations", 0)),
    "--solver", config.get("solver", "outcome"),
    "--report-mode", config.get("report_mode", "summary"),
    "--artifact-mode", config.get("artifact_mode", "lightweight"),
    "--range-last-n", str(config.get("range_last_n", 2000)),
    "--output-json", str(config.get("output_json", "./overnight_runs/runner/report.json")),
]
checkpoint_every = config.get("checkpoint_every")
stability_checkpoint = config.get("stability_checkpoint")
if checkpoint_every not in (None, 0):
    if stability_checkpoint in (None, 0):
        stability_checkpoint = checkpoint_every
    elif int(stability_checkpoint) != int(checkpoint_every):
        stability_checkpoint = checkpoint_every
if stability_checkpoint not in (None, 0):
    args.extend(["--stability-checkpoint", str(stability_checkpoint)])
if config.get("checkpoint_history_limit") not in (None, 0):
    args.extend(["--checkpoint-history-limit", str(config["checkpoint_history_limit"])])
print(' '.join(__import__('shlex').quote(x) for x in args))
PY
)"

SOLVER_ENV_EXPORTS="$(python3 - "$CONFIG_PATH" <<'PY'
import json, shlex, sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding='utf-8'))
env_map = config.get('solver_env') or config.get('solver_overrides') or {}
if not isinstance(env_map, dict):
    env_map = {}
exports = []
for key, value in env_map.items():
    if value is None:
        continue
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(',', ':'))
    exports.append(f"{key}={shlex.quote(str(value))}")
print(' '.join(exports))
PY
)"

printf '==> resolved command for %s\n' "$DEPLOY_TARGET"
printf '%s\n' "$JSON_OUT"
if [[ -n "$SOLVER_ENV_EXPORTS" ]]; then
  SOLVER_ENV_COUNT="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print(len((config.get('solver_env') or config.get('solver_overrides') or {})))
PY
)"
  printf '==> exporting %s env values\n' "$SOLVER_ENV_COUNT"
fi

if [[ "$DEPLOY_TARGET" == "local" ]]; then
  echo "==> local deploy"
  cd "$ROOT_DIR"
  if [[ -n "$SOLVER_ENV_EXPORTS" ]]; then
    eval "export $SOLVER_ENV_EXPORTS; docker compose run --rm pokerkit-open-spiel $JSON_OUT"
  else
    eval "docker compose run --rm pokerkit-open-spiel $JSON_OUT"
  fi
  exit 0
fi

if [[ "$DEPLOY_TARGET" == "gce" ]]; then
  echo "==> gce deploy"

  PROJECT="${PROJECT:-$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get('project', 'pokerspiel'))
PY
)}"
  ZONE="${ZONE:-$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get('zone', 'us-west1-b'))
PY
)}"

  GCE_INSTANCES="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
instances = cfg.get('instances') or []
if cfg.get('instance'):
    instances = [cfg['instance']] + list(instances)
if not instances:
    instances = ['instance-20260818-234442']
print('\n'.join(instances))
PY
)"
  RANGE_SAMPLES_OVERRIDE="$(python3 - "$CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
print(cfg.get('range_samples', 1326))
PY
)"

  if [[ -n "${INSTANCE:-}" ]]; then
    GCE_INSTANCES="$INSTANCE"
  fi

  SOLVER_ENV_DOCKER_ARGS="$(python3 - "$CONFIG_PATH" <<'PY'
import json, shlex, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
env_map = config.get('solver_env') or config.get('solver_overrides') or {}
if not isinstance(env_map, dict):
    env_map = {}
args = []
for key, value in env_map.items():
    if value is None:
        continue
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(',', ':'))
    args.append(f"-e {key}={shlex.quote(str(value))}")
print(' '.join(args))
PY
)"

  deploy_failures=0
  while IFS= read -r target_instance; do
    [[ -z "$target_instance" ]] && continue
    SSH_COMMAND=$(cat <<EOF
set -eux
sudo apt-get update
sudo apt-get install -y git docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker "\$USER"

newgrp docker <<'REMOTE'
set -eux
cd "\$HOME"

if [ ! -d pokerspiel ]; then
  git clone https://github.com/lalligagger/pokerspiel pokerspiel
fi

cd pokerspiel
git pull --ff-only || true

docker build -t pokerspiel-live .

docker ps -aq --filter "publish=8080" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker rm -f pokerspiel-run >/dev/null 2>&1 || true

docker run -d \
  --name pokerspiel-run \
  -p 8080:8080 \
  $SOLVER_ENV_DOCKER_ARGS \
  -e "POKERSPIEL_RANGE_SAMPLES=$RANGE_SAMPLES_OVERRIDE" \
  -v "\$HOME/pokerspiel:/app" \
  -w /app \
  pokerspiel-live \
  uvicorn api.app:app --host 0.0.0.0 --port 8080

echo "Started detached FastAPI container: pokerspiel-run"
echo "Monitor with: docker logs -f pokerspiel-run"
REMOTE
EOF
)

    echo "==> deploying to gce instance: $target_instance"
    EXTERNAL_IP="$(gcloud compute instances describe "$target_instance" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || true)"

    if gcloud compute ssh "$target_instance" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --command="$SSH_COMMAND"; then
      echo "==> success on gce instance: $target_instance"
      echo "==> local status: http://localhost:8080/status"
      if [[ -n "$EXTERNAL_IP" ]]; then
        echo "==> external status: http://$EXTERNAL_IP:8080/status"
      else
        echo "==> external status: unavailable (instance IP lookup failed)"
      fi
    else
      echo "==> failed on gce instance: $target_instance" >&2
      deploy_failures=$((deploy_failures + 1))
    fi
  done <<< "$GCE_INSTANCES"

  if [[ "$deploy_failures" -gt 0 ]]; then
    echo "==> gce deploy summary: $deploy_failures failed out of $(printf '%s\n' "$GCE_INSTANCES" | sed '/^$/d' | wc -l | tr -d ' ' )" >&2
    exit 1
  fi

  echo "==> gce deploy summary: all instances launched successfully"
  exit 0
fi

echo "No deployment target matched" >&2
exit 2

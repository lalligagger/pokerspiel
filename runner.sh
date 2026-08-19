#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/solve_config_light.json}"
DEPLOY_TARGET="${2:-local}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

python3 - <<'PY' "$CONFIG_PATH" "$DEPLOY_TARGET"
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

# Merge six config-driven flags used by the solver CLI.
args = [
    "python", "app_solver.py", config.get("mode", "hulh"),
    "--iterations", str(config.get("iterations", 100)),
    "--preset", config.get("preset", "hulh-preflop"),
    "--samples", str(config.get("range_samples", 1000)),
    "--stability-threshold", str(config.get("stability_threshold", 0.01)),
    "--stop-patience", str(config.get("stop_patience", 3)),
    "--min-iterations", str(config.get("min_iterations", 0)),
    "--solver", config.get("solver", "outcome"),
    "--report-mode", config.get("report_mode", "summary"),
    "--artifact-mode", config.get("artifact_mode", "lightweight"),
    "--range-last-n", str(config.get("range_last_n", 2000)),
    "--output-json", str(config.get("output_json", "./overnight_runs/runner/report.json")),
]

if config.get("checkpoint_every") not in (None, 0):
    args.extend(["--checkpoint-every", str(config["checkpoint_every"])])
if config.get("stability_checkpoint") not in (None, 0):
    args.extend(["--stability-checkpoint", str(config["stability_checkpoint"])])
if config.get("checkpoint_history_limit") not in (None, 0):
    args.extend(["--checkpoint-history-limit", str(config["checkpoint_history_limit"])])

# Convert to JSON-serializable value for shell usage.
print(json.dumps({"deploy": deploy_target, "config": config, "argv": args}, separators=(",", ":")))
PY

JSON_OUT="$(python3 - <<'PY' "$CONFIG_PATH" "$DEPLOY_TARGET"
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
    "--samples", str(config.get("range_samples", 1000)),
    "--stability-threshold", str(config.get("stability_threshold", 0.01)),
    "--stop-patience", str(config.get("stop_patience", 3)),
    "--min-iterations", str(config.get("min_iterations", 0)),
    "--solver", config.get("solver", "outcome"),
    "--report-mode", config.get("report_mode", "summary"),
    "--artifact-mode", config.get("artifact_mode", "lightweight"),
    "--range-last-n", str(config.get("range_last_n", 2000)),
    "--output-json", str(config.get("output_json", "./overnight_runs/runner/report.json")),
]
if config.get("checkpoint_every") not in (None, 0):
    args.extend(["--checkpoint-every", str(config["checkpoint_every"])])
if config.get("stability_checkpoint") not in (None, 0):
    args.extend(["--stability-checkpoint", str(config["stability_checkpoint"])])
if config.get("checkpoint_history_limit") not in (None, 0):
    args.extend(["--checkpoint-history-limit", str(config["checkpoint_history_limit"])])
print(' '.join(__import__('shlex').quote(x) for x in args))
PY
)"

if [[ "$DEPLOY_TARGET" == "local" ]]; then
  echo "==> local deploy"
  cd "$ROOT_DIR"
  eval "docker compose run --rm pokerkit-open-spiel $JSON_OUT"
  exit 0
fi

if [[ "$DEPLOY_TARGET" == "gce" ]]; then
  echo "==> gce deploy"
  PROJECT="${PROJECT:-pokerspiel}"
  ZONE="${ZONE:-us-west1-b}"
  INSTANCE="${INSTANCE:-instance-20260818-234442}"

  gcloud compute ssh "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --command="
      set -eux
      cd ~/pokerspiel || cd /home/\$USER/pokerspiel || true
      if [ ! -d ~/pokerspiel ]; then
        git clone https://github.com/lalligagger/pokerspiel ~/pokerspiel
      fi
      cd ~/pokerspiel
      git pull --ff-only || true
      docker build -t pokerspiel-live .
      docker run --rm \
        -v \$HOME/pokerspiel:/app \
        -w /app \
        pokerspiel-live \
        $JSON_OUT
    "
  exit 0
fi

echo "No deployment target matched" >&2
exit 2

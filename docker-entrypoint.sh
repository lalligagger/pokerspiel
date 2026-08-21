#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-/app/cfg/solve_config_debug.json}"

if [[ -f "$CONFIG_PATH" ]]; then
  eval "$(python3 - "$CONFIG_PATH" <<'PY'
import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding='utf-8'))

mapping = {
    'POKERSPIEL_SOLVER': config.get('solver'),
    'POKERSPIEL_PRESET': config.get('preset'),
    'POKERSPIEL_RANGE_SAMPLES': config.get('range_samples'),
    'POKERSPIEL_POSTFLOP_SAMPLES': config.get('postflop_samples'),
    'POKERSPIEL_STABILITY_THRESHOLD': config.get('stability_threshold'),
    'POKERSPIEL_STOP_THRESHOLD': config.get('stop_threshold', 0.85),
    'POKERSPIEL_STOP_PATIENCE': config.get('stop_patience'),
    'POKERSPIEL_MIN_ITERATIONS': config.get('min_iterations'),
    'POKERSPIEL_CHECKPOINT_EVERY': config.get('checkpoint_every') if config.get('checkpoint_every') is not None else config.get('stability_checkpoint'),
    'POKERSPIEL_ITERATIONS': config.get('iterations'),
    'POKERSPIEL_MAX_ITERATIONS': config.get('iterations'),
    'POKERSPIEL_MEMORY_THRESHOLD': config.get('memory_threshold'),
    'POKERSPIEL_OUTPUT_JSON': config.get('output_json'),
}

for key, value in mapping.items():
    if value is None:
        continue
    if isinstance(value, (dict, list)):
        value = json.dumps(value, separators=(',', ':'))
    print(f'export {key}={shlex.quote(str(value))}')
PY
)"
fi

exec uvicorn api.app:app --host 0.0.0.0 --port 8080

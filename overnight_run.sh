#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/app/overnight_runs}"
N_ITERATIONS="${2:-5000}"
CHECKPOINT_EVERY="${3:-500}"

mkdir -p "$OUT_DIR"
run_dir="$OUT_DIR/baseline"
mkdir -p "$run_dir"

echo "=== baseline run ==="
docker compose run --rm pokerkit-open-spiel \
  python app_solver.py \
    hulh \
    --iterations "$N_ITERATIONS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --preset hulh-preflop \
    --samples 1000 \
    --stability-threshold 0.01 \
    --stop-patience 3 \
    --min-iterations 500000 \
    --solver outcome \
    --report-mode all \
    --output-json "$run_dir/report.json"

echo "saved report: $run_dir/report.json"
printf 'baseline run complete\n'

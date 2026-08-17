#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT}/environment.yml"
ENV_NAME="${ENV_NAME:-pokerkit_test}"

CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -f "/Users/lalligagger/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="/Users/lalligagger/miniconda3"
elif [ -f "/Users/lalligagger/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="/Users/lalligagger/anaconda3"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="/opt/miniconda3"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="/opt/anaconda3"
else
  echo "conda was not found in PATH and no known Miniconda/Anaconda install path was detected."
  echo "If your install lives elsewhere, edit this script to point at your conda base directory."
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Environment '$ENV_NAME' exists; updating it..."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "Creating environment '$ENV_NAME' from $ENV_FILE ..."
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

conda activate "$ENV_NAME"
python - <<'PY'
import importlib
mods = ['open_spiel', 'pokerkit']
for name in mods:
    try:
        importlib.import_module(name)
        print(f'{name}: OK')
    except Exception as exc:
        print(f'{name}: FAIL -> {exc}')
        raise
PY

echo
echo "Environment is ready. Use:"
echo "  conda activate $ENV_NAME"
echo "  cd $ROOT"
echo "  export PYTHONPATH=."

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/run_YCB_liberhand.json"
CPU_SET="0-15"
RUN_J=16
RENDER_J=2
FORCE=1
DROP_CACHES=1
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_pipeline.sh [-c CONFIG] [--cpu-set CPU_SET] [--run-j N] [--render-j N]
                          [--no-force] [--no-drop-caches] [--python PYTHON]

Description:
  Run the full dataset pipeline for one config:
  1) optionally drop Linux page caches
  2) run CPU grasp sampling via run_multi.py
  3) render partial point clouds via run_warp_render.py
  4) build train/test split manifests via build_dataset_splits.py
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"
      shift 2
      ;;
    --cpu-set)
      CPU_SET="$2"
      shift 2
      ;;
    --run-j)
      RUN_J="$2"
      shift 2
      ;;
    --render-j)
      RENDER_J="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-force)
      FORCE=0
      shift
      ;;
    --no-drop-caches)
      DROP_CACHES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

echo "[run_pipeline] root=$ROOT_DIR"
echo "[run_pipeline] config=$CONFIG"
echo "[run_pipeline] cpu_set=$CPU_SET run_j=$RUN_J render_j=$RENDER_J force=$FORCE drop_caches=$DROP_CACHES"

if [[ "$DROP_CACHES" -eq 1 ]]; then
  echo "[run_pipeline] dropping Linux page caches..."
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
fi

RUN_MULTI_CMD=(
  taskset -c "$CPU_SET"
  env
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  "$PYTHON_BIN" run_multi.py
  -c "$CONFIG"
  -j "$RUN_J"
)
if [[ "$FORCE" -eq 1 ]]; then
  RUN_MULTI_CMD+=(--force)
fi

echo "[run_pipeline] step 1/3: run_multi.py"
"${RUN_MULTI_CMD[@]}"

echo "[run_pipeline] step 2/3: run_warp_render.py"
"$PYTHON_BIN" run_warp_render.py -c "$CONFIG" -j "$RENDER_J"

echo "[run_pipeline] step 3/3: build_dataset_splits.py"
"$PYTHON_BIN" build_dataset_splits.py -c "$CONFIG"

echo "[run_pipeline] done"

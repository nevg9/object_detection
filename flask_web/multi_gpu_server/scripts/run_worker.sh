#!/usr/bin/env bash
set -euo pipefail

gpu_index="${1:?usage: run_worker.sh GPU_INDEX}"
case "${gpu_index}" in
  0|1|2|3|4|5|6|7) ;;
  *) echo "GPU_INDEX must be between 0 and 7" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${OD_REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
venv="${OD_VENV:-/opt/object-detection/venv}"
worker_base_port="${OD_WORKER_BASE_PORT:-5100}"
worker_port="$((worker_base_port + gpu_index))"
log_dir="${OD_LOG_DIR:-/mnt/data2/object_detection/logs}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export OD_GPU_INDEX="${gpu_index}"
export OD_WORKER_PORT="${worker_port}"
export OD_LOG_FILE="${log_dir}/worker-${gpu_index}.log"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "${repo_root}"
exec "${venv}/bin/gunicorn" \
  --bind "127.0.0.1:${worker_port}" \
  --workers 1 \
  --worker-class sync \
  --threads 1 \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 2 \
  --access-logfile "${log_dir}/worker-${gpu_index}-access.log" \
  --error-logfile "${log_dir}/worker-${gpu_index}-error.log" \
  "flask_web.multi_gpu_server.worker:app"

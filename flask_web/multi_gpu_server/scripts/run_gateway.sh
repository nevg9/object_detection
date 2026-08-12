#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${OD_REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
venv="${OD_VENV:-/opt/object-detection/venv}"
gateway_port="${OD_GATEWAY_PORT:-5090}"
log_dir="${OD_LOG_DIR:-/mnt/data2/object_detection/logs}"

export OD_LOG_FILE="${log_dir}/gateway.log"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

cd "${repo_root}"
exec "${venv}/bin/gunicorn" \
  --bind "127.0.0.1:${gateway_port}" \
  --workers 1 \
  --worker-class gthread \
  --threads 80 \
  --timeout 1900 \
  --graceful-timeout 30 \
  --keep-alive 5 \
  --access-logfile "${log_dir}/gateway-access.log" \
  --error-logfile "${log_dir}/gateway-error.log" \
  "flask_web.multi_gpu_server.gateway:app"

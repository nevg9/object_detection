#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${OD_REPO_ROOT:-$(cd "${script_dir}/../../.." && pwd)}"
venv="${OD_VENV:-/opt/object-detection/venv}"

cd "${repo_root}"
exec "${venv}/bin/python" -m flask_web.multi_gpu_server.cleanup

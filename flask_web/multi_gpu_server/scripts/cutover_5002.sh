#!/usr/bin/env bash
set -euo pipefail

readonly old_port=5002
readonly staging_port=15002
readonly gateway_port=5090
readonly rule_comment="object-detection-multi-gpu-cutover"
readonly nginx_config="${OD_NGINX_CONFIG:-/etc/nginx/conf.d/object-detection.conf}"
readonly service_user="${OD_SERVICE_USER:-yuzhong}"

readonly -a worker_units=(
  object-detection-worker@0.service
  object-detection-worker@1.service
  object-detection-worker@2.service
  object-detection-worker@3.service
  object-detection-worker@4.service
  object-detection-worker@5.service
  object-detection-worker@6.service
  object-detection-worker@7.service
)

readonly -a prerouting_rule=(
  -p tcp
  -m addrtype --dst-type LOCAL
  --dport "${old_port}"
  -m comment --comment "${rule_comment}"
  -j REDIRECT --to-ports "${staging_port}"
)

readonly -a output_rule=(
  -p tcp
  -m addrtype --dst-type LOCAL
  --dport "${old_port}"
  -m comment --comment "${rule_comment}"
  -j REDIRECT --to-ports "${staging_port}"
)

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || die "run this script with sudo"
}

require_commands() {
  local command_name
  for command_name in curl grep iptables nginx pgrep sed ss systemctl; do
    command -v "${command_name}" >/dev/null || die "missing command: ${command_name}"
  done
}

new_service_ready() {
  local port="$1"
  local payload
  payload="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/health/ready")" || return 1
  grep -q '"online_gpus":\[0,1,2,3,4,5,6,7\]' <<<"${payload}" &&
    grep -q '"status":"ready"' <<<"${payload}"
}

assert_systemd_stack_ready() {
  local unit
  systemctl is-active --quiet nginx || die "nginx is not active"
  systemctl is-active --quiet object-detection-gateway.service ||
    die "gateway systemd service is not active"
  for unit in "${worker_units[@]}"; do
    systemctl is-active --quiet "${unit}" || die "${unit} is not active"
  done
  new_service_ready "${gateway_port}" || die "gateway ${gateway_port} is not ready"
  new_service_ready "${staging_port}" || die "nginx staging port ${staging_port} is not ready"
}

old_server_pids() {
  pgrep -u "${service_user}" -f \
    '(^|[[:space:]])server_241124\.py([[:space:]]|$)' || true
}

old_server_pid() {
  local -a pids=()
  mapfile -t pids < <(old_server_pids)
  if [[ "${#pids[@]}" -gt 1 ]]; then
    die "multiple legacy server processes found: ${pids[*]}"
  fi
  if [[ "${#pids[@]}" -eq 1 ]]; then
    printf '%s\n' "${pids[0]}"
  fi
  return 0
}

old_connection_count() {
  ss -Htn state established "( sport = :${old_port} )" | wc -l | tr -d ' '
}

port_is_listening() {
  local port="$1"
  ss -Hltn "( sport = :${port} )" | grep -q .
}

redirect_exists() {
  iptables -t nat -C PREROUTING "${prerouting_rule[@]}" >/dev/null 2>&1 &&
    iptables -t nat -C OUTPUT "${output_rule[@]}" >/dev/null 2>&1
}

add_redirect() {
  local added_prerouting=0
  if ! iptables -t nat -C PREROUTING "${prerouting_rule[@]}" >/dev/null 2>&1; then
    iptables -t nat -I PREROUTING 1 "${prerouting_rule[@]}"
    added_prerouting=1
  fi
  if ! iptables -t nat -C OUTPUT "${output_rule[@]}" >/dev/null 2>&1; then
    if ! iptables -t nat -I OUTPUT 1 "${output_rule[@]}"; then
      if [[ "${added_prerouting}" -eq 1 ]]; then
        iptables -t nat -D PREROUTING "${prerouting_rule[@]}" || true
      fi
      die "failed to install OUTPUT redirect"
    fi
  fi
}

remove_redirect() {
  while iptables -t nat -C OUTPUT "${output_rule[@]}" >/dev/null 2>&1; do
    iptables -t nat -D OUTPUT "${output_rule[@]}"
  done
  while iptables -t nat -C PREROUTING "${prerouting_rule[@]}" >/dev/null 2>&1; do
    iptables -t nat -D PREROUTING "${prerouting_rule[@]}"
  done
}

print_status() {
  local legacy_pid=""
  legacy_pid="$(old_server_pid)"
  log "legacy_pid=${legacy_pid:-none} legacy_connections=$(old_connection_count)"
  log "gateway_ready=$(new_service_ready "${gateway_port}" && echo yes || echo no)"
  log "staging_ready=$(new_service_ready "${staging_port}" && echo yes || echo no)"
  log "redirect_installed=$(redirect_exists && echo yes || echo no)"
  log "listen_5002=$(port_is_listening "${old_port}" && echo yes || echo no)"
  log "listen_15002=$(port_is_listening "${staging_port}" && echo yes || echo no)"
}

prepare_cutover() {
  local legacy_pid
  assert_systemd_stack_ready
  legacy_pid="$(old_server_pid)"
  [[ -n "${legacy_pid}" ]] || die "legacy server process was not found"
  port_is_listening "${old_port}" || die "legacy port ${old_port} is not listening"

  log "installing temporary redirects for new ${old_port} connections"
  add_redirect
  if ! new_service_ready "${old_port}"; then
    remove_redirect
    die "redirect verification failed; temporary rules were rolled back"
  fi

  log "redirect verified; new ${old_port} connections now use the multi-GPU service"
  log "legacy PID ${legacy_pid} remains alive for existing connections"
  print_status
}

wait_for_drain() {
  local timeout_seconds="$1"
  local deadline=$((SECONDS + timeout_seconds))
  local connections
  while true; do
    connections="$(old_connection_count)"
    [[ "${connections}" -eq 0 ]] && return 0
    if [[ "${SECONDS}" -ge "${deadline}" ]]; then
      return 1
    fi
    log "waiting for ${connections} legacy connections to drain"
    sleep 5
  done
}

ensure_nginx_listens_5002() {
  local backup
  if grep -Eq '^[[:space:]]*listen[[:space:]]+5002;' "${nginx_config}"; then
    log "nginx config already contains listen 5002"
  else
    grep -Eq '^[[:space:]]*listen[[:space:]]+15002;' "${nginx_config}" ||
      die "listen 15002 was not found in ${nginx_config}"
    backup="$(mktemp "${nginx_config}.cutover.XXXXXX")"
    cp -a "${nginx_config}" "${backup}"
    sed -i '/^[[:space:]]*listen[[:space:]]\+15002;/a\    listen 5002;' "${nginx_config}"
    if ! nginx -t; then
      cp -a "${backup}" "${nginx_config}"
      die "nginx config test failed; original config restored"
    fi
    rm -f "${backup}"
  fi
  systemctl reload nginx
  for _ in $(seq 1 20); do
    port_is_listening "${old_port}" && return 0
    sleep 1
  done
  die "nginx did not start listening on ${old_port}; redirect remains active"
}

finalize_cutover() {
  local timeout_seconds="${1:-1800}"
  local force="${2:-}"
  local legacy_pid=""
  [[ "${timeout_seconds}" =~ ^[0-9]+$ ]] || die "drain timeout must be an integer"
  [[ -z "${force}" || "${force}" == "--force" ]] || die "unknown finalize option: ${force}"

  assert_systemd_stack_ready
  redirect_exists || die "temporary redirect is not installed; run prepare first"
  new_service_ready "${old_port}" || die "port ${old_port} is not reaching the new service"

  if ! wait_for_drain "${timeout_seconds}"; then
    if [[ "${force}" != "--force" ]]; then
      die "legacy connections remain; old service was not stopped. Re-run with a longer timeout or append --force after confirming they are idle"
    fi
    log "WARNING: forcing finalization with $(old_connection_count) legacy connections"
  fi

  legacy_pid="$(old_server_pid)"
  if [[ -n "${legacy_pid}" ]]; then
    log "sending TERM to drained legacy PID ${legacy_pid}"
    kill -TERM "${legacy_pid}"
    for _ in $(seq 1 30); do
      kill -0 "${legacy_pid}" >/dev/null 2>&1 || break
      sleep 1
    done
    if kill -0 "${legacy_pid}" >/dev/null 2>&1; then
      die "legacy process did not exit after 30 seconds; redirect remains active"
    fi
  fi

  port_is_listening "${old_port}" &&
    die "port ${old_port} is unexpectedly still occupied; redirect remains active"

  log "adding nginx listen ${old_port} while redirect continues to protect traffic"
  ensure_nginx_listens_5002
  log "removing temporary redirects"
  remove_redirect
  if ! new_service_ready "${old_port}"; then
    add_redirect
    die "direct nginx verification failed; temporary redirect was restored"
  fi

  log "cutover complete: nginx directly owns ${old_port}"
  print_status
}

rollback_cutover() {
  local legacy_pid=""
  legacy_pid="$(old_server_pid)"
  if [[ -z "${legacy_pid}" ]] && ! grep -Eq \
    '^[[:space:]]*listen[[:space:]]+5002;' "${nginx_config}"; then
    die "neither the legacy service nor nginx can receive direct ${old_port} traffic; refusing rollback"
  fi
  remove_redirect
  log "temporary redirects removed"
  print_status
}

usage() {
  cat <<'EOF'
Usage:
  sudo cutover_5002.sh status
  sudo cutover_5002.sh prepare
  sudo cutover_5002.sh finalize [drain-timeout-seconds] [--force]
  sudo cutover_5002.sh rollback

prepare sends only new 5002 connections to Nginx 15002 and keeps the old
process alive. finalize drains old connections, stops the old process, makes
Nginx listen on both ports, and removes the temporary redirects.
EOF
}

main() {
  require_root
  require_commands
  case "${1:-status}" in
    status) print_status ;;
    prepare) prepare_cutover ;;
    finalize) finalize_cutover "${2:-1800}" "${3:-}" ;;
    rollback) rollback_cutover ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"

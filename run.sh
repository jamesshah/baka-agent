#!/usr/bin/env bash
# Run llama-server + baka-agent in the background (macOS).
#
# Usage:
#   ./run.sh start [llama|agent|all]
#   ./run.sh stop [llama|agent|all]
#   ./run.sh restart [llama|agent|all]
#   ./run.sh status
#   ./run.sh logs [llama|agent]
#
# Optional env overrides (or set in .env — only LLAMA_* / ports below are read here):
#   LLAMA_HF_REPO   Hugging Face model for llama-server -hf (default below)
#   LLAMA_PORT      llama-server port (default 8080)
#   AGENT_HOST      uvicorn host (default 0.0.0.0)
#   AGENT_PORT      uvicorn port (default 8000)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${ROOT}/.run"
LOG_DIR="${RUN_DIR}/logs"
LLAMA_PID_FILE="${RUN_DIR}/llama-server.pid"
AGENT_PID_FILE="${RUN_DIR}/agent.pid"
LLAMA_LOG="${LOG_DIR}/llama-server.log"
AGENT_LOG="${LOG_DIR}/agent.log"

# Defaults — override via environment.
LLAMA_HF_REPO="${LLAMA_HF_REPO:-unsloth/gemma-4-E2B-it-GGUF:BF16}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
AGENT_HOST="${AGENT_HOST:-0.0.0.0}"
AGENT_PORT="${AGENT_PORT:-8000}"
LLAMA_HEALTH_URL="http://127.0.0.1:${LLAMA_PORT}/health"
AGENT_HEALTH_URL="http://127.0.0.1:${AGENT_PORT}/health"

mkdir -p "${LOG_DIR}"

is_running() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  rm -f "${pid_file}"
  return 1
}

wait_for_url() {
  local url="$1"
  local name="$2"
  local attempts="${3:-60}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      echo "  ${name} is ready (${url})"
      return 0
    fi
    sleep 1
  done
  echo "  timed out waiting for ${name} at ${url}" >&2
  return 1
}

start_llama() {
  if is_running "${LLAMA_PID_FILE}"; then
    echo "llama-server already running (pid $(cat "${LLAMA_PID_FILE}"))"
    return 0
  fi

  if ! command -v llama-server >/dev/null 2>&1; then
    echo "llama-server not found. Install with: brew install llama.cpp" >&2
    exit 1
  fi

  echo "Starting llama-server (-hf ${LLAMA_HF_REPO} --port ${LLAMA_PORT})…"
  # --jinja enables chat templates / tool calling.
  nohup llama-server \
    -hf "${LLAMA_HF_REPO}" \
    --port "${LLAMA_PORT}" \
    --ctx-size 100000 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --jinja \
    >>"${LLAMA_LOG}" 2>&1 &
  echo $! >"${LLAMA_PID_FILE}"
  echo "  pid $(cat "${LLAMA_PID_FILE}")  log ${LLAMA_LOG}"
  wait_for_url "${LLAMA_HEALTH_URL}" "llama-server" 120
}

start_agent() {
  if is_running "${AGENT_PID_FILE}"; then
    echo "agent already running (pid $(cat "${AGENT_PID_FILE}"))"
    return 0
  fi

  local uvicorn_bin="${ROOT}/.venv/bin/uvicorn"
  if [[ ! -x "${uvicorn_bin}" ]]; then
    echo "Missing ${uvicorn_bin}. Create the venv and install deps first:" >&2
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
  fi

  if [[ ! -f "${ROOT}/.env" ]]; then
    echo "Warning: ${ROOT}/.env not found — copy .env.example and fill in credentials." >&2
  fi

  echo "Starting agent (uvicorn app:app --host ${AGENT_HOST} --port ${AGENT_PORT})…"
  (
    cd "${ROOT}"
    # Don't `source .env` — values with spaces/special chars can break bash.
    # pydantic-settings loads .env from the working directory.
    nohup "${uvicorn_bin}" app:app --host "${AGENT_HOST}" --port "${AGENT_PORT}" \
      >>"${AGENT_LOG}" 2>&1 &
    echo $! >"${AGENT_PID_FILE}"
  )
  echo "  pid $(cat "${AGENT_PID_FILE}")  log ${AGENT_LOG}"
  wait_for_url "${AGENT_HEALTH_URL}" "agent" 30
}

stop_pid_file() {
  local name="$1"
  local pid_file="$2"
  if ! is_running "${pid_file}"; then
    echo "${name} is not running"
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}")"
  echo "Stopping ${name} (pid ${pid})…"
  kill "${pid}" 2>/dev/null || true
  local i
  for ((i = 1; i <= 20; i++)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.25
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "  force killing ${name}…"
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
}

cmd_start() {
  local target="${1:-all}"
  case "${target}" in
    all|both|"")
      start_llama
      start_agent
      echo
      echo "Both services are up."
      ;;
    llama|llama-server)
      start_llama
      ;;
    agent|app)
      start_agent
      ;;
    *)
      echo "Unknown service: ${target} (use llama, agent, or all)" >&2
      exit 1
      ;;
  esac
  echo "  Agent health: ${AGENT_HEALTH_URL}"
  echo "  Llama health: ${LLAMA_HEALTH_URL}"
  echo "  Logs:         ./run.sh logs"
}

cmd_stop() {
  local target="${1:-all}"
  case "${target}" in
    all|both|"")
      stop_pid_file "agent" "${AGENT_PID_FILE}"
      stop_pid_file "llama-server" "${LLAMA_PID_FILE}"
      ;;
    llama|llama-server)
      stop_pid_file "llama-server" "${LLAMA_PID_FILE}"
      ;;
    agent|app)
      stop_pid_file "agent" "${AGENT_PID_FILE}"
      ;;
    *)
      echo "Unknown service: ${target} (use llama, agent, or all)" >&2
      exit 1
      ;;
  esac
}

cmd_restart() {
  local target="${1:-all}"
  cmd_stop "${target}"
  cmd_start "${target}"
}

cmd_status() {
  if is_running "${LLAMA_PID_FILE}"; then
    echo "llama-server: running (pid $(cat "${LLAMA_PID_FILE}"))"
  else
    echo "llama-server: stopped"
  fi
  if curl -sf "${LLAMA_HEALTH_URL}" >/dev/null 2>&1; then
    echo "  health: ok (${LLAMA_HEALTH_URL})"
  else
    echo "  health: unreachable (${LLAMA_HEALTH_URL})"
  fi

  if is_running "${AGENT_PID_FILE}"; then
    echo "agent:        running (pid $(cat "${AGENT_PID_FILE}"))"
  else
    echo "agent:        stopped"
  fi
  if curl -sf "${AGENT_HEALTH_URL}" >/dev/null 2>&1; then
    echo "  health: ok (${AGENT_HEALTH_URL})"
  else
    echo "  health: unreachable (${AGENT_HEALTH_URL})"
  fi
}

cmd_logs() {
  local which="${1:-both}"
  case "${which}" in
    llama|llama-server)
      touch "${LLAMA_LOG}"
      tail -n 50 -f "${LLAMA_LOG}"
      ;;
    agent|app)
      touch "${AGENT_LOG}"
      tail -n 50 -f "${AGENT_LOG}"
      ;;
    both|all|*)
      touch "${LLAMA_LOG}" "${AGENT_LOG}"
      echo "=== tailing ${LLAMA_LOG} + ${AGENT_LOG} (Ctrl-C to stop) ==="
      tail -n 20 -f "${LLAMA_LOG}" "${AGENT_LOG}"
      ;;
  esac
}

usage() {
  cat <<'EOF'
Run llama-server + baka-agent in the background (macOS).

Usage:
  ./run.sh start [llama|agent|all]     # default: all
  ./run.sh stop [llama|agent|all]
  ./run.sh restart [llama|agent|all]
  ./run.sh status
  ./run.sh logs [llama|agent]

Examples:
  ./run.sh restart agent    # only re-run the agent
  ./run.sh restart llama    # only re-run llama-server
  ./run.sh start            # start both

Optional env overrides:
  LLAMA_HF_REPO   Hugging Face model for llama-server -hf
  LLAMA_PORT      llama-server port (default 8080)
  AGENT_HOST      uvicorn host (default 0.0.0.0)
  AGENT_PORT      uvicorn port (default 8000)
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    start) shift || true; cmd_start "${1:-all}" ;;
    stop) shift || true; cmd_stop "${1:-all}" ;;
    restart) shift || true; cmd_restart "${1:-all}" ;;
    status) cmd_status ;;
    logs) shift || true; cmd_logs "${1:-both}" ;;
    -h|--help|help|"") usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"

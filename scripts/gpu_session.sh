#!/usr/bin/env bash
# GPU time-share protocol for the shared 24 GB laptop GPU (see CLAUDE.md GPU RULE: at most ONE
# process may use the GPU at any time). Every GPU-using runner is expected to wrap its work as:
#
#   scripts/gpu_session.sh acquire <name>
#   scripts/gpu_session.sh assert-idle
#   ... GPU work (e.g. `with-llama-server up` ... queries ... `with-llama-server down`) ...
#   scripts/gpu_session.sh release <name>
#
# The lock is a single well-known file (not per-name), so only one owner can hold the GPU at a time.
# It is a cooperative (advisory) lock, not a kernel-enforced mutex: it stops well-behaved runners from
# racing each other, it does not stop a rogue process from touching the GPU directly.
set -euo pipefail

DATA="${SPEECHRL_DATA_DIR:-$HOME/speechrl-data}"
LOCKFILE="$DATA/_gpu.lock"

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMA_BIN="$LLAMACPP_DIR/build/bin/llama-server"
GGUF_DIR="$DATA/models/qwen3-omni-30b-a3b-instruct-gguf"
LLAMA_MODEL="${LLAMA_MODEL:-$GGUF_DIR/Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf}"
LLAMA_MMPROJ="${LLAMA_MMPROJ:-$GGUF_DIR/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8091}"
LLAMA_NGL="${LLAMA_NGL:-28}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
LLAMA_PIDFILE="$DATA/_llama_server.pid"
LLAMA_LOGFILE="$DATA/_llama_server.log"
LLAMA_UP_TIMEOUT="${LLAMA_UP_TIMEOUT:-600}"   # seconds to poll for readiness; measured cold load ~4 min

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  acquire <name>            Take the exclusive GPU lock as <name>. Fails loudly (exit 1) if the
                             lock is already held by a name whose pid is still alive.
  release <name>            Release the GPU lock. Refuses (exit 1) to release a lock held by a
                             different, still-alive owner.
  status                     Print current lock state (free / held / stale) and exit 0.
  assert-idle                Exit 0 iff \`nvidia-smi\` reports 0 compute processes; otherwise print
                             the offending process list on stderr and exit 1.
  with-llama-server up       Start the resident Qwen3-Omni-30B llama-server (GGUF via llama.cpp),
                             writing a pidfile. No-op (exit 0) if already running.
  with-llama-server down     Stop it via the pidfile: graceful SIGTERM, wait, SIGKILL if it does
                             not exit within 30s.

Env overrides:
  SPEECHRL_DATA_DIR          data root (default \$HOME/speechrl-data); lock/pidfile/log live under it.
  GPU_SESSION_OWNER_PID       pid recorded as the lock owner (default: \$PPID, i.e. the caller's shell).
  LLAMACPP_DIR                 llama.cpp checkout (default \$HOME/llama.cpp).
  LLAMA_MODEL / LLAMA_MMPROJ / LLAMA_HOST / LLAMA_PORT / LLAMA_NGL / LLAMA_CTX / LLAMA_UP_TIMEOUT
                                llama-server launch overrides (defaults match wiki/Inference-Engine-Choice.md).
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

# pid_alive PID -> true iff PID is a plausible positive integer AND that process exists.
pid_alive() {
  case "${1:-}" in
    ''|*[!0-9]*) return 1 ;;
  esac
  kill -0 "$1" 2>/dev/null
}

# lock_field KEY -> value of KEY=... in $LOCKFILE (empty if absent/no lock file).
lock_field() {
  [ -f "$LOCKFILE" ] || return 0
  sed -n "s/^$1=//p" "$LOCKFILE" | head -n1
}

cmd_acquire() {
  local name="${1:?Usage: gpu_session.sh acquire <name>}"
  local owner_pid="${GPU_SESSION_OWNER_PID:-$PPID}"
  mkdir -p "$DATA"

  if [ -e "$LOCKFILE" ]; then
    local h_owner h_pid h_ts
    h_owner="$(lock_field owner)"; h_pid="$(lock_field pid)"; h_ts="$(lock_field ts)"
    if pid_alive "$h_pid"; then
      echo "ERROR: GPU lock already held by '$h_owner' (pid $h_pid, since $h_ts)." >&2
      echo "       Refusing to acquire for '$name'. If you are certain the holder is gone," >&2
      echo "       verify with 'ps -p $h_pid' and remove $LOCKFILE by hand." >&2
      exit 1
    fi
    echo "WARN: stale GPU lock (owner='$h_owner' pid=$h_pid is dead) - reclaiming for '$name'." >&2
    rm -f "$LOCKFILE"
  fi

  # Atomic claim: `set -o noclobber` makes `: > file` fail (EEXIST) if a concurrent acquire won the
  # race between our check above and here. Only the winner proceeds to write real lock content.
  if ! ( set -o noclobber; : > "$LOCKFILE" ) 2>/dev/null; then
    die "lock file appeared concurrently (lost the race to acquire) - another process just took it for '$name'."
  fi
  {
    printf 'owner=%s\n' "$name"
    printf 'pid=%s\n' "$owner_pid"
    printf 'ts=%s\n' "$(date -u +%FT%TZ)"
    printf 'host=%s\n' "$(hostname)"
  } > "$LOCKFILE"
  echo "acquired GPU lock: owner=$name pid=$owner_pid ts=$(lock_field ts)"
}

cmd_release() {
  local name="${1:?Usage: gpu_session.sh release <name>}"
  if [ ! -f "$LOCKFILE" ]; then
    echo "no GPU lock held (nothing to release for '$name')"
    return 0
  fi
  local h_owner h_pid
  h_owner="$(lock_field owner)"; h_pid="$(lock_field pid)"
  if [ "$h_owner" != "$name" ] && pid_alive "$h_pid"; then
    echo "ERROR: GPU lock is held by '$h_owner' (pid $h_pid, alive), not '$name'." >&2
    echo "       Refusing to release someone else's live lock." >&2
    exit 1
  fi
  rm -f "$LOCKFILE"
  echo "released GPU lock (was owner=$h_owner pid=$h_pid)"
}

cmd_status() {
  if [ ! -f "$LOCKFILE" ]; then
    echo "GPU lock: FREE ($LOCKFILE does not exist)"
    return 0
  fi
  local h_owner h_pid h_ts h_host
  h_owner="$(lock_field owner)"; h_pid="$(lock_field pid)"; h_ts="$(lock_field ts)"; h_host="$(lock_field host)"
  if pid_alive "$h_pid"; then
    echo "GPU lock: HELD by owner='$h_owner' pid=$h_pid host=$h_host since=$h_ts (pid alive)"
  else
    echo "GPU lock: STALE owner='$h_owner' pid=$h_pid host=$h_host since=$h_ts (pid NOT alive; next acquire reclaims)"
  fi
}

cmd_assert_idle() {
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found on PATH"
  local procs
  procs="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [ -n "$procs" ]; then
    echo "ERROR: GPU not idle - compute processes present:" >&2
    echo "$procs" >&2
    exit 1
  fi
  echo "GPU idle: 0 compute processes"
}

llama_up() {
  if [ -f "$LLAMA_PIDFILE" ]; then
    local p; p="$(cat "$LLAMA_PIDFILE")"
    if pid_alive "$p"; then
      echo "llama-server already running (pid $p, pidfile $LLAMA_PIDFILE) - not starting a second one."
      return 0
    fi
    echo "WARN: stale llama-server pidfile (pid $p is dead) - cleaning up." >&2
    rm -f "$LLAMA_PIDFILE"
  fi
  [ -x "$LLAMA_BIN" ] || die "llama-server binary not found/executable: $LLAMA_BIN"
  [ -f "$LLAMA_MODEL" ] || die "model GGUF not found: $LLAMA_MODEL"
  [ -f "$LLAMA_MMPROJ" ] || die "mmproj GGUF not found: $LLAMA_MMPROJ"
  mkdir -p "$DATA"

  echo "starting resident llama-server: Qwen3-Omni-30B, -ngl $LLAMA_NGL -c $LLAMA_CTX, $LLAMA_HOST:$LLAMA_PORT ..."
  # Exact flags per scripts/repro_asr_best_of_n_llamacpp.py:16-20 / wiki/Inference-Engine-Choice.md.
  LD_LIBRARY_PATH="$LLAMACPP_DIR/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LLAMA_BIN" \
      -m "$LLAMA_MODEL" \
      --mmproj "$LLAMA_MMPROJ" \
      -ngl "$LLAMA_NGL" -c "$LLAMA_CTX" \
      --host "$LLAMA_HOST" --port "$LLAMA_PORT" --no-warmup \
      >"$LLAMA_LOGFILE" 2>&1 &
  local newpid=$!
  echo "$newpid" > "$LLAMA_PIDFILE"
  echo "launched llama-server pid=$newpid log=$LLAMA_LOGFILE pidfile=$LLAMA_PIDFILE"

  echo "polling http://$LLAMA_HOST:$LLAMA_PORT/health for readiness (timeout ${LLAMA_UP_TIMEOUT}s) ..."
  local waited=0
  while [ "$waited" -lt "$LLAMA_UP_TIMEOUT" ]; do
    if ! pid_alive "$newpid"; then
      rm -f "$LLAMA_PIDFILE"
      die "llama-server exited during startup - see $LLAMA_LOGFILE"
    fi
    if command -v curl >/dev/null 2>&1 && curl -sf "http://$LLAMA_HOST:$LLAMA_PORT/health" >/dev/null 2>&1; then
      echo "llama-server ready (pid=$newpid) after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "WARN: not confirmed ready after ${LLAMA_UP_TIMEOUT}s (pid=$newpid still running - may still be loading; check $LLAMA_LOGFILE)" >&2
}

llama_down() {
  if [ ! -f "$LLAMA_PIDFILE" ]; then
    echo "no llama-server pidfile ($LLAMA_PIDFILE) - nothing to stop"
    return 0
  fi
  local p; p="$(cat "$LLAMA_PIDFILE")"
  if ! pid_alive "$p"; then
    echo "WARN: pidfile pid $p not alive - removing stale pidfile." >&2
    rm -f "$LLAMA_PIDFILE"
    return 0
  fi
  echo "stopping llama-server pid=$p (SIGTERM) ..."
  kill -TERM "$p" 2>/dev/null || true
  local waited=0
  while pid_alive "$p" && [ "$waited" -lt 30 ]; do
    sleep 1
    waited=$((waited + 1))
  done
  if pid_alive "$p"; then
    echo "WARN: pid $p still alive after 30s SIGTERM wait - sending SIGKILL." >&2
    kill -KILL "$p" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$LLAMA_PIDFILE"
  echo "llama-server stopped"
}

cmd_with_llama_server() {
  case "${1:-}" in
    up) llama_up ;;
    down) llama_down ;;
    *) echo "Usage: gpu_session.sh with-llama-server <up|down>" >&2; exit 2 ;;
  esac
}

main() {
  local cmd="${1:-help}"
  if [ $# -gt 0 ]; then shift; fi
  case "$cmd" in
    acquire) cmd_acquire "$@" ;;
    release) cmd_release "$@" ;;
    status) cmd_status ;;
    assert-idle) cmd_assert_idle ;;
    with-llama-server) cmd_with_llama_server "$@" ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
  esac
}

main "$@"

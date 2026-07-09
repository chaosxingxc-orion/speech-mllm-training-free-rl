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
#
# POSTMORTEM (2026-07-09, 224-cell wave-1 run, zero cells executed): `acquire`'s pid default is
# `${GPU_SESSION_OWNER_PID:-$PPID}`. That is only correct when `acquire` is invoked DIRECTLY. If a
# caller instead runs it inside a command substitution -- `out="$(bash gpu_session.sh acquire NAME)"`
# (exactly what run_wave1.sh's gpu_acquire_polite did) -- bash forks a TRANSIENT subshell to host the
# substitution, and `acquire`'s own $PPID resolves to THAT subshell's pid, not the long-lived caller's.
# The transient subshell exits the instant `acquire` returns, so the very next liveness check (`pid
# alive`) reports the lock STALE seconds later, even though the real caller (e.g. run_wave1.sh) is
# still alive and mid-run. Verified empirically: `$$` (unlike `$PPID`) is NOT re-evaluated inside a
# subshell -- it stays pinned to the outermost/login shell's pid -- so any caller that may be wrapped
# in a subshell (command substitution, `( ... )`, a pipeline stage) MUST pass its own pid explicitly
# via `acquire NAME --pid "$$"` rather than rely on the $PPID default. See `check` below for the
# matching robust liveness query (used by run_baseline.py's assert_gpu_session_held instead of
# grepping `status` text).
set -euo pipefail

DATA="${SPEECHRL_DATA_DIR:-$HOME/speechrl-data}"
LOCKFILE="$DATA/_gpu.lock"

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMA_BIN="$LLAMACPP_DIR/build/bin/llama-server"
GGUF_DIR="$DATA/models/qwen3-omni-30b-a3b-instruct-gguf"
MERALION_DIR="$DATA/models/meralion-2-gguf"
LLAMA_UP_TIMEOUT="${LLAMA_UP_TIMEOUT:-600}"   # seconds to poll for readiness; measured cold load ~4 min

# ---- qwen3-omni-30b-gguf (legacy env-var names, kept for backward compat -- other scripts in
# this repo default LLAMA_SERVER to http://127.0.0.1:8091, i.e. THIS model's port) ----
LLAMA_MODEL="${LLAMA_MODEL:-$GGUF_DIR/Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf}"
LLAMA_MMPROJ="${LLAMA_MMPROJ:-$GGUF_DIR/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_PORT="${LLAMA_PORT:-8091}"
LLAMA_NGL="${LLAMA_NGL:-28}"
LLAMA_CTX="${LLAMA_CTX:-8192}"
LLAMA_PIDFILE="$DATA/_llama_server.pid"
LLAMA_LOGFILE="$DATA/_llama_server.log"

# ---- meralion-2-gguf (own port/pidfile/logfile so it can run alongside -- never simultaneously
# with, GPU RULE is still one-at-a-time -- the qwen3 config above) ----
MERALION_MODEL="${MERALION_MODEL:-$MERALION_DIR/meralion-3b-decoder-q8_0.gguf}"
MERALION_MMPROJ="${MERALION_MMPROJ:-$MERALION_DIR/meralion-3b-mmproj-f16.gguf}"
MERALION_HOST="${MERALION_HOST:-127.0.0.1}"
MERALION_PORT="${MERALION_PORT:-8197}"
MERALION_NGL="${MERALION_NGL:-99}"          # MERaLiON-2-3B is small enough to fit fully on-GPU
MERALION_CTX="${MERALION_CTX:-8192}"
MERALION_PIDFILE="$DATA/_llama_server_meralion.pid"
MERALION_LOGFILE="$DATA/_llama_server_meralion.log"

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  acquire <name> [--pid P]  Take the exclusive GPU lock as <name>. Fails loudly (exit 1) if the
                             lock is already held by a name whose pid is still alive. The pid
                             recorded as the lock owner's liveness token is, in priority order:
                             (1) --pid P if given, (2) \$GPU_SESSION_OWNER_PID if set, (3) \$PPID.
                             ALWAYS pass --pid "\$\$" explicitly if you might invoke this from
                             inside a subshell (command substitution, a pipeline stage, etc.) --
                             \$PPID there is the transient subshell, not your real long-lived pid
                             (see POSTMORTEM comment at the top of this file).
  release <name>            Release the GPU lock. Refuses (exit 1) to release a lock held by a
                             different, still-alive owner.
  status                     Print current lock state (free / held / stale) and exit 0.
  check <name>               Exit 0 iff the lock is HELD (owner's recorded pid alive), else exit 1
                             with a diagnostic on stderr. Does NOT require the owner to equal
                             <name> (advisory lock, any legitimate holder counts) -- only prints a
                             NOTE to stderr if the names differ. This is the robust primitive
                             run_baseline.py's assert_gpu_session_held uses instead of parsing
                             \`status\` output.
  assert-idle                Exit 0 iff \`nvidia-smi\` reports 0 compute processes; otherwise print
                             the offending process list on stderr and exit 1.
  with-llama-server up       Start the resident Qwen3-Omni-30B llama-server (GGUF via llama.cpp),
                             writing a pidfile. No-op (exit 0) if already running. Back-compat
                             alias for \`serve qwen3-omni-30b-gguf up\` (same model/port/pidfile).
  with-llama-server down     Stop it via the pidfile: graceful SIGTERM, wait, SIGKILL if it does
                             not exit within 30s. Alias for \`serve qwen3-omni-30b-gguf down\`.
  serve <model-key> up       Start the resident llama-server for <model-key> (own port + pidfile +
                             logfile per key, see MODEL KEYS below). No-op if already running.
                             Multiple DIFFERENT model-keys may have pidfiles at once (e.g. left
                             over from a previous session) but the GPU RULE (CLAUDE.md: at most ONE
                             process may use the GPU) means only one should be launched at a time
                             in practice -- this script does not enforce that beyond the gpu_session
                             lock itself, so always \`acquire\` first and \`down\` before switching.
  serve <model-key> down     Stop it via its pidfile (same SIGTERM/wait/SIGKILL semantics as above).

MODEL KEYS (for \`serve\`):
  qwen3-omni-30b-gguf   Qwen3-Omni-30B-A3B-Instruct Q8_0 GGUF, port 8091, -ngl 28 (see LLAMA_* below).
  meralion-2-gguf       MERaLiON-2-3B GGUF (decoder q8_0 + mmproj f16), port 8197, -ngl 99
                        (see MERALION_* below). Emits a "<Speaker1>: " turn-prefix in its replies --
                        callers must strip it before scoring (see run_baseline.py's gen_llamacpp).

Env overrides:
  SPEECHRL_DATA_DIR          data root (default \$HOME/speechrl-data); lock/pidfile/log live under it.
  GPU_SESSION_OWNER_PID       pid recorded as the lock owner (default: \$PPID, i.e. the caller's shell).
  LLAMACPP_DIR                 llama.cpp checkout (default \$HOME/llama.cpp), shared llama-server binary
                                for every model key.
  LLAMA_MODEL / LLAMA_MMPROJ / LLAMA_HOST / LLAMA_PORT / LLAMA_NGL / LLAMA_CTX / LLAMA_UP_TIMEOUT
                                qwen3-omni-30b-gguf launch overrides (defaults match
                                wiki/Inference-Engine-Choice.md).
  MERALION_MODEL / MERALION_MMPROJ / MERALION_HOST / MERALION_PORT / MERALION_NGL / MERALION_CTX
                                meralion-2-gguf launch overrides (LLAMA_UP_TIMEOUT is shared).
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
  local name="${1:?Usage: gpu_session.sh acquire <name> [--pid <pid>]}"
  shift
  local explicit_pid=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --pid) explicit_pid="${2:?--pid requires a value}"; shift 2 ;;
      *) die "cmd_acquire: unknown arg '$1' (usage: acquire <name> [--pid <pid>])" ;;
    esac
  done
  # Priority: explicit --pid > $GPU_SESSION_OWNER_PID > $PPID. See the POSTMORTEM comment at the
  # top of this file -- $PPID is WRONG whenever the caller runs `acquire` inside a subshell (e.g.
  # command substitution); such callers must pass --pid "$$" explicitly.
  local owner_pid="${explicit_pid:-${GPU_SESSION_OWNER_PID:-$PPID}}"
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

cmd_check() {
  local name="${1:?Usage: gpu_session.sh check <name>}"
  if [ ! -f "$LOCKFILE" ]; then
    echo "GPU lock: NOT HELD (no lock file) -- checked for '$name'" >&2
    return 1
  fi
  local h_owner h_pid h_ts
  h_owner="$(lock_field owner)"; h_pid="$(lock_field pid)"; h_ts="$(lock_field ts)"
  if ! pid_alive "$h_pid"; then
    echo "GPU lock: STALE (owner='$h_owner' pid=$h_pid dead since=$h_ts) -- checked for '$name'" >&2
    return 1
  fi
  if [ "$h_owner" != "$name" ]; then
    echo "NOTE: GPU lock held by a different owner ('$h_owner' pid=$h_pid, alive) than checked" >&2
    echo "      name '$name' -- still counts as HELD (advisory lock, identity is informational)." >&2
  fi
  echo "GPU lock: HELD by owner='$h_owner' pid=$h_pid since=$h_ts (checked for '$name')"
  return 0
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

# _model_cfg <model-key> -- sets the M_* globals (label/model/mmproj/host/port/ngl/ctx/pidfile/
# logfile/timeout) for a known model key. One place to add a new GGUF backbone (extend the case).
_model_cfg() {
  local key="${1:?_model_cfg: missing model-key}"
  case "$key" in
    qwen3-omni-30b-gguf)
      M_LABEL="Qwen3-Omni-30B"
      M_MODEL="$LLAMA_MODEL"; M_MMPROJ="$LLAMA_MMPROJ"
      M_HOST="$LLAMA_HOST"; M_PORT="$LLAMA_PORT"
      M_NGL="$LLAMA_NGL"; M_CTX="$LLAMA_CTX"
      M_PIDFILE="$LLAMA_PIDFILE"; M_LOGFILE="$LLAMA_LOGFILE"
      ;;
    meralion-2-gguf)
      M_LABEL="MERaLiON-2"
      M_MODEL="$MERALION_MODEL"; M_MMPROJ="$MERALION_MMPROJ"
      M_HOST="$MERALION_HOST"; M_PORT="$MERALION_PORT"
      M_NGL="$MERALION_NGL"; M_CTX="$MERALION_CTX"
      M_PIDFILE="$MERALION_PIDFILE"; M_LOGFILE="$MERALION_LOGFILE"
      ;;
    *) die "unknown model-key '$key' (known: qwen3-omni-30b-gguf, meralion-2-gguf)" ;;
  esac
  M_UP_TIMEOUT="$LLAMA_UP_TIMEOUT"
}

# serve_up <model-key> -- generic resident-llama-server launcher, parameterized via _model_cfg.
# Replaces the old qwen3-only llama_up() (identical polling/pidfile semantics, just not hardcoded
# to one model) so a second GGUF backbone (meralion-2-gguf) does not need a copy-pasted twin.
serve_up() {
  local key="${1:?Usage: gpu_session.sh serve <model-key> up}"
  _model_cfg "$key"
  if [ -f "$M_PIDFILE" ]; then
    local p; p="$(cat "$M_PIDFILE")"
    if pid_alive "$p"; then
      echo "llama-server ($M_LABEL) already running (pid $p, pidfile $M_PIDFILE) - not starting a second one."
      return 0
    fi
    echo "WARN: stale $M_LABEL pidfile (pid $p is dead) - cleaning up." >&2
    rm -f "$M_PIDFILE"
  fi
  [ -x "$LLAMA_BIN" ] || die "llama-server binary not found/executable: $LLAMA_BIN"
  [ -f "$M_MODEL" ] || die "model GGUF not found: $M_MODEL"
  [ -f "$M_MMPROJ" ] || die "mmproj GGUF not found: $M_MMPROJ"
  mkdir -p "$DATA"

  echo "starting resident llama-server: $M_LABEL, -ngl $M_NGL -c $M_CTX, $M_HOST:$M_PORT ..."
  # Exact flag shape per scripts/repro_asr_best_of_n_llamacpp.py:16-20 / wiki/Inference-Engine-Choice.md.
  LD_LIBRARY_PATH="$LLAMACPP_DIR/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$LLAMA_BIN" \
      -m "$M_MODEL" \
      --mmproj "$M_MMPROJ" \
      -ngl "$M_NGL" -c "$M_CTX" \
      --host "$M_HOST" --port "$M_PORT" --no-warmup \
      >"$M_LOGFILE" 2>&1 &
  local newpid=$!
  echo "$newpid" > "$M_PIDFILE"
  echo "launched llama-server ($M_LABEL) pid=$newpid log=$M_LOGFILE pidfile=$M_PIDFILE"

  echo "polling http://$M_HOST:$M_PORT/health for readiness (timeout ${M_UP_TIMEOUT}s) ..."
  local waited=0
  while [ "$waited" -lt "$M_UP_TIMEOUT" ]; do
    if ! pid_alive "$newpid"; then
      rm -f "$M_PIDFILE"
      die "llama-server ($M_LABEL) exited during startup - see $M_LOGFILE"
    fi
    if command -v curl >/dev/null 2>&1 && curl -sf "http://$M_HOST:$M_PORT/health" >/dev/null 2>&1; then
      echo "llama-server ($M_LABEL) ready (pid=$newpid) after ${waited}s"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "WARN: ($M_LABEL) not confirmed ready after ${M_UP_TIMEOUT}s (pid=$newpid still running - may still be loading; check $M_LOGFILE)" >&2
}

# serve_down <model-key> -- generic counterpart to serve_up (identical SIGTERM/wait/SIGKILL
# semantics as the old llama_down()).
serve_down() {
  local key="${1:?Usage: gpu_session.sh serve <model-key> down}"
  _model_cfg "$key"
  if [ ! -f "$M_PIDFILE" ]; then
    echo "no $M_LABEL pidfile ($M_PIDFILE) - nothing to stop"
    return 0
  fi
  local p; p="$(cat "$M_PIDFILE")"
  if ! pid_alive "$p"; then
    echo "WARN: pidfile pid $p not alive - removing stale pidfile." >&2
    rm -f "$M_PIDFILE"
    return 0
  fi
  echo "stopping llama-server ($M_LABEL) pid=$p (SIGTERM) ..."
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
  rm -f "$M_PIDFILE"
  echo "llama-server ($M_LABEL) stopped"
}

cmd_serve() {
  local key="${1:?Usage: gpu_session.sh serve <model-key> <up|down>}"
  local action="${2:?Usage: gpu_session.sh serve <model-key> <up|down>}"
  case "$action" in
    up) serve_up "$key" ;;
    down) serve_down "$key" ;;
    *) echo "Usage: gpu_session.sh serve <model-key> <up|down>" >&2; exit 2 ;;
  esac
}

# with-llama-server up/down -- back-compat alias for `serve qwen3-omni-30b-gguf <up|down>` (same
# model/port/pidfile as before this generalization; every existing caller keeps working unchanged).
cmd_with_llama_server() {
  case "${1:-}" in
    up) serve_up qwen3-omni-30b-gguf ;;
    down) serve_down qwen3-omni-30b-gguf ;;
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
    check) cmd_check "$@" ;;
    assert-idle) cmd_assert_idle ;;
    with-llama-server) cmd_with_llama_server "$@" ;;
    serve) cmd_serve "$@" ;;
    help|-h|--help) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage >&2; exit 2 ;;
  esac
}

main "$@"

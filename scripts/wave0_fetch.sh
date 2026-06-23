#!/usr/bin/env bash
# Wave 0 - fetch selected W1 assets (models, datasets, reference repos) to local.
#
# Manual run only. Downloads are large. By default this script prints help and
# downloads nothing.
#
# China mainland + VPN is the default mode. It prefers ModelScope for assets
# with verified replacements and uses HF/hf-mirror only as a fallback:
#   bash scripts/wave0_fetch.sh setup-env m_qwen3omni d_librispeech
#
# If your VPN is reliable and you prefer the official Hugging Face endpoint:
#   SPEECHRL_CN_MIRROR=0 SPEECHRL_MODEL_SOURCE=hf bash scripts/wave0_fetch.sh models
set -euo pipefail

# ---------- paths ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/../.." && pwd)"
DATA_ROOT="${SPEECHRL_DATA_DIR:-$WORKSPACE_ROOT/speechrl-data}"
MODELS="$DATA_ROOT/models"
DSETS="$DATA_ROOT/datasets"
REPOS="$DATA_ROOT/repos"
MANIFESTS="$DATA_ROOT/manifests"
mkdir -p "$MODELS" "$DSETS" "$REPOS" "$MANIFESTS"

# ---------- network/source config ----------
# SPEECHRL_CN_MIRROR=1:
# - uses hf-mirror.com unless SPEECHRL_HF_ENDPOINT or a non-official HF_ENDPOINT is set
# - uses ModelScope for exact domestic model mirrors where available
# Use this as the default because these assets are intended to be fetched from
# China mainland with a VPN, where official HF is often the least stable path.
SPEECHRL_CN_MIRROR="${SPEECHRL_CN_MIRROR:-1}"
SPEECHRL_MODEL_SOURCE="${SPEECHRL_MODEL_SOURCE:-auto}" # auto | hf | modelscope
SPEECHRL_DATASET_SOURCE="${SPEECHRL_DATASET_SOURCE:-auto}" # auto | hf | modelscope
SPEECHRL_HF_ENDPOINT="${SPEECHRL_HF_ENDPOINT:-}"
HF_CLI="${HF_CLI:-hf}"
CURL_TIMEOUT="${CURL_TIMEOUT:-20}"
GIT_CLONE_PREFIX="${GIT_CLONE_PREFIX:-}"
SPEECHRL_ALLOW_HF_DOWNLOADS="${SPEECHRL_ALLOW_HF_DOWNLOADS:-}"
SPEECHRL_SKIP_HF_DRY_RUN="${SPEECHRL_SKIP_HF_DRY_RUN:-}"
SPEECHRL_HF_MAX_RETRIES="${SPEECHRL_HF_MAX_RETRIES:-3}"
SPEECHRL_MS_MAX_RETRIES="${SPEECHRL_MS_MAX_RETRIES:-3}"
SPEECHRL_HFD_ENABLED="${SPEECHRL_HFD_ENABLED:-auto}"
SPEECHRL_HFD_THREADS="${SPEECHRL_HFD_THREADS:-8}"
SPEECHRL_MS_WORKERS="${SPEECHRL_MS_WORKERS:-16}"
SPEECHRL_HF_TRANSFER="${SPEECHRL_HF_TRANSFER:-auto}"
SPEECHRL_GH_MIRRORS="${SPEECHRL_GH_MIRRORS:-ghfast.top,ghproxy.cc}"
SPEECHRL_SKIP_EXISTING="${SPEECHRL_SKIP_EXISTING:-1}"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "$SPEECHRL_SKIP_HF_DRY_RUN" ]]; then
  if is_true "$SPEECHRL_CN_MIRROR"; then
    SPEECHRL_SKIP_HF_DRY_RUN=1
  else
    SPEECHRL_SKIP_HF_DRY_RUN=0
  fi
fi
if [[ -z "$SPEECHRL_ALLOW_HF_DOWNLOADS" ]]; then
  if is_true "$SPEECHRL_CN_MIRROR"; then
    SPEECHRL_ALLOW_HF_DOWNLOADS=0
  else
    SPEECHRL_ALLOW_HF_DOWNLOADS=1
  fi
fi

if [[ -n "$SPEECHRL_HF_ENDPOINT" ]]; then
  HF_ENDPOINT="$SPEECHRL_HF_ENDPOINT"
elif is_true "$SPEECHRL_CN_MIRROR"; then
  case "${HF_ENDPOINT:-}" in
    ""|https://huggingface.co|https://huggingface.co/|http://huggingface.co|http://huggingface.co/|https://www.huggingface.co|https://www.huggingface.co/|http://www.huggingface.co|http://www.huggingface.co/)
      HF_ENDPOINT="https://hf-mirror.com"
      ;;
  esac
elif [[ -z "${HF_ENDPOINT:-}" ]]; then
  HF_ENDPOINT="https://huggingface.co"
fi
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if is_true "$SPEECHRL_CN_MIRROR"; then
    HF_ENDPOINT="https://hf-mirror.com"
  else
    HF_ENDPOINT="https://huggingface.co"
  fi
fi
export HF_ENDPOINT
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
if is_true "$SPEECHRL_CN_MIRROR"; then
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
else
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
fi

log() { printf '[wave0] %s\n' "$*"; }
warn() { printf '[wave0] WARNING: %s\n' "$*" >&2; }
die() { printf '[wave0] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ "$SPEECHRL_HF_TRANSFER" == "1" ]] || { [[ "$SPEECHRL_HF_TRANSFER" == "auto" ]] && ! is_true "$SPEECHRL_CN_MIRROR"; }; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import hf_transfer" 2>/dev/null; then
    export HF_HUB_ENABLE_HF_TRANSFER=1
    log "hf_transfer enabled (Rust-based parallel download)"
  else
    if [[ "$SPEECHRL_HF_TRANSFER" == "1" ]]; then
      warn "hf_transfer requested but not installed. Install: pip install hf_transfer"
    fi
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  fi
else
  if is_true "$SPEECHRL_CN_MIRROR"; then
    log "hf_transfer disabled in CN mode (sensitive to packet loss on mirror); using hfd/aria2 instead"
  fi
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
fi

log_numbered_file() {
  local file="$1"
  local i=1
  while IFS= read -r line; do
    log "  [$i] $line"
    i=$((i + 1))
  done <"$file"
}

validate_model_source() {
  case "$SPEECHRL_MODEL_SOURCE" in
    auto|hf|modelscope) ;;
    *) die "SPEECHRL_MODEL_SOURCE must be auto, hf, or modelscope" ;;
  esac
  case "$SPEECHRL_DATASET_SOURCE" in
    auto|hf|modelscope) ;;
    *) die "SPEECHRL_DATASET_SOURCE must be auto, hf, or modelscope" ;;
  esac
}

prefer_modelscope() {
  case "$SPEECHRL_MODEL_SOURCE" in
    modelscope) return 0 ;;
    hf) return 1 ;;
    auto) is_true "$SPEECHRL_CN_MIRROR" ;;
  esac
}

prefer_modelscope_dataset() {
  case "$SPEECHRL_DATASET_SOURCE" in
    modelscope) return 0 ;;
    hf) return 1 ;;
    auto) is_true "$SPEECHRL_CN_MIRROR" ;;
  esac
}

hf_download_allowed() {
  local repo_kind="$1"

  if is_true "$SPEECHRL_ALLOW_HF_DOWNLOADS"; then
    return 0
  fi
  if [[ "$repo_kind" == "dataset" && "$SPEECHRL_DATASET_SOURCE" == "hf" ]]; then
    return 0
  fi
  if [[ "$repo_kind" == "model" && "$SPEECHRL_MODEL_SOURCE" == "hf" ]]; then
    return 0
  fi
  return 1
}

hf_repo_kind_from_args() {
  local arg next_is_repo_type=0
  for arg in "$@"; do
    if [[ "$next_is_repo_type" -eq 1 ]]; then
      if [[ "$arg" == "dataset" ]]; then
        printf 'dataset'
      else
        printf 'model'
      fi
      return
    fi
    if [[ "$arg" == "--repo-type" ]]; then
      next_is_repo_type=1
      continue
    fi
    case "$arg" in
      --repo-type=dataset)
        printf 'dataset'
        return
        ;;
    esac
  done
  printf 'model'
}

no_modelscope_model() {
  local name="$1"
  local reason="$2"

  if [[ "$SPEECHRL_MODEL_SOURCE" == "modelscope" ]] || ! hf_download_allowed model; then
    die "$name has no verified exact ModelScope replacement, and HF downloads are disabled in mainland mode. $reason"
  fi
  warn "$name has no verified exact ModelScope replacement. Falling back to HF/hf-mirror because HF downloads are explicitly allowed. $reason"
}

no_modelscope_dataset() {
  local name="$1"
  local reason="$2"

  if [[ "$SPEECHRL_DATASET_SOURCE" == "modelscope" ]] || ! hf_download_allowed dataset; then
    die "$name has no verified exact ModelScope replacement, and HF downloads are disabled in mainland mode. $reason"
  fi
  warn "$name has no verified exact ModelScope replacement. Falling back to HF/hf-mirror because HF downloads are explicitly allowed. $reason"
}

ensure_hf_cli() {
  command -v "$HF_CLI" >/dev/null 2>&1 || die "'$HF_CLI' CLI not found. Run setup-env first or install huggingface_hub[cli]."
}

dir_has_model_files() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  [[ -f "$dir/config.json" ]] || [[ -f "$dir/configuration.json" ]] || return 1
  compgen -G "$dir/*.safetensors" >/dev/null 2>&1 && return 0
  compgen -G "$dir/*.safetensors.index.json" >/dev/null 2>&1 && return 0
  compgen -G "$dir/*.bin" >/dev/null 2>&1 && return 0
  compgen -G "$dir/*.gguf" >/dev/null 2>&1 && return 0
  return 1
}

dir_has_dataset_files() {
  local dir="$1"
  [[ -d "$dir" ]] || return 1
  local file_count
  file_count="$(find "$dir" -maxdepth 2 -type f ! -name '.*' ! -name '*.md' 2>/dev/null | wc -l)"
  [[ "$file_count" -gt 3 ]]
}

skip_if_exists() {
  local dir="$1"
  local kind="$2"
  if ! is_true "$SPEECHRL_SKIP_EXISTING"; then
    return 1
  fi
  case "$kind" in
    model) dir_has_model_files "$dir" ;;
    dataset) dir_has_dataset_files "$dir" ;;
    *) [[ -d "$dir" ]] ;;
  esac
}

retry_cmd() {
  local max_retries="$1"
  shift
  local attempt=1
  while [[ "$attempt" -le "$max_retries" ]]; do
    log "attempt $attempt/$max_retries: $*"
    if "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$max_retries" ]]; then
      local wait_sec=$((attempt * 5))
      warn "attempt $attempt failed; retrying in ${wait_sec}s..."
      sleep "$wait_sec"
    fi
    attempt=$((attempt + 1))
  done
  die "all $max_retries attempts failed: $*"
}

ensure_hfd() {
  local hfd_bin="${SPEECHRL_HFD_BIN:-$HOME/.local/bin/hfd}"
  if [[ -x "$hfd_bin" ]]; then
    printf '%s' "$hfd_bin"
    return 0
  fi
  if ! is_true "$SPEECHRL_HFD_ENABLED" && [[ "$SPEECHRL_HFD_ENABLED" != "auto" ]]; then
    return 1
  fi
  if ! command -v aria2c >/dev/null 2>&1; then
    if [[ "$SPEECHRL_HFD_ENABLED" == "auto" ]]; then
      return 1
    fi
    die "hfd requires aria2c. Install: sudo apt install aria2"
  fi
  log "Installing hfd (hf-mirror downloader) to $hfd_bin"
  local tmp
  tmp="$(mktemp)"
  curl -L -sS -m "$CURL_TIMEOUT" "https://hf-mirror.com/hfd/hfd.sh" -o "$tmp" \
    || { rm -f "$tmp"; return 1; }
  chmod +x "$tmp"
  mkdir -p "$(dirname "$hfd_bin")"
  mv "$tmp" "$hfd_bin"
  printf '%s' "$hfd_bin"
}

hfd_download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  local repo_kind="model"
  local hfd_bin
  hfd_bin="$(ensure_hfd)" || return 1
  for arg in "$@"; do
    if [[ "$arg" == "dataset" ]]; then
      repo_kind="dataset"
    fi
  done
  local dataset_flag=""
  if [[ "$repo_kind" == "dataset" ]]; then
    dataset_flag="--dataset"
  fi
  log "hfd download: $repo_kind $repo_id -> $local_dir (aria2 -x $SPEECHRL_HFD_THREADS)"
  HF_ENDPOINT="$HF_ENDPOINT" retry_cmd "$SPEECHRL_HF_MAX_RETRIES" \
    "$hfd_bin" "$repo_id" --local-dir "$local_dir" $dataset_flag --tool aria2c -x "$SPEECHRL_HFD_THREADS"
}

hf_download() {
  local repo_id="$1"
  shift
  local repo_kind
  repo_kind="$(hf_repo_kind_from_args "$@")"
  if ! hf_download_allowed "$repo_kind"; then
    die "HF download is disabled in mainland mode for $repo_kind $repo_id. Use a ModelScope target, or explicitly set SPEECHRL_${repo_kind^^}_SOURCE=hf SPEECHRL_ALLOW_HF_DOWNLOADS=1 if you really want HF/hf-mirror."
  fi
  if is_true "$SPEECHRL_CN_MIRROR"; then
    local local_dir=""
    local arg prev_arg=""
    for arg in "$@"; do
      if [[ "$prev_arg" == "--local-dir" ]]; then local_dir="$arg"; break; fi
      prev_arg="$arg"
    done
    if [[ -n "$local_dir" ]] && hfd_bin="$(ensure_hfd)" 2>/dev/null; then
      log "CN mode: using hfd/aria2 for faster download ($SPEECHRL_HFD_THREADS threads)"
      hfd_download "$repo_id" "$local_dir" "$repo_kind"
      return
    fi
    warn "hfd not available, falling back to hf CLI (slower)"
  fi
  ensure_hf_cli
  log "HF_ENDPOINT=$HF_ENDPOINT"
  log "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
  log "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}"
  if ! is_true "$SPEECHRL_SKIP_HF_DRY_RUN"; then
    hf_dry_run "$repo_id" "$@"
  fi
  retry_cmd "$SPEECHRL_HF_MAX_RETRIES" "$HF_CLI" download "$repo_id" --resume-download "$@"
}

hf_dry_run() {
  local repo_id="$1"
  shift
  ensure_hf_cli
  local dry_run_output
  if ! dry_run_output="$("$HF_CLI" download "$repo_id" "$@" --dry-run 2>&1)"; then
    printf '%s\n' "$dry_run_output" >&2
    return 1
  fi
  printf '%s\n' "$dry_run_output"
  if grep -Eq 'Fetching 0 files|Will download 0 files \(out of 0\)' <<<"$dry_run_output"; then
    die "HF dry-run matched 0 files for $repo_id. Check --include patterns and repo layout."
  fi
}

ensure_modelscope_cli() {
  command -v modelscope >/dev/null 2>&1 || die "'modelscope' CLI not found. Run setup-env first or install modelscope."
}

ms_model_download() {
  local repo_id="$1"
  local local_dir="$2"

  if command -v modelscope >/dev/null 2>&1; then
    log "ModelScope download: model $repo_id (--max-workers $SPEECHRL_MS_WORKERS)"
    retry_cmd "$SPEECHRL_MS_MAX_RETRIES" modelscope download --max-workers "$SPEECHRL_MS_WORKERS" --model "$repo_id" --local_dir "$local_dir"
    return
  fi

  retry_cmd "$SPEECHRL_MS_MAX_RETRIES" python - "$repo_id" "$local_dir" <<'PY'
import sys
from modelscope import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id, local_dir=local_dir)
PY
}

ms_dataset_download() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2

  ensure_modelscope_cli
  log "ModelScope download: dataset $repo_id (--max-workers $SPEECHRL_MS_WORKERS)"
  retry_cmd "$SPEECHRL_MS_MAX_RETRIES" modelscope download --max-workers "$SPEECHRL_MS_WORKERS" --dataset "$repo_id" "$@" --local_dir "$local_dir"
}

git_url() {
  printf '%s%s' "$GIT_CLONE_PREFIX" "$1"
}

git_clone_once() {
  local url="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    log "exists: $target"
    return
  fi
  if [[ -n "$GIT_CLONE_PREFIX" ]]; then
    git clone "$(git_url "$url")" "$target"
    return
  fi
  local mirror
  local IFS=','
  for mirror in $SPEECHRL_GH_MIRRORS; do
    local prefixed="https://${mirror}/${url}"
    log "trying GitHub mirror: ${mirror}"
    if git clone "$prefixed" "$target" 2>/dev/null; then
      return
    fi
    rm -rf "$target" 2>/dev/null || true
    warn "mirror ${mirror} failed, trying next..."
  done
  log "all mirrors failed, trying direct GitHub..."
  git clone "$url" "$target"
}

# ---------- A. python env + libs ----------
setup_env() {
  local venv="${SPEECHRL_VENV:-$HOME/.venvs/speechrl}"
  [[ -f "$venv/bin/activate" ]] || die "venv missing: $venv (run ../../scripts/env-setup.sh first)"
  # shellcheck disable=SC1090
  source "$venv/bin/activate"

  if ! command -v aria2c >/dev/null 2>&1; then
    log "Installing aria2 for hfd download support..."
    sudo apt-get install -y aria2 2>/dev/null || warn "aria2 install failed; hfd will not be available"
  fi

  uv pip install -U "huggingface_hub[cli]" hydra-core omegaconf mlflow  --break-system-packages
  uv pip install -U hf_transfer  --break-system-packages 2>/dev/null \
    || warn "hf_transfer install failed; HF downloads will use single-thread mode"
  uv pip install -U jiwer "sacrebleu[ja]" evaluate  --break-system-packages
  uv pip install -U datasets soundfile librosa  --break-system-packages
  uv pip install -U vllm  --break-system-packages
  if prefer_modelscope || prefer_modelscope_dataset; then
    uv pip install -U modelscope  --break-system-packages
  fi
  (cd "$PROJECT_ROOT" && uv pip install -e ../../common -e .)

  log "Speed optimization summary:"
  if command -v aria2c >/dev/null 2>&1; then
    log "  aria2c: installed (hfd multi-thread downloads available)"
  else
    warn "  aria2c: NOT installed (hfd unavailable, downloads will be slower)"
  fi
  if python3 -c "import hf_transfer" 2>/dev/null; then
    log "  hf_transfer: installed (Rust parallel HF downloads available)"
  else
    warn "  hf_transfer: NOT installed (HF single-thread mode only)"
  fi
  if command -v modelscope >/dev/null 2>&1; then
    log "  modelscope: installed (workers=$SPEECHRL_MS_WORKERS)"
  fi
}

# ---------- B. models -> $MODELS ----------
m_nemotron() {
  if skip_if_exists "$MODELS/nemotron3-nano-omni-nvfp4" model; then
    log "skip existing: $MODELS/nemotron3-nano-omni-nvfp4"
    return
  fi
  if prefer_modelscope; then
    ms_model_download nv-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 "$MODELS/nemotron3-nano-omni-nvfp4"
  else
    hf_download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
      --local-dir "$MODELS/nemotron3-nano-omni-nvfp4"
  fi
}

m_qwen3omni() {
  if skip_if_exists "$MODELS/qwen3-omni-30b-a3b-instruct" model; then
    log "skip existing: $MODELS/qwen3-omni-30b-a3b-instruct"
    return
  fi
  if prefer_modelscope; then
    ms_model_download Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound "$MODELS/qwen3-omni-30b-a3b-instruct"
  else
    hf_download Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --local-dir "$MODELS/qwen3-omni-30b-a3b-instruct"
  fi
}

m_moss() {
  if skip_if_exists "$MODELS/moss-audio-8b-instruct" model; then
    log "skip existing: $MODELS/moss-audio-8b-instruct"
    return
  fi
  if prefer_modelscope; then
    ms_model_download openmoss/MOSS-Audio-8B-Instruct "$MODELS/moss-audio-8b-instruct"
  else
    hf_download OpenMOSS-Team/MOSS-Audio-8B-Instruct \
      --local-dir "$MODELS/moss-audio-8b-instruct"
  fi
}

m_minicpm_gguf() {
  if skip_if_exists "$MODELS/minicpm-o-4_5-gguf" model; then
    log "skip existing: $MODELS/minicpm-o-4_5-gguf"
    return
  fi
  if prefer_modelscope; then
    ms_model_download OpenBMB/MiniCPM-o-4_5 "$MODELS/minicpm-o-4_5"
  else
    hf_download openbmb/MiniCPM-o-4_5-gguf \
      --local-dir "$MODELS/minicpm-o-4_5-gguf"
  fi
}

m_baichuan_omni() {
  if skip_if_exists "$MODELS/baichuan-omni-1d5" model; then
    log "skip existing: $MODELS/baichuan-omni-1d5"
    return
  fi
  if prefer_modelscope; then
    ms_model_download baichuan-inc/Baichuan-Omni-1d5 "$MODELS/baichuan-omni-1d5"
  else
    hf_download baichuan-inc/Baichuan-Omni-1d5 \
      --local-dir "$MODELS/baichuan-omni-1d5"
  fi
}

m_kimi_audio() {
  if skip_if_exists "$MODELS/kimi-audio-7b-instruct" model; then
    log "skip existing: $MODELS/kimi-audio-7b-instruct"
    return
  fi
  if prefer_modelscope; then
    warn "Kimi-Audio-7B-Instruct has no verified ModelScope mirror. Falling back to hf-mirror for download."
    SPEECHRL_ALLOW_HF_DOWNLOADS=1 hf_download moonshotai/Kimi-Audio-7B-Instruct \
      --local-dir "$MODELS/kimi-audio-7b-instruct"
  else
    hf_download moonshotai/Kimi-Audio-7B-Instruct \
      --local-dir "$MODELS/kimi-audio-7b-instruct"
  fi
}

m_omni_embed_nemotron() {
  if skip_if_exists "$MODELS/omni-embed-nemotron-3b" model; then
    log "skip existing: $MODELS/omni-embed-nemotron-3b"
    return
  fi
  if prefer_modelscope; then
    ms_model_download nv-community/omni-embed-nemotron-3b "$MODELS/omni-embed-nemotron-3b"
  else
    hf_download nvidia/omni-embed-nemotron-3b \
      --local-dir "$MODELS/omni-embed-nemotron-3b"
  fi
}

models() {
  m_qwen3omni
  m_moss
  m_nemotron
  m_minicpm_gguf
  m_omni_embed_nemotron
}

# ---------- C. datasets -> $DSETS ----------
d_librispeech() {
  if skip_if_exists "$DSETS/librispeech" dataset; then
    log "skip existing: $DSETS/librispeech"
    return
  fi
  if prefer_modelscope_dataset; then
    ms_dataset_download openslr/librispeech_asr "$DSETS/librispeech"
  else
    hf_download openslr/librispeech_asr --repo-type dataset --local-dir "$DSETS/librispeech"
  fi
}

d_mmau_mini() {
  if skip_if_exists "$DSETS/mmau-mini" dataset; then
    log "skip existing: $DSETS/mmau-mini"
    return
  fi
  if prefer_modelscope_dataset; then
    warn "MMAU has no verified exact ModelScope mirror. Falling back to hf-mirror for download."
    SPEECHRL_ALLOW_HF_DOWNLOADS=1 hf_download TwinkStart/MMAU --repo-type dataset --local-dir "$DSETS/mmau-mini"
  else
    hf_download TwinkStart/MMAU --repo-type dataset --local-dir "$DSETS/mmau-mini"
  fi
}

d_mmar() {
  if skip_if_exists "$DSETS/mmar" dataset; then
    log "skip existing: $DSETS/mmar"
    return
  fi
  if prefer_modelscope_dataset; then
    warn "MMAR has no verified exact ModelScope mirror. Falling back to hf-mirror for download."
    SPEECHRL_ALLOW_HF_DOWNLOADS=1 hf_download BoJack/MMAR --repo-type dataset --local-dir "$DSETS/mmar"
  else
    hf_download BoJack/MMAR --repo-type dataset --local-dir "$DSETS/mmar"
  fi
}

d_meld() {
  if skip_if_exists "$DSETS/meld" dataset; then
    log "skip existing: $DSETS/meld"
    return
  fi
  if prefer_modelscope_dataset; then
    warn "MELD has no verified exact ModelScope mirror. Falling back to hf-mirror for download."
    SPEECHRL_ALLOW_HF_DOWNLOADS=1 hf_download declare-lab/MELD --repo-type dataset --local-dir "$DSETS/meld"
  else
    hf_download declare-lab/MELD --repo-type dataset --local-dir "$DSETS/meld"
  fi
}

d_cremad() {
  if skip_if_exists "$DSETS/crema-d" dataset; then
    log "skip existing: $DSETS/crema-d"
    return
  fi
  if prefer_modelscope_dataset; then
    warn "CREMA-D has no verified exact ModelScope mirror. Falling back to hf-mirror for download."
    SPEECHRL_ALLOW_HF_DOWNLOADS=1 hf_download MahiA/CREMA-D --repo-type dataset --local-dir "$DSETS/crema-d"
  else
    hf_download MahiA/CREMA-D --repo-type dataset --local-dir "$DSETS/crema-d"
  fi
}

d_minds14() {
  if skip_if_exists "$DSETS/minds14-xtreme_s" dataset && skip_if_exists "$DSETS/minds14" dataset; then
    log "skip existing: minds14"
    return
  fi
  if prefer_modelscope_dataset; then
    warn "PolyAI/minds14 has no verified exact ModelScope mirror. Downloading google/xtreme_s, which contains the Minds-14 XTREME-S task."
    ms_dataset_download google/xtreme_s "$DSETS/minds14-xtreme_s"
  else
    hf_download PolyAI/minds14 --repo-type dataset --local-dir "$DSETS/minds14"
  fi
}

d_covost2() {
  if skip_if_exists "$DSETS/covost2" dataset; then
    log "skip existing: $DSETS/covost2"
    return
  fi
  warn "facebook/covost2 is a canonical loader-style dataset repo; verify the downloaded files before treating it as a complete offline payload."
  if prefer_modelscope_dataset; then
    ms_dataset_download facebook/covost2 "$DSETS/covost2"
  else
    hf_download facebook/covost2 --repo-type dataset --local-dir "$DSETS/covost2"
  fi
}

d_fleurs() {
  if skip_if_exists "$DSETS/fleurs" dataset; then
    log "skip existing: $DSETS/fleurs"
    return
  fi
  if prefer_modelscope_dataset; then
    ms_dataset_download google/fleurs "$DSETS/fleurs"
  else
    hf_download google/fleurs --repo-type dataset --local-dir "$DSETS/fleurs"
  fi
}

d_voxceleb() {
  if skip_if_exists "$DSETS/voxceleb" dataset; then
    log "skip existing: $DSETS/voxceleb"
    return
  fi
  if prefer_modelscope_dataset; then
    ms_dataset_download juliuscn/voxceleb "$DSETS/voxceleb"
  else
    warn "VoxCeleb1 on HF requires authentication. Falling back to ModelScope (public)."
    ms_dataset_download juliuscn/voxceleb "$DSETS/voxceleb"
  fi
}

d_air_bench() {
  if skip_if_exists "$DSETS/air-bench" dataset; then
    log "skip existing: $DSETS/air-bench"
    return
  fi
  if prefer_modelscope_dataset; then
    ms_dataset_download qfq/AIR-Bench_24.09 "$DSETS/air-bench"
  else
    warn "AIR-Bench has no verified HF repo. Falling back to ModelScope."
    ms_dataset_download qfq/AIR-Bench_24.09 "$DSETS/air-bench"
  fi
}

d_slurp() {
  slurp_link_manifest
  log "SLURP repo target: $REPOS/slurp"
  log "SLURP dataset link: $DSETS/slurp -> $REPOS/slurp"
  git_clone_once https://github.com/pswietojanski/slurp "$REPOS/slurp"
  if is_true "${SPEECHRL_SKIP_SLURP_AUDIO:-0}"; then
    log "skip SLURP audio because SPEECHRL_SKIP_SLURP_AUDIO=1"
  else
    log "Running upstream SLURP audio downloader: $REPOS/slurp/scripts/download_audio.sh"
    log "SLURP audio target directory: $REPOS/slurp/audio"
    log "SLURP upstream download logs:"
    log "  $REPOS/slurp/audio/slurp_real_download.log"
    log "  $REPOS/slurp/audio/slurp_synth_download.log"
    (cd "$REPOS/slurp" && bash scripts/download_audio.sh)
    if [[ -d "$REPOS/slurp/audio" ]]; then
      local audio_size
      audio_size="$(du -sh "$REPOS/slurp/audio" 2>/dev/null || true)"
      [[ -n "$audio_size" ]] && log "SLURP audio directory after download: $audio_size"
    fi
  fi
  ln -sfn "$REPOS/slurp" "$DSETS/slurp"
}

datasets() {
  d_librispeech
  d_mmau_mini
  d_mmar
  d_meld
  d_cremad
  d_minds14
  d_covost2
  d_fleurs
  d_voxceleb
  d_air_bench
  d_slurp
}

# ---------- D. reference repos (git clone, not run) -> $REPOS ----------
clone_refs() {
  local url
  for url in \
    https://github.com/CyberAgentAILab/mbr-for-asr \
    https://github.com/ryysayhi/AudioGenie-Reasoner \
    https://github.com/PRIME-RL/TTRL \
    https://github.com/yafuly/TPO \
    https://github.com/liushiliushi/JitRL \
    https://github.com/asappresearch/slue-toolkit; do
    git_clone_once "$url" "$REPOS/$(basename "$url")"
  done
}

# ---------- E. probes/listing ----------
hf_api_status() {
  local api_kind="$1"
  local repo_id="$2"
  local base="${HF_ENDPOINT%/}"
  curl -L -sS -m "$CURL_TIMEOUT" -o /dev/null -w "%{http_code}" \
    "$base/api/$api_kind/$repo_id" || true
}

hf_dataset_link_manifest() {
  local repo_id="$1"
  local name="$2"
  local include_prefix="${3:-}"
  local min_links="${4:-1}"
  local base="${HF_ENDPOINT%/}"
  local out="$MANIFESTS/$name.links.txt"
  local tmp="$out.tmp"
  local json_tmp="$out.tree.json.tmp"

  log "Building dataset link manifest: $repo_id -> $out"
  curl -L -sS -m "$CURL_TIMEOUT" -o "$json_tmp" \
    "$base/api/datasets/$repo_id/tree/main?recursive=true"
  python3 - "$base" "$repo_id" "$include_prefix" "$json_tmp" >"$tmp" <<'PY'
import json
import sys
from urllib.parse import quote

base, repo_id, include_prefix, json_path = sys.argv[1:5]
try:
    with open(json_path, "r", encoding="utf-8") as handle:
        items = json.load(handle)
except Exception as exc:
    print(f"ERROR: failed to parse tree JSON: {exc}", file=sys.stderr)
    raise SystemExit(2)

if isinstance(items, dict) and "siblings" in items:
    items = items["siblings"]
if not isinstance(items, list):
    print("ERROR: unexpected tree JSON shape", file=sys.stderr)
    raise SystemExit(2)

count = 0
for item in items:
    path = item.get("path") or item.get("rfilename")
    kind = item.get("type")
    if not path:
        continue
    if kind and kind not in {"file", "blob"}:
        continue
    if include_prefix and not path.startswith(include_prefix):
        continue
    encoded_path = quote(path, safe="/")
    print(f"{base}/datasets/{repo_id}/resolve/main/{encoded_path}")
    count += 1

if count == 0:
    raise SystemExit(3)
PY
  rm -f "$json_tmp"
  mv "$tmp" "$out"
  local count
  count="$(wc -l <"$out" | tr -d ' ')"
  [[ "$count" -gt 0 ]] || die "manifest has 0 links: $out"
  if [[ "$count" -lt "$min_links" ]]; then
    warn "manifest has only $count links for $repo_id; this may be a loader/script repo rather than offline data."
  fi
  log "Manifest links: $count ($out)"
}

slurp_link_manifest() {
  local out="$MANIFESTS/slurp.links.txt"
  local tmp="$out.tmp"
  local script_url="https://raw.githubusercontent.com/pswietojanski/slurp/master/scripts/download_audio.sh"

  log "Building dataset link manifest: SLURP audio -> $out"
  log "SLURP upstream script: $script_url"
  curl -L -sS -m "$CURL_TIMEOUT" "$script_url" \
    | grep -Eo 'https://[^[:space:]\\]+' \
    | grep -E 'zenodo\.org/.*/files/.*\.tar\.gz' \
    | sort -u >"$tmp"

  local count
  count="$(wc -l <"$tmp" | tr -d ' ')"
  [[ "$count" -gt 0 ]] || die "manifest has 0 links: $out"
  mv "$tmp" "$out"
  log "Manifest links: $count ($out)"
  log "SLURP audio download URLs:"
  log_numbered_file "$out"
}

probe_hf() {
  local repo code
  log "Probing Hugging Face-compatible endpoint: $HF_ENDPOINT"
  printf '%-8s %-58s %s\n' "type" "repo" "http"
  for repo in \
    nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    Qwen/Qwen3-Omni-30B-A3B-Instruct \
    OpenMOSS-Team/MOSS-Audio-8B-Instruct \
    openbmb/MiniCPM-o-4_5-gguf \
    baichuan-inc/Baichuan-Omni-1d5 \
    moonshotai/Kimi-Audio-7B-Instruct; do
    code="$(hf_api_status models "$repo")"
    printf '%-8s %-58s %s\n' "model" "$repo" "$code"
  done
  for repo in \
    openslr/librispeech_asr \
    TwinkStart/MMAU \
    BoJack/MMAR \
    declare-lab/MELD \
    MahiA/CREMA-D \
    PolyAI/minds14 \
    facebook/covost2 \
    google/fleurs \
    ProgramV/VoxCeleb1; do
    code="$(hf_api_status datasets "$repo")"
    printf '%-8s %-58s %s\n' "dataset" "$repo" "$code"
  done
}

modelscope_page_status() {
  local repo_id="$1"
  curl -L -sS -m "$CURL_TIMEOUT" -o /dev/null -w "%{http_code}" \
    "https://modelscope.cn/models/$repo_id" || true
}

modelscope_dataset_status() {
  local repo_id="$1"
  curl -L -sS -m "$CURL_TIMEOUT" -o /dev/null -w "%{http_code}" \
    "https://www.modelscope.cn/api/v1/datasets/$repo_id" || true
}

check_modelscope_asset() {
  local repo_id="$1"
  local code
  code="$(modelscope_page_status "$repo_id")"
  printf '%-12s %-58s %s\n' "modelscope" "$repo_id" "$code"
}

check_modelscope_dataset() {
  local repo_id="$1"
  local code
  code="$(modelscope_dataset_status "$repo_id")"
  printf '%-12s %-58s %s\n' "ms-dataset" "$repo_id" "$code"
}

check_git_ref() {
  local url="$1"
  if git ls-remote --exit-code "$(git_url "$url")" HEAD >/dev/null 2>&1; then
    printf '%-12s %-58s %s\n' "git" "$url" "ok"
  else
    printf '%-12s %-58s %s\n' "git" "$url" "check-failed"
  fi
}

check_dataset_link_manifests() {
  log "HF-only dataset directory link manifests"
  hf_dataset_link_manifest TwinkStart/MMAU mmau-mini
  hf_dataset_link_manifest BoJack/MMAR mmar
  hf_dataset_link_manifest declare-lab/MELD meld
  hf_dataset_link_manifest MahiA/CREMA-D crema-d
  if prefer_modelscope_dataset; then
    warn "Skipping PolyAI/minds14 HF manifest because mainland mode uses ModelScope google/xtreme_s as the Minds-14 substitute."
  else
    hf_dataset_link_manifest PolyAI/minds14 minds14
  fi
  slurp_link_manifest
}

check_assets() {
  log "DATA_ROOT=$DATA_ROOT"
  log "Checking addresses only; no model/dataset files will be downloaded."

  check_dataset_link_manifests

  log "Hugging Face-compatible API probes (no hf download dry-run)"
  probe_hf

  log "ModelScope page checks"
  printf '%-12s %-58s %s\n' "source" "repo" "http"
  check_modelscope_asset nv-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
  check_modelscope_asset Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound
  check_modelscope_asset openmoss/MOSS-Audio-8B-Instruct
  check_modelscope_asset OpenBMB/MiniCPM-o-4_5
  check_modelscope_asset baichuan-inc/Baichuan-Omni-1d5
  check_modelscope_asset nv-community/omni-embed-nemotron-3b
  
  log "ModelScope dataset checks"
  printf '%-12s %-58s %s\n' "source" "repo" "http"
  check_modelscope_dataset openslr/librispeech_asr
  check_modelscope_dataset google/fleurs
  check_modelscope_dataset facebook/covost2
  check_modelscope_dataset google/xtreme_s
  check_modelscope_dataset juliuscn/voxceleb
  check_modelscope_dataset qfq/AIR-Bench_24.09

  log "Git reference repo checks"
  printf '%-12s %-58s %s\n' "source" "repo" "status"
  check_git_ref https://github.com/pswietojanski/slurp
  check_git_ref https://github.com/CyberAgentAILab/mbr-for-asr
  check_git_ref https://github.com/ryysayhi/AudioGenie-Reasoner
  check_git_ref https://github.com/PRIME-RL/TTRL
  check_git_ref https://github.com/yafuly/TPO
  check_git_ref https://github.com/liushiliushi/JitRL
  check_git_ref https://github.com/asappresearch/slue-toolkit

  log "SLURP link manifest was printed above and saved to: $MANIFESTS/slurp.links.txt"
}

list_assets() {
  cat <<EOF

Assets and mainland+VPN notes (current date: $(date +%Y-%m-%d)):

Models (default set, 24GB VRAM friendly):
  m_qwen3omni      ModelScope uses INT4-AutoRound (Intel/) for 24GB VRAM;
                   HF uses BF16 original (Qwen/). The INT4 version is the
                   recommended choice for RTX 5090 24GB.
  m_moss           Fixed to OpenMOSS-Team/MOSS-Audio-8B-Instruct on HF and
                   openmoss/MOSS-Audio-8B-Instruct on ModelScope.
  m_nemotron       ModelScope nv-community/ mirror; HF/hf-mirror as fallback.
  m_minicpm_gguf   HF/hf-mirror exact GGUF repo. ModelScope has MiniCPM-o-4_5
                   variants (not GGUF format).

Models (optional, run individually):
  m_baichuan_omni  baichuan-inc/Baichuan-Omni-1d5 on ModelScope and HF.
                   7B omni model (2025), good for LoRA RL experiments.
  m_kimi_audio     moonshotai/Kimi-Audio-7B-Instruct on HF only.
                   7B audio model (2025), no ModelScope mirror.
  m_omni_embed_nemotron
                   nv-community/omni-embed-nemotron-3b on ModelScope
                   (HF: nvidia/omni-embed-nemotron-3b). NVIDIA multimodal
                   contrastive embedding model (~4.7B, dim 2048) for W4
                   omni-embedding retrieval/contrastive RL. NVIDIA OneWay
                   Noncommercial license.

Datasets:
  d_librispeech    ModelScope exact repo in mainland mode; downloads the full dataset
  d_mmau_mini      No ModelScope mirror; falls back to hf-mirror in CN mode
  d_mmar           No ModelScope mirror; falls back to hf-mirror in CN mode
  d_meld           No ModelScope mirror; falls back to hf-mirror in CN mode
  d_cremad         No ModelScope mirror; falls back to hf-mirror in CN mode
  d_minds14        ModelScope mainland substitute is google/xtreme_s Minds-14 task;
                   exact PolyAI/minds14 remains HF/hf-mirror only
  d_covost2        ModelScope exact repo in mainland mode, but verify payload layout
  d_fleurs         ModelScope exact repo in mainland mode
  d_voxceleb       ModelScope juliuscn/voxceleb in mainland mode; HF ProgramV/VoxCeleb1
  d_air_bench      ModelScope qfq/AIR-Bench_24.09; speaker verification benchmark
  d_slurp          GitHub + Zenodo audio. If mirrors fail, set
                   SPEECHRL_SKIP_SLURP_AUDIO=1 to clone metadata only.

Reference repos:
  refs             GitHub only. git_clone_once auto-tries mirrors from
                   SPEECHRL_GH_MIRRORS (default: ghfast.top,ghproxy.cc).

EOF
}

usage() {
  cat <<EOF
Usage:
  bash scripts/wave0_fetch.sh help
  bash scripts/wave0_fetch.sh list
  bash scripts/wave0_fetch.sh probe
  bash scripts/wave0_fetch.sh check
  bash scripts/wave0_fetch.sh setup-env
  bash scripts/wave0_fetch.sh m_qwen3omni d_librispeech
  bash scripts/wave0_fetch.sh models
  bash scripts/wave0_fetch.sh datasets
  bash scripts/wave0_fetch.sh refs
  bash scripts/wave0_fetch.sh all

China mainland + VPN mode:
  SPEECHRL_CN_MIRROR=1 bash scripts/wave0_fetch.sh setup-env
  SPEECHRL_CN_MIRROR=1 bash scripts/wave0_fetch.sh m_qwen3omni m_moss d_librispeech

Environment:
  SPEECHRL_DATA_DIR        default: <workspace>/speechrl-data on D drive
  SPEECHRL_VENV            default: $HOME/.venvs/speechrl
  SPEECHRL_CN_MIRROR       default: 1; use hf-mirror + ModelScope-friendly mode
  SPEECHRL_MODEL_SOURCE    auto | hf | modelscope (default: auto)
  SPEECHRL_DATASET_SOURCE  auto | hf | modelscope (default: auto)
  SPEECHRL_HF_ENDPOINT     script-specific HF-compatible endpoint override
  HF_ENDPOINT              ignored in CN mode when set to the official HF default
  HF_HUB_DISABLE_XET       default: 1 in CN mode, 0 otherwise
  SPEECHRL_ALLOW_HF_DOWNLOADS
                           default: 0 in CN mode, 1 otherwise
  SPEECHRL_SKIP_HF_DRY_RUN default: 1 in CN mode, 0 otherwise
  SPEECHRL_HF_MAX_RETRIES  default: 3; retry count for HF downloads
  SPEECHRL_MS_MAX_RETRIES  default: 3; retry count for ModelScope downloads
  SPEECHRL_HFD_ENABLED     auto | 1 | 0; use hfd (aria2) downloader from hf-mirror
  SPEECHRL_HFD_THREADS     default: 16; aria2 thread count for hfd downloads
  SPEECHRL_MS_WORKERS      default: 16; ModelScope concurrent download workers
  SPEECHRL_HF_TRANSFER     auto | 1 | 0; enable hf_transfer (Rust parallel)
  SPEECHRL_SKIP_EXISTING   default: 1; skip download if model/dataset dir exists
  SPEECHRL_GH_MIRRORS      default: ghfast.top,ghproxy.cc; comma-separated GitHub mirrors
  HF_CLI                   default: hf
  GIT_CLONE_PREFIX         optional prefix for GitHub clone URLs

Check mode writes small link manifests to:
  $MANIFESTS

No arguments prints this help and downloads nothing.
EOF
}

run_target() {
  case "$1" in
    help|-h|--help) usage; list_assets ;;
    list) list_assets ;;
    probe) probe_hf ;;
    check) check_assets ;;
    setup-env|setup_env) setup_env ;;
    models) models ;;
    datasets) datasets ;;
    refs|clone_refs) clone_refs ;;
    all) setup_env; models; datasets; clone_refs ;;
    m_nemotron|m_qwen3omni|m_moss|m_minicpm_gguf) "$1" ;;
    m_baichuan_omni|m_kimi_audio|m_omni_embed_nemotron) "$1" ;;
    d_librispeech|d_mmau_mini|d_mmar|d_meld|d_cremad|d_minds14|d_covost2|d_fleurs|d_voxceleb|d_air_bench|d_slurp) "$1" ;;
    *) die "unknown target: $1 (run: bash scripts/wave0_fetch.sh help)" ;;
  esac
}

main() {
  validate_model_source
  if [[ "${EUID:-$(id -u)}" -eq 0 && -z "${SPEECHRL_DATA_DIR:-}" ]]; then
    warn "running as root; DATA_ROOT still defaults to the D-drive workspace: $DATA_ROOT."
  fi
  if [[ $# -eq 0 ]]; then
    usage
    list_assets
    exit 0
  fi
  local target
  for target in "$@"; do
    run_target "$target"
  done
}

main "$@"
